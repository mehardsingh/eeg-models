from dataset import KFoldDatasets
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import sys

from dataset import KFoldDatasets

sys.path.append("experiments/utils")
from training_utils import Config, get_num_params, get_model_class

def train(model_name, model_config_fp, exp_config_fp, results_dir):
    print(f"=== Loading all dataset folds ===")   
    with open(exp_config_fp, mode="r") as config_f:
        exp_config = json.load(config_f)
        exp_config = Config(**exp_config)

    kf_datasets = KFoldDatasets(
        data_dir=exp_config.data_dir,
        k=exp_config.k,
        val_percent=exp_config.val_percent,
        motion_type=exp_config.motion_type
    )

    model_class = get_model_class(model_name)
    with open(model_config_fp, mode="r") as config_f:
        model_config = json.load(config_f)[model_name]
        architecture_config = Config(**model_config["architecture"])
        training_config = Config(**model_config["training"])

    model = model_class(architecture_config)
    print(f"=== Model Name: {model_name} ===")    
    num_params = get_num_params(model)
    print(f"=== Parameter Count: {num_params} ===")   

    for fold_idx in range(len(kf_datasets)):
        fold_results_dir = os.path.join(results_dir, model_name, f"fold_{fold_idx+1}")
        train_ds, val_ds, test_ds = kf_datasets[fold_idx]
        print(f"=== Fold: {fold_idx + 1} / {len(kf_datasets)} ===")

        train_single_fold(
            model=model, 
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            training_config=training_config,
            fold_results_dir=fold_results_dir
        )

def train_single_fold(model, train_ds, val_ds, test_ds, training_config, fold_results_dir):
    num_epochs = training_config.num_epochs
    lr = training_config.lr
    batch_size = training_config.batch_size
    wd = training_config.wd

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    if training_config.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    elif training_config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError("Invalid optimizer name")
    
    if training_config.lr_decay == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer)
    elif training_config.lr_decay == "StepLR":
        scheduler = StepLR(optimizer, step_size=training_config.step_size, gamma=training_config.lr_decay_rate)
    elif training_config.lr_decay == "None":
        scheduler = None
    else:
        raise ValueError("Invalid lr decay name")
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    train_losses = list()
    val_losses = list()
    val_accs = list()
    val_f1s = list()

    for epoch_idx in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            epoch_data = batch["epoch_data"].float().to(device) # B x T x C
            labels = batch["labels"].to(device)

            outputs = model(epoch_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_dl)

        model.eval()
        running_val_loss = 0.0
        predictions = list()
        targets = list()
        with torch.no_grad():
            for batch in val_dl:
                epoch_data = batch["epoch_data"].float().to(device)
                labels = batch["labels"].to(device)

                outputs = model(epoch_data)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.append(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_dl)

        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        val_accuracy = accuracy_score(targets, predictions)
        val_f1 = f1_score(targets, predictions, average='micro')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)
        val_f1s.append(val_f1)

        save_progress(model, train_losses, val_losses, val_accs, val_f1s, fold_results_dir)

        if training_config.lr_decay == "ReduceLROnPlateau":
            scheduler.step(val_accuracy)
        elif not scheduler == None:
            scheduler.step()

def save_progress(model, train_losses, val_losses, val_accs, val_f1s, fold_results_dir):
    directory = Path(fold_results_dir)
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(fold_results_dir, "model.pt"))

    progress_dict = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "val_f1": val_f1s
    }
    progress_df = pd.DataFrame(progress_dict)
    progress_df.to_csv(os.path.join(fold_results_dir, "progress.csv"), index=False)

# train(
#     model_name="gtn", 
#     model_config_fp="experiments/speed_rom/configs/model_configs.json",
#     exp_config_fp="experiments/speed_rom/configs/exp1_config.json",
#     results_dir = "results/speed_rom/"
# )
    
train(
    model_name="fbcnet", 
    model_config_fp="experiments/speed_rom/configs/model_configs.json",
    exp_config_fp="experiments/speed_rom/configs/exp1_config.json",
    results_dir = "results/speed_rom/"
)

    
    



# kf_ds = KFoldDatasets(
#     data_dir="data/speed_rom",
#     motion_type="Disc",
#     k=4,
#     val_percent=0.15
# )

# train, val, test = kf_ds[0]
# print(len(train))
# print(len(val))
# print(len(test))