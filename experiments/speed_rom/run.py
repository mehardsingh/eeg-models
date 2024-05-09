from dataset import KFoldDatasets
import json
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

from dataset import KFoldDatasets

sys.path.append("experiments/utils")
from training_utils import Config, get_num_params, get_model_class

def train(model_name, model_config_fp, exp_config_fp):
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
        train_ds, val_ds, test_ds = kf_datasets[fold_idx]
        print(f"=== Fold: {fold_idx + 1} / {len(kf_datasets)} ===")

        train_single_fold(
            model=model, 
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            training_config=training_config
        )

def train_single_fold(model, train_ds, val_ds, test_ds, training_config):
    # "lr": 0.0001,
    # "optimizer": "adagrad",
    # "lr_decay": "plateau",
    # "batch_size": 16

    lr = training_config.lr
    batch_size = training_config.batch_size
    
    if training_config.optimizer == "adagrad":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name")
    
    if training_config.lr_decay == "plateau":
        scheduler = ReduceLROnPlateau(optimizer)
    else:
        raise ValueError("Invalid lr decay name")


train(
    model_name="gtn", 
    model_config_fp="experiments/speed_rom/configs/model_configs.json",
    exp_config_fp="experiments/speed_rom/configs/exp1_config.json"
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