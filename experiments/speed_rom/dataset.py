import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset, random_split

class KFoldDatasets():
    def __init__(self, data_dir, motion_type, k, val_percent, seed=42):
        self.seed = seed
        self.val_percent = val_percent

        epoch_data, participants, groups, labels = self.load_all(data_dir, motion_type)
        epoch_data, participants, groups, labels = self.shuffle_arrays_in_order(epoch_data, participants, groups, labels)

        participant_data = self.group_by_participants(epoch_data, participants, groups, labels)
        folds = self.create_folds(participant_data, k)
        self.datasets = self.generate_datasets(folds)

    def load_file(self, filepath):
        df = pd.read_csv(filepath)

        m1 = np.array(df["M1"])
        sma = np.array(df["SMA"])
        lpfc = np.array(df["lPFC"])
        rpfc = np.array(df["rPFC"])

        all_data = np.array([m1, sma, lpfc, rpfc]).T
        epoch_data = np.reshape(all_data, (-1, 501, 4))

        participants = np.array(df["Participant"])
        participants = np.reshape(participants, (-1, 501))

        groups = np.array(df["GroupID"])
        groups = np.reshape(groups, (-1, 501))

        labels = np.array(df["ML Label"])
        mapping_dict = {"ROM": 0, "Speed": 1}
        labels = np.vectorize(mapping_dict.get)(labels)
        labels = np.reshape(labels, (-1, 501))

        return epoch_data, participants, groups, labels

    def load_all(self, data_dir, motion_type):
        epoch_data = list()
        participants = list()
        groups = list()
        labels = list()

        if motion_type == "both":
            filepaths = os.listdir(data_dir)
        else:
            filepaths = os.listdir(data_dir)
            filepaths = [fname for fname in filepaths if motion_type in fname]

        for filepath in tqdm(filepaths, desc="Loading data"):
            f_epoch_data, f_participants, f_groups, f_labels = self.load_file(os.path.join(data_dir, filepath))
            
            epoch_data.append(f_epoch_data)
            participants.append(f_participants)
            groups.append(f_groups)
            labels.append(f_labels)

        epoch_data = np.concatenate(epoch_data, axis=0)
        participants = np.concatenate(participants, axis=0)[:,0]
        groups = np.concatenate(groups, axis=0)[:,0]
        labels = np.concatenate(labels, axis=0)[:,0]

        return epoch_data, participants, groups, labels

    def shuffle_arrays_in_order(self, *arrays):
        # Generate shuffled indices
        num_samples = len(arrays[0])
        shuffled_indices = np.random.default_rng(seed=self.seed).permutation(num_samples)
        
        # Use shuffled indices to shuffle all arrays in the same order
        shuffled_arrays = [arr[shuffled_indices] for arr in arrays]
        
        return shuffled_arrays

    def group_by_participants(self, epoch_data, participants, groups, labels):
        participant_data = dict()
        for i in range(participants.shape[0]):
            participant_id = participants[i]
            if not participant_id in participant_data:
                participant_data[participant_id] = list()
            
            participant_data[participant_id].append({
                "epoch_data": epoch_data[i],
                "participant": participant_id,
                "groups": groups[i],
                "labels": labels[i],
            })

        return participant_data
    
    def create_folds(self, participant_data, k):
        folds = list()
        for i in range(k):
            folds.append(list())

        for i, (_, data) in enumerate(participant_data.items()):
            folds[i % k] += data

        return folds

    def generate_datasets(self, folds):
        datasets = list()
        for fold in folds:
            datasets.append(SpeedROMDataset(fold))
        return datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        test_ds = self.datasets[index]
        train_ds_folds = list()

        for i in range(len(self.datasets)):
            if not i == index:
                train_ds_folds.append(self.datasets[i])

        train_ds = ConcatDataset(train_ds_folds)
        train_size = int(len(train_ds) * (1 - self.val_percent))
        val_size = len(train_ds) - train_size
        train_ds, val_ds = random_split(train_ds, [train_size, val_size])

        return train_ds, val_ds, test_ds

class SpeedROMDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]