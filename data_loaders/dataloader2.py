# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import glob
import os

import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TrainDataset(Dataset):
    def __init__(
        self,
        dataset,
        mean,
        std,
        motions,
        sparses,
        input_motion_length=196,
        train_dataset_repeat_times=1,
        no_normalization=False,
    ):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.motions = motions
        self.sparses = sparses
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization

        self.motions = motions
        self.sparses = sparses

        self.input_motion_length = input_motion_length

    def __len__(self):
        return len(self.motions) * self.train_dataset_repeat_times

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx):
        motion = self.motions[idx % len(self.motions)]
        sparse = self.sparses[idx % len(self.motions)]
        seqlen = motion.shape[0]

        if seqlen <= self.input_motion_length:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]
        motion = motion[idx : idx + self.input_motion_length]
        sparse = sparse[idx : idx + self.input_motion_length]

        # Normalization
        if not self.no_normalization:
            motion = (motion - self.mean) / (self.std + 1e-8)

        return motion.float(), sparse.float()


class TestDataset(Dataset):
    def __init__(
        self,
        name,
        mean,
        std,
        all_info,
        filename_list,
        normalize_sparse="none",
    ):
        self.name = name
        self.mean = mean
        self.std = std
        self.filename_list = filename_list
        self.normalize_sparse = normalize_sparse

        self.motions = []
        self.sparses = []
        self.body_params = []
        self.head_motion = []
        for i in all_info:
            self.motions.append(i["rotation_local_full_gt_list"])
            self.sparses.append(i["hmd_position_global_full_gt_list"])
            self.body_params.append(i["body_parms_list"])
            self.head_motion.append(i["head_global_trans_list"])

    def __len__(self):
        return len(self.motions)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx):
        motion = self.motions[idx]
        sparse = self.sparses[idx]
        body_param = self.body_params[idx]
        head_motion = self.head_motion[idx]
        filename = self.filename_list[idx]

        return (
            motion,
            sparse.unsqueeze(0),
            body_param,
            head_motion,
            filename,
        )


def get_mean_std_path(dataset):
    return dataset + "_mean.pt", dataset + "_std.pt"


def get_motion(motion_list):
    # rotation_local_full_gt_list : 6d rotation parameters
    # hmd_position_global_full_gt_list : 3 joints(head, hands) 6d rotation/6d rotation velocity/global translation/global translation velocity
    motions = [i["rotation_local_full_gt_list"] for i in motion_list]
    sparses = [i["hmd_position_global_full_gt_list"] for i in motion_list]
    return motions, sparses


def get_path(dataset_path, split, protocol):
    data_list_path = []
    assert split in ["train", "test"]
    assert protocol in ['1_small_dataset', '2_BMLtest', '2_CMUtest', '2_MPItest', '2_full_dataset', 'real',
                        'randomsplit_0']
    if protocol == '1_small_dataset':
        if split == "train":
            dataset = ['CMU_train', 'BioMotionLab_NTroje_train', 'MPI_HDM05_train']
        elif split == "test":
            dataset = ['CMU_test', 'BioMotionLab_NTroje_test', 'MPI_HDM05_test']
    elif protocol == '2_full_dataset':
        if split == "train":
            dataset = ['MPI_HDM05_train', 'MPI_HDM05_test', 'BioMotionLab_NTroje_train', 'BioMotionLab_NTroje_test',
                       'CMU_train', 'CMU_test', 'ACCAD', 'BMLmovi', 'EKUT', 'Eyes_Japan_Dataset', 'KIT',
                       'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture']
        elif split == "test":
            dataset = ['HumanEva', 'Transitions_mocap']
    elif protocol == 'real':
        dataset = ['real']
    elif protocol == 'randomsplit_0':
        # specific method
        if split == 'train':
            dataset_file_name = 'prepare_data/data_split/train_full_rand0.txt'
        else:
            dataset_file_name = 'prepare_data/data_split/test_full_rand0.txt'
        print(f"{split} using {dataset_file_name}")
        with open(dataset_file_name, 'r') as f:
            for line in f:
                path = line.strip()
                data_list_path.append(path)
        return data_list_path
    else:
        return

    print(f"{split} using {dataset}")
    for d in dataset:
        d = os.path.join(dataset_path, d)
        if os.path.isdir(d):
            files = glob.glob(d + "/" + "/*pt")
            data_list_path.extend(files)
    return data_list_path

def load_data(dataset,dataset_path, split, protocol, **kwargs):
    if split == "test":
        motion_list = get_path(dataset_path, split, protocol)
        mean_path, std_path = get_mean_std_path(dataset)
        filename_list = [
            "-".join([i.split("/")[-3], i.split("/")[-1]]).split(".")[0]
            for i in motion_list
        ]
        motion_list = [torch.load(i) for i in tqdm(motion_list)]
        mean = torch.load(os.path.join(dataset_path, mean_path))
        print('mean_path', os.path.join(dataset_path, mean_path))
        std = torch.load(os.path.join(dataset_path, std_path))
        return filename_list, motion_list, mean, std

    assert split == "train"
    assert ("input_motion_length" in kwargs), "Please specify the input_motion_length"

    motion_list = get_path(dataset_path, split, protocol)
    mean_path, std_path = get_mean_std_path(dataset)
    input_motion_length = kwargs["input_motion_length"]
    motion_list = [torch.load(i) for i in tqdm(motion_list)]

    motions, sparses = get_motion(motion_list)

    new_motions = []
    new_sparses = []
    for idx, motion in enumerate(motions):
        if motion.shape[0] < input_motion_length:  # Arbitrary choice
            continue
        new_sparses.append(sparses[idx])
        new_motions.append(motions[idx])

    if os.path.exists(os.path.join(dataset_path, mean_path)):
        mean = torch.load(os.path.join(dataset_path, mean_path))
        std = torch.load(os.path.join(dataset_path, std_path))
    else:
        tmp_data_list = torch.cat(new_motions, dim=0)
        mean = tmp_data_list.mean(axis=0).float()
        std = tmp_data_list.std(axis=0).float()
        with open(os.path.join(dataset_path, mean_path), "wb") as f:
            torch.save(mean, f)
        with open(os.path.join(dataset_path, std_path), "wb") as f:
            torch.save(std, f)

    return new_motions, new_sparses, mean, std, motion_list



def get_dataloader(
    dataset,
    split,
    batch_size,
    num_workers=32,
):

    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader
