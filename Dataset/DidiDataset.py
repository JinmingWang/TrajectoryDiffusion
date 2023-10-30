from typing import Any
import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
from typing import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as trans_func
import pickle
import random


class DidiTrajectoryDataset(data.Dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, dataset_root: str, traj_length: int, lat_mean: float, lat_std: float, lon_mean: float, lon_std: float):
        """
        滴滴 Trajectory Dataset, contains 成都 and 西安 datasets

        The processed dataset folder contains many gps_YYYYMMDD.pt files
        They record trajectories of all orders in that day, formatted as:

        {
            order_0: (driver_id, trajectory),
            order_1: (driver_id, trajectory),
            ...
        }

        trajectory is torch.Tensor of shape (N, 3), 3 features are (time, lat, lon)


        :param dataset_root: The path to the folder containing gps_YYYYMMDD.pt files
        :param traj_length: shorter: not included, longer: included but cropped
        :param lat_mean: The mean of latitude
        :param lat_std: The std of latitude
        :param lon_mean: The mean of longitude
        :param lon_std: The std of longitude
        """

        self.traj_mean = torch.tensor([lat_mean, lon_mean], dtype=torch.float32, device=self.device).view(2, 1)
        self.traj_std = torch.tensor([lat_std, lon_std], dtype=torch.float32, device=self.device).view(2, 1)

        self.dataset_root = dataset_root
        self.file_paths = [os.path.join(dataset_root, file) for file in os.listdir(dataset_root) if file.endswith('.pt')]
        self.part_idx = 0

        self.traj_length = traj_length

        self.dataset_part = []


    def loadNextFiles(self, load_n: int) -> bool:
        """
        Load next n files into memory
        :return: True if there are still files to load, False if all files are loaded
        """
        # First clear the previous dataset_part
        self.dataset_part = []
        for i in range(load_n):
            if self.part_idx >= len(self.file_paths):
                self.part_idx = 0
                break
            file_path = self.file_paths[self.part_idx]
            print(f'Loading {file_path}')
            with open(file_path, 'rb') as f:
                raw_dataset_part = torch.load(f)

            for v in raw_dataset_part.values():
                driver_id, traj = v
                if len(traj) >= self.traj_length:
                    self.dataset_part.append(traj[:self.traj_length])   # Crop

            self.part_idx += 1

        return len(self.dataset_part) > 0
    

    def shuffleAllFiles(self):
        if self.part_idx != 0:
            raise RuntimeError('You should call loadNextFiles() to load all files before shuffling')
        random.shuffle(self.file_paths)


    def __len__(self):
        return len(self.dataset_part)
    

    def __getitem__(self, index: Any) -> Any:
        """
        :param index: The index of the trajectory in dataset_part
        :return: A trajectory of shape (2, N), 2 features are (lat, lon)
        """
        traj = self.dataset_part[index][:, 1:].to(self.device)     # exclude time feature
        travel_distance = torch.sqrt(torch.sum((traj[1:] - traj[:-1]) ** 2, dim=1)).sum()   # the length of the trajectory
        avg_move_distance = travel_distance / (len(traj) - 1)
        departure_time = self.dataset_part[index][0, 0]     # the departure time of the trajectory
        attr = torch.tensor([travel_distance, avg_move_distance, departure_time], dtype=torch.float32, device=self.device)
        traj = traj.transpose(0, 1).contiguous()
        return (traj - self.traj_mean) / self.traj_std, attr
        

    @property
    def n_files(self) -> int:
        return len(self.file_paths)
    

def collectFunc(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """
    :param batch: A list of trajectories and attributes
    :return: A tensor of shape (B, 2, N) and a tensor of shape (B, 3)
    """
    traj_list, attr_list = zip(*batch)
    return torch.stack(traj_list, dim=0), torch.stack(attr_list, dim=0)


if __name__ == "__main__":
    dataset = DidiTrajectoryDataset('E:/Data/Didi/xian/nov', traj_length=120)
    dataset.loadNextFiles(1)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[1][1].shape)
    print(dataset[2][0].shape)