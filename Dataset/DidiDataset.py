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
    Unix_time_20161001 = 1475276400
    seconds_per_day = 86400
    Unix_time_20161101 = 1477958400

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, dataset_root: str, traj_length: int, feature_mean: List[float], feature_std: List[float]):
        """
        滴滴 Trajectory Dataset, contains 成都 and 西安 datasets

        The processed dataset folder contains many gps_YYYYMMDD.pt files
        They record trajectories of all orders in that day, formatted as:

        [
            (driver_id, order_id, lon_lat_tensor, time_tensor),
            (driver_id, order_id, lon_lat_tensor, time_tensor),
            ...
        ]

        lon_lat_tensor: (N, 2) torch.float64
        time_tensor: (N,) torch.long

        :param dataset_root: The path to the folder containing gps_YYYYMMDD.pt files
        :param traj_length: shorter: not included, longer: included but cropped
        :param feature_mean: The mean of time&lon&lat
        :param feature_std: The std of time&lon&lat
        """

        if "oct" in dataset_root:
            self.time_shift = self.Unix_time_20161001
        elif "nov" in dataset_root:
            self.time_shift = self.Unix_time_20161101
        else:
            raise ValueError(f"dataset_root should contain 'oct' or 'nov', but got {dataset_root}")

        self.traj_mean = torch.tensor(feature_mean, dtype=torch.float32, device=self.device).view(1, 3)
        self.traj_std = torch.tensor(feature_std, dtype=torch.float32, device=self.device).view(1, 3)

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
            raw_dataset_part = torch.load(file_path)

            for (_, _, lon_lat_tensor, time_tensor) in raw_dataset_part:
                if lon_lat_tensor.shape[0] >= self.traj_length:
                    self.dataset_part.append((
                        lon_lat_tensor[:self.traj_length].to(torch.float32),
                        ((time_tensor[:self.traj_length] - self.time_shift) / 60).to(torch.float32),
                    ))

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
        :return: lon_lat: (2, N), attr: (3,), times: (N,)
        """
        lon_lat, times = self.dataset_part[index]
        times = (times.to(self.device) - self.traj_mean[:, 0]) / self.traj_std[:, 0]

        traj_length = torch.sqrt(torch.sum((lon_lat[1:] - lon_lat[:-1]) ** 2, dim=1)).sum()   # the length of the trajectory
        traj_length = traj_length / self.traj_std[:, 1:].sum()

        avg_move_distance = traj_length / (len(lon_lat) - 1)
        departure_time = times[0]     # the departure time of the trajectory
        attr = torch.tensor([traj_length, avg_move_distance, departure_time], dtype=torch.float32, device=self.device)

        lon_lat = (lon_lat.to(self.device) - self.traj_mean[:, 1:]) / self.traj_std[:, 1:]
        lon_lat = lon_lat.transpose(0, 1).to(torch.float32).contiguous()    # (N, 2) -> (2, N)
        return lon_lat, attr, times.to(torch.float32)
        

    @property
    def n_files(self) -> int:
        return len(self.file_paths)
    

def collectFunc(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param batch: A list of trajectories and attributes
    :return: lon_lat: (B, 2, N), attr: (B, 3), times: (B, N)
    """
    traj_list, attr_list, time_list = zip(*batch)
    return torch.stack(traj_list, dim=0), torch.stack(attr_list, dim=0), torch.stack(time_list, dim=0)


if __name__ == "__main__":
    dataset = DidiTrajectoryDataset('E:/Data/Didi/chengdu/nov', traj_length=120, feature_mean=[0, 0, 0], feature_std=[1, 1, 1])
    dataset.loadNextFiles(1)
    print(len(dataset))
    print(dataset[0][0])
    print(dataset[1][1])
    print(dataset[2][2])