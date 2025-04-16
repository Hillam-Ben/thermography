import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import torchvision.transforms.functional as TF
import os
import pandas as pd

"""
Custom Image dataset for image-to-image translation using pytorch for handling .png and .csv files

This is a custom dataset class for pytorch that reads images (.png) and corresponding target data (.csv) from a directory and returns them as tensors.
It also has the ability to apply transformations to the images before returning them.
"""
class ImageDataset3to1(Dataset):
    """Dataset class for handling .png and .csv files"""
    in_channels = 3
    out_channels = 1

    def __init__(self, input_globbing_pattern: str, target_globbing_pattern: str, transform: callable = None) -> None:
        self.input_globbing_pattern = input_globbing_pattern
        self.target_globbing_pattern = target_globbing_pattern
        self.transform = transform

        self.images = sorted(glob(os.path.join(input_globbing_pattern, '*.png'), recursive=True))
        self.targets = sorted(glob(os.path.join(target_globbing_pattern, '*.csv'), recursive=True))

        assert len(self.images) == len(self.targets), "Number of images and targets must be equal, is {} and {}".format(len(self.images), len(self.targets))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> tuple:
        input_tensor = Image.open(self.images[idx]).convert("RGB" if self.in_channels == 3 else "L")

        # Read CSV target file
        target = pd.read_csv(self.targets[idx], header=None, delimiter=' ').values.astype('float32')
        target = target[0:240, 0:240]
        target_tensor = torch.tensor(target).unsqueeze(0)

        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = TF.to_tensor(input_tensor)

        if not isinstance(target_tensor, torch.Tensor):
            target_tensor = torch.tensor(target_tensor, dtype=torch.float32)

        return input_tensor, target_tensor, idx

    def get_filenames(self, idx, default=None) -> str:
        return self.images[idx]
