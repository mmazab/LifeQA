#!/usr/bin/env python
"""Script to extract ResNet features from video frames."""
import os
from typing import Callable

import numpy as np
import PIL.Image
import torch.utils.data
import torch.nn as nn
import torchvision


class LifeQaDataset(torch.utils.data.Dataset):
    """Dataset of LifeQA video frames with ``PIL.Image.Image`` type."""

    FRAMES_FILE_DIR = 'data/frames'

    def __init__(self, transform: Callable=None) -> None:
        self.transform = transform

        self.frame_paths = []
        for video_folder_name in os.listdir(LifeQaDataset.FRAMES_FILE_DIR):
            video_folder_path = os.path.join(LifeQaDataset.FRAMES_FILE_DIR, video_folder_name)
            for frame_file_name in os.listdir(video_folder_path):
                self.frame_paths.append(os.path.join(video_folder_path, frame_file_name))

    def __getitem__(self, index) -> np.ndarray:
        img = PIL.Image.open(self.frame_paths[index])

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self) -> int:
        return len(self.frame_paths)


def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    # modules = list(resnet152.children())[:-1]
    # resnet152 = nn.Sequential(*modules)
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152


def main():
    resnet152 = pretrained_resnet152()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])
    dataset = LifeQaDataset(transform=transforms)

    data_loader = torch.utils.data.DataLoader(dataset)
    for instance in data_loader:
        value = resnet152(instance)
        # TODO: get intermediate value output.
        break


if __name__ == '__main__':
    main()
