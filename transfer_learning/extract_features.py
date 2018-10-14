#!/usr/bin/env python
"""Script to extract ResNet features from video frames."""
import os

import h5py
import torch.utils.data
import torchvision
from tqdm import tqdm

from transfer_learning.c3d import C3D
from transfer_learning.lifeqa_dataset import LifeQaDataset

FEATURES_DIR = 'data/features'


def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152


def features_file_name(model_name, layer_name):
    return f"TGIF_{model_name}_{layer_name}.hdf5"


def save_resnet_features():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = LifeQaDataset(transform=transforms)

    resnet = pretrained_resnet152()

    res5c_features_path = os.path.join(FEATURES_DIR, features_file_name('RESNET', 'res5c'))
    pool5_features_path = os.path.join(FEATURES_DIR, features_file_name('RESNET', 'pool5'))
    with h5py.File(res5c_features_path, 'w') as res5c_features_file, \
            h5py.File(pool5_features_path, 'w') as pool5_features_file:

        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            res5c_features_file.create_dataset(video_id, shape=(video_frame_count, 2048, 7, 7))
            pool5_features_file.create_dataset(video_id, shape=(video_frame_count, 2048))

        res5c_output = None
        avg_pool_value = None

        def avg_pool_hook(_module, input_, output):
            nonlocal res5c_output, avg_pool_value
            res5c_output = input_[0]
            avg_pool_value = output.view(output.size(0), -1)

        resnet.avgpool.register_forward_hook(avg_pool_hook)

        for instance in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting ResNet features"):
            # Remember DataLoader returns the data transformed to tensors (except strings which are inside lists).
            video_id = instance['video_id'][0]
            frame_id = instance['frame_id'].item()
            frame = instance['frame']

            resnet(frame)  # The fc1000 layer is computed unnecessarily, but it's just 1 layer.

            res5c_features_file[video_id][frame_id] = res5c_output
            pool5_features_file[video_id][frame_id] = avg_pool_value


def save_c3d_features():
    transforms = torchvision.transforms.Compose([  # TODO
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = LifeQaDataset(transform=transforms)

    c3d = C3D()

    conv5b_features_path = os.path.join(FEATURES_DIR, features_file_name('C3D', 'conv5b'))
    fc6_features_path = os.path.join(FEATURES_DIR, features_file_name('C3D', 'fc6'))
    with h5py.File(conv5b_features_path, 'w') as conv5b_features_file, \
            h5py.File(fc6_features_path, 'w') as fc6_features_file:

        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            conv5b_features_file.create_dataset(video_id, shape=(video_frame_count, 1024, 7, 7))
            fc6_features_file.create_dataset(video_id, shape=(video_frame_count, 4096))

        for instance in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting C3D features"):
            # Remember DataLoader returns the data transformed to tensors (except strings which are inside lists).
            video_id = instance['video_id'][0]
            frame_id = instance['frame_id'].item()
            frame = instance['frame']

            c3d(frame)  # TODO: take 16 frames, what overlapping?


def main():
    save_resnet_features()
    save_c3d_features()


if __name__ == '__main__':
    main()
