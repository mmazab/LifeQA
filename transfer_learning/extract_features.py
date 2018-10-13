#!/usr/bin/env python
"""Script to extract ResNet features from video frames."""
import os
from typing import Callable, Dict

import h5py
import PIL.Image
import torch.utils.data
import torchvision
from tqdm import tqdm

FEATURES_DIR = 'data/features'


class LifeQaDataset(torch.utils.data.Dataset):
    """Dataset of LifeQA video frames with ``PIL.Image.Image`` type."""
    FRAMES_FILE_DIR = 'data/frames'

    def __init__(self, transform: Callable = None) -> None:
        self.transform = transform

        self.video_ids = []
        self.frame_paths = []
        self.video_ids_by_idx = []
        self.frame_ids_by_idx = []
        self.frame_count_by_video_id = {}
        for video_folder_name in os.listdir(LifeQaDataset.FRAMES_FILE_DIR):
            self.video_ids.append(video_folder_name)
            video_folder_path = os.path.join(LifeQaDataset.FRAMES_FILE_DIR, video_folder_name)
            frame_file_names = os.listdir(video_folder_path)
            for i, frame_file_name in enumerate(frame_file_names):
                self.frame_paths.append(os.path.join(video_folder_path, frame_file_name))
                self.frame_ids_by_idx.append(i)
                self.video_ids_by_idx.append(video_folder_name)
            self.frame_count_by_video_id[video_folder_name] = len(frame_file_names)

    def __getitem__(self, index) -> Dict[str, object]:
        frame = PIL.Image.open(self.frame_paths[index])

        if self.transform:
            frame = self.transform(frame)

        video_id = self.video_ids_by_idx[index]
        return {
            'video_id': video_id,
            'video_frame_count': self.frame_count_by_video_id[video_id],
            'frame_id': self.frame_ids_by_idx[index],
            'frame': frame
        }

    def __len__(self) -> int:
        return len(self.frame_paths)


def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152


def features_file_name(model_name, layer_name):
    return f"TGIF_{model_name}_{layer_name}.hdf5"


def save_resnet_features(dataset, resnet):
    resnet_res5c_path = os.path.join(FEATURES_DIR, features_file_name('RESNET', 'res5c'))
    resnet_pool5_path = os.path.join(FEATURES_DIR, features_file_name('RESNET', 'pool5'))
    with h5py.File(resnet_res5c_path, 'w') as resnet_res5c_file, \
            h5py.File(resnet_pool5_path, 'w') as resnet_pool5_file:

        for video_id in dataset.video_ids:
            video_frame_count = dataset.frame_count_by_video_id[video_id]
            # resnet_res5c_file.create_dataset(video_id, shape=(video_frame_count,))
            resnet_pool5_file.create_dataset(video_id, shape=(video_frame_count, 2048))

        res5c_output = None
        avg_pool_value = None

        def avg_pool_hook(_module, input_, output):
            nonlocal res5c_output, avg_pool_value
            res5c_output = input_[0]
            avg_pool_value = output.view(output.size(0), -1)

        resnet.avgpool.register_forward_hook(avg_pool_hook)

        for instance in tqdm(torch.utils.data.DataLoader(dataset), desc="Extracting ResNet features"):
            # Remember DataLoad gives the data transformed to tensors (except strings).
            video_id = instance['video_id'][0]
            frame_id = instance['frame_id'].item()
            frame = instance['frame']

            resnet(frame)  # The fc1000 layer is computed unnecessarily, but it's just 1 layer.

            # resnet_res5c_file[video_id][frame_id] = res5c_output
            resnet_pool5_file[video_id][frame_id] = avg_pool_value


def save_c3d_features(dataset, c3d):
    pass


def main():
    resnet152 = pretrained_resnet152()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])
    dataset = LifeQaDataset(transform=transforms)

    save_resnet_features(dataset, resnet152)
    save_c3d_features(dataset, None)


if __name__ == '__main__':
    main()
