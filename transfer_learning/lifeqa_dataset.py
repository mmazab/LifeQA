import json
import os
from typing import Callable, Dict

import h5py
import PIL.Image
import torch
import torch.utils.data


class LifeQaDataset(torch.utils.data.Dataset):
    """Dataset of LifeQA videos."""
    FRAMES_DIR_PATH = 'data/frames'

    def __init__(self, transform: Callable = None, videos_data_path: str = 'data/lqa_data.json',
                 check_missing_videos: bool = True, load_frames: bool = True, load_resnet_features: str = None,
                 load_c3d_features: str = None) -> None:
        self.transform = transform

        self.load_frames = load_frames
        self.load_resnet_features = load_resnet_features
        self.load_c3d_features = load_c3d_features

        with open(videos_data_path) as file:
            self.videos_data_dict = json.load(file)

        if load_frames:
            for video_id in list(self.videos_data_dict.keys()):  # Convert to list to be able to possibly remove items.
                video_folder_path = self._video_folder_path(video_id)
                if not os.path.exists(video_folder_path):
                    if check_missing_videos:
                        raise FileNotFoundError(f"Directory {video_folder_path} not found, which was referenced in"
                                                f" {videos_data_path}")
                    else:
                        del self.videos_data_dict[video_id]

        if self.load_resnet_features:
            with h5py.File(self.features_file_path('resnet', self.load_resnet_features)) as features_file:
                # Convert to list to be able to possibly remove items.
                for video_id in list(self.videos_data_dict.keys()):
                    if video_id not in features_file:
                        if check_missing_videos:
                            raise Exception(f"Frames features from video {video_id} not present in the features file.")
                        else:
                            del self.videos_data_dict[video_id]

        if self.load_c3d_features:
            with h5py.File(self.features_file_path('c3d', self.load_c3d_features)) as features_file:
                # Convert to list to be able to possibly remove items.
                for video_id in list(self.videos_data_dict.keys()):
                    if video_id not in features_file:
                        if check_missing_videos:
                            raise Exception(f"Frames features from video {video_id} not present in the features file.")
                        else:
                            del self.videos_data_dict[video_id]

        self.video_ids = list(key for key in self.videos_data_dict.keys() if int(key) >= 173)

        if load_frames:
            self.frame_count_by_video_id = {video_id: len(os.listdir(self._video_folder_path(video_id)))
                                            for video_id in self.video_ids}

    @staticmethod
    def _video_folder_path(video_id: str) -> str:
        return os.path.join(LifeQaDataset.FRAMES_DIR_PATH, video_id)

    @staticmethod
    def features_file_path(model_name: str, layer_name: str) -> str:
        return f"data/features/LifeQA_{model_name.upper()}_{layer_name}.hdf5"

    def __getitem__(self, index) -> Dict[str, object]:
        video_id = self.video_ids[index]
        video_data_dict = self.videos_data_dict[video_id]

        item = {
            'id': video_id,
            'questions': video_data_dict['questions'],
        }

        captions_dict = video_data_dict['captions']
        if captions_dict:
            item['captions'] = captions_dict

        if self.load_frames:
            frames = None

            video_folder_path = self._video_folder_path(video_id)
            for i, frame_file_name in enumerate(os.listdir(video_folder_path)):
                frame = PIL.Image.open(os.path.join(video_folder_path, frame_file_name))
                if self.transform:
                    frame = self.transform(frame)

                if frames is None:
                    # noinspection PyUnresolvedReferences
                    frames = torch.empty((self.frame_count_by_video_id[video_id], *frame.size()))
                frames[i] = frame

            item['frames'] = frames

        if self.load_resnet_features:
            with h5py.File(self.features_file_path('resnet', self.load_resnet_features)) as features_file:
                item['resnet_features'] = features_file[video_id].value

        if self.load_c3d_features:
            with h5py.File(self.features_file_path('c3d', self.load_resnet_features)) as features_file:
                item['c3d_features'] = features_file[video_id].value

        return item

    def __len__(self) -> int:
        return len(self.video_ids)
