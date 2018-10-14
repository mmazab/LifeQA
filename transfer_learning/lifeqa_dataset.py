import PIL.Image
import os
from typing import Callable, Dict

import torch.utils.data


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
