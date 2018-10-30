#!/usr/bin/env python
import torch
import torch.nn
import torch.utils.data
from tqdm import tqdm

from lifeqa_dataset import LifeQaDataset

# noinspection PyUnresolvedReferences
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    epochs = 50
    pool5_size = 2048
    hidden_layer_size = 3  # 512
    glove_vector_size = 300
    output_size = 4
    dataset_options = {
        'load_frames': False,
        'load_resnet_features': 'pool5',
        'check_missing_videos': False,  # FIXME
    }
    training_dataset = LifeQaDataset(videos_data_path='data/lqa_train.json', **dataset_options)
    dev_dataset = LifeQaDataset(videos_data_path='data/lqa_dev.json', **dataset_options)
    test_dataset = LifeQaDataset(videos_data_path='data/lqa_test.json', **dataset_options)

    video_encoder_lstm = torch.nn.LSTM(pool5_size, hidden_layer_size)
    question_encoder_lstm = torch.nn.LSTM(glove_vector_size, hidden_layer_size)
    answer_encoder_lstm = torch.nn.LSTM(glove_vector_size, hidden_layer_size)
    linear = torch.nn.Linear(hidden_layer_size, output_size)

    with tqdm(desc="Training", total=epochs * len(training_dataset)) as progress_bar:
        for _ in range(epochs):
            for instance in torch.utils.data.DataLoader(training_dataset):  # shuffle=True
                resnet_features = instance['resnet_features'].to(DEVICE)
                resnet_features = resnet_features.permute(1, 0, 2)
                _, (h_video, _) = video_encoder_lstm(resnet_features)
                h_video

                progress_bar.update()


if __name__ == '__main__':
    main()
