#!/usr/bin/env python
import json

import _jsonnet
import sklearn.model_selection


def save_part_of_fold(filename, indices, video_id_value_list):
    with open(filename, 'w') as file:
        json.dump(dict(video_id_value_list[i] for i in indices), file, sort_keys=True, indent=2)


def main():
    video_dict = json.loads(_jsonnet.evaluate_file('data/lqa_train_dev.jsonnet'))

    video_id_value_list = list(video_dict.items())
    parent_video_ids = [video['parent_video_id'] for _, video in video_id_value_list]

    fold_splitter = sklearn.model_selection.GroupKFold(n_splits=5)
    for i, (train_indices, test_indices) in enumerate(fold_splitter.split(video_id_value_list,
                                                                          groups=parent_video_ids)):
        save_part_of_fold(f'data/folds/fold{i}_train.json', train_indices, video_id_value_list)
        save_part_of_fold(f'data/folds/fold{i}_test.json', test_indices, video_id_value_list)


if __name__ == '__main__':
    main()
