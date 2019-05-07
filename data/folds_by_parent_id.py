#!/usr/bin/env python
import json

import _jsonnet
import numpy as np
import sklearn.model_selection


def main():
    video_dict = json.loads(_jsonnet.evaluate_file('data/lqa_train_dev.jsonnet'))
    parent_video_ids = np.asarray(sorted(list({video['parent_video_id'] for video in video_dict.values()})))  # 50

    for i, (train_index, test_index) in enumerate(sklearn.model_selection.KFold(n_splits=5).split(parent_video_ids)):
        with open(f'data/folds/fold{i}_train.json', 'w') as file:
            json.dump({video_id: video for video_id, video in video_dict.items()
                       if video['parent_video_id'] in set(parent_video_ids[train_index])}, file, sort_keys=True,
                      indent=2)
        with open(f'data/folds/fold{i}_test.json', 'w') as file:
            json.dump({video_id: video for video_id, video in video_dict.items()
                       if video['parent_video_id'] in set(parent_video_ids[test_index])}, file, sort_keys=True,
                      indent=2)


if __name__ == '__main__':
    main()
