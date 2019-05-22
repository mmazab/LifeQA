#!/usr/bin/env python
import json
import math

import pandas as pd


def load_data():
    with open('data/lqa_train.json') as file:
        lqa_train = json.load(file)
    with open('data/lqa_dev.json') as file:
        lqa_dev = json.load(file)
    with open('data/lqa_test.json') as file:
        lqa_test = json.load(file)

    return lqa_train, lqa_dev, lqa_test


def main():
    data_dicts = load_data()

    df = pd.read_csv('data/sources.csv')

    for _, row in df.iterrows():
        ids = row['Video IDs']
        if isinstance(ids, str):
            for video_id in ids.split(','):
                video_id = f'{int(video_id.strip()):03d}'
                for data_dict in data_dicts:
                    if video_id in data_dict:
                        # We check there's parent video ID hasn't been previously assigned, just in case.
                        assert 'parent_video_id' not in data_dict[video_id]

                        data_dict[video_id]['parent_video_id'] = row['Link']
                        break
                else:
                    raise ValueError(f"The video ID {video_id} was not found in the dataset files")

    video_ids_without_parent = {video_id
                                for data_dict in data_dicts
                                for video_id in data_dict
                                if 'parent_video_id' not in data_dict[video_id]}

    assert not video_ids_without_parent, f"{len(video_ids_without_parent)} videos don't have the parent video ID:" \
        f" {sorted(list(video_ids_without_parent))}"

    lqa_train, lqa_dev, lqa_test = data_dicts
    with open('data/lqa_train.json', 'w') as f:
        json.dump(lqa_train, f, sort_keys=True, indent=2)
    with open('data/lqa_dev.json', 'w') as f:
        json.dump(lqa_dev, f, sort_keys=True, indent=2)
    with open('data/lqa_test.json', 'w') as f:
        json.dump(lqa_test, f, sort_keys=True, indent=2)


if __name__ == '__main__':
    main()
