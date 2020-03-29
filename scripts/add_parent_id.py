#!/usr/bin/env python
import pandas as pd

import scripts.util


def main():
    data_dicts = scripts.util.load_video_dicts_split()

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

    scripts.util.save_video_dicts_split(data_dicts)


if __name__ == '__main__':
    main()
