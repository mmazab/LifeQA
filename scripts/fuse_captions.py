#!/usr/bin/env python
import os
from typing import Iterable

from scripts import util


def update_captions(id_: str, captions: Iterable[str], data_dict: util.DATA_TYPE) -> None:
    # We don't consider the timestamps, the empty lines, and the subtitle indices.
    data_dict[id_]["manual_captions"] = [{"transcript": line}
                                         for line in captions if line and not line.isdecimal() and " --> " not in line]


def rename_automatic_captions_field(data_dict: util.DATA_TYPE) -> None:
    for id_ in data_dict:
        if "captions" in data_dict[id_]:
            data_dict[id_]["automatic_captions"] = data_dict[id_]["captions"]
            del data_dict[id_]["captions"]


def main() -> None:
    data_dicts = util.load_video_dicts_split()

    for data_dict in data_dicts:
        rename_automatic_captions_field(data_dict)

    for directory in ["data/lqa_trans/high_quality", "data/lqa_trans/low_quality"]:
        for filename in os.listdir(directory):
            with open(os.path.join(directory, filename)) as file:
                captions = [line.strip() for line in file.readlines()]

            id_ = filename[:-4]
            for data_dict in data_dicts:
                if id_ in data_dict:
                    update_captions(id_, captions, data_dict)
                    break
            else:
                raise ValueError(f"The ID {id_} is neither in train, dev, nor test.")

    util.save_video_dicts_split(data_dicts)


if __name__ == "__main__":
    main()
