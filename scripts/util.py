import json
from typing import Any, MutableMapping, Tuple

DATA_TYPE = MutableMapping[str, Any]


def load_video_dicts_split() -> Tuple[DATA_TYPE, DATA_TYPE, DATA_TYPE]:
    with open("data/lqa_train.json") as file:
        lqa_train = json.load(file)
    with open("data/lqa_dev.json") as file:
        lqa_dev = json.load(file)
    with open("data/lqa_test.json") as file:
        lqa_test = json.load(file)
    return lqa_train, lqa_dev, lqa_test


def save_video_dicts_split(data_dicts: Tuple[DATA_TYPE, DATA_TYPE, DATA_TYPE]) -> None:
    lqa_train, lqa_dev, lqa_test = data_dicts
    with open("data/lqa_train.json", "w") as file:
        json.dump(lqa_train, file, sort_keys=True, indent=2)
    with open("data/lqa_dev.json", "w") as file:
        json.dump(lqa_dev, file, sort_keys=True, indent=2)
    with open("data/lqa_test.json", "w") as file:
        json.dump(lqa_test, file, sort_keys=True, indent=2)
