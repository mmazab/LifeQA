import json
from typing import Any, MutableMapping, Tuple


def load_video_dicts_split() -> Tuple[MutableMapping[str, Any], MutableMapping[str, Any], MutableMapping[str, Any]]:
    with open('data/lqa_train.json') as file:
        lqa_train = json.load(file)
    with open('data/lqa_dev.json') as file:
        lqa_dev = json.load(file)
    with open('data/lqa_test.json') as file:
        lqa_test = json.load(file)
    return lqa_train, lqa_dev, lqa_test


def save_video_dicts_split(
        data_dicts: Tuple[MutableMapping[str, Any], MutableMapping[str, Any], MutableMapping[str, Any]]) -> None:
    lqa_train, lqa_dev, lqa_test = data_dicts
    with open('data/lqa_train.json', 'w') as f:
        json.dump(lqa_train, f, sort_keys=True, indent=2)
    with open('data/lqa_dev.json', 'w') as f:
        json.dump(lqa_dev, f, sort_keys=True, indent=2)
    with open('data/lqa_test.json', 'w') as f:
        json.dump(lqa_test, f, sort_keys=True, indent=2)
