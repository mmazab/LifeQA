import json


def load_data():
    with open('data/lqa_train.json') as file:
        lqa_train = json.load(file)
    with open('data/lqa_dev.json') as file:
        lqa_dev = json.load(file)
    with open('data/lqa_test.json') as file:
        lqa_test = json.load(file)
    return lqa_train, lqa_dev, lqa_test


def save_data(data_dicts):
    lqa_train, lqa_dev, lqa_test = data_dicts
    with open('data/lqa_train.json', 'w') as f:
        json.dump(lqa_train, f, sort_keys=True, indent=2)
    with open('data/lqa_dev.json', 'w') as f:
        json.dump(lqa_dev, f, sort_keys=True, indent=2)
    with open('data/lqa_test.json', 'w') as f:
        json.dump(lqa_test, f, sort_keys=True, indent=2)
