#!/usr/bin/python
import argparse
import json
import os
from collections import Mapping

import numpy as np
from tqdm import tqdm

FEATURE_EXTENSION = ".npy"


def get_scores_corresponding_objects(features_dir: str, output_dir: str) -> None:
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for features_filename in sorted(os.listdir(features_dir)):
        features_path = os.path.join(features_dir, features_filename)

        objects = []
        print(f"Currently processing: {features_filename}")
        for video_features_filename in tqdm(sorted(os.listdir(features_path))):
            video_feature_filename_without_extension, extension = video_features_filename.rsplit(".", maxsplit=1)

            if extension != FEATURE_EXTENSION:
                continue

            video_features_path = os.path.join(features_path, video_features_filename)

            video_features: Mapping[str, np.ndarray] = np.load(video_features_path, allow_pickle=True).item()  # noqa
            if not video_features:
                print(f"Empty feature file encountered: {video_features_path}")
                continue

            scores = video_features["cls_scores"]
            cls_indices = np.argmax(scores, axis=1)

            with open("objects_vocab.txt") as file:
                index_to_object = {i: line.strip() for i, line in enumerate(file.readlines())}

            frame_objects = list({index_to_object[i] for i in cls_indices if i > 0})
            objects.append((video_feature_filename_without_extension, frame_objects))

        with open(os.path.join(output_dir, features_path + ".json"), "w") as file:
            json.dump(objects, file, indent=4, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir")
    parser.add_argument("output_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    get_scores_corresponding_objects(features_dir=args.features_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
