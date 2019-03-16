#!/usr/bin/env python
import argparse
import logging
import os
import shutil

from allennlp.common import Params
from allennlp.common.util import prepare_environment
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.training.trainer import TrainerPieces
from allennlp.training.util import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Runs simple baselines that are parameterless. As they are"
                                                 " parameterless, it makes little sense to 'train' them (and it fails"
                                                 " because of that). But the allennlp's 'evaluate' subcommand needs a"
                                                 " saved model and doesn't support config params. Hence, this script"
                                                 " is necessary.")
    parser.add_argument('model', choices=['longest_answer', 'shortest_answer', 'most_similar_answer'],
                        help="model to run")
    return parser.parse_args()


def remove_serialization_dir_if_exists(serialization_dir):
    if os.path.exists(serialization_dir):
        shutil.rmtree(serialization_dir)


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)

    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from lqa_framework import MostSimilarAnswer

    args = parse_args()

    params = Params.from_file(f'lqa_framework/experiments/{args.model}.jsonnet')

    prepare_environment(params)

    serialization_dir = f'models/{args.model}'

    remove_serialization_dir_if_exists(serialization_dir)

    trainer_pieces = TrainerPieces.from_params(params, serialization_dir)

    if isinstance(trainer_pieces.model, MostSimilarAnswer):
        # noinspection PyProtectedMember
        embedder = trainer_pieces.model.text_field_embedder._token_embedders['tokens']
        embedder.weight[trainer_pieces.model.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'tokens')].fill_(0)

    metrics = evaluate(trainer_pieces.model, trainer_pieces.validation_dataset, trainer_pieces.iterator, -1, '')
    print(f"accuracy: {metrics['accuracy']:.4f}")


if __name__ == '__main__':
    main()
