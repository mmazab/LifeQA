#!/usr/bin/env python
import argparse
import logging

from allennlp.common import Params
from allennlp.common.util import prepare_environment
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.training.trainer import TrainerPieces
from allennlp.training.util import evaluate


def parse_args():
    parser = argparse.ArgumentParser()  # TODO
    parser.add_argument('model', choices=['longest_answer', 'shortest_answer', 'most_similar_answer'],
                        help="model to run")
    return parser.parse_args()


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

    trainer_pieces = TrainerPieces.from_params(params, serialization_dir)

    if isinstance(trainer_pieces.model, MostSimilarAnswer):
        # noinspection PyProtectedMember
        embedder = trainer_pieces.model.text_field_embedder._token_embedders['tokens']

        embedder.weight[trainer_pieces.model.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'tokens')].fill_(0)
        # Shouldn't be necessary, but just in case:
        embedder.weight[trainer_pieces.model.vocab.get_token_index(DEFAULT_PADDING_TOKEN, 'tokens')].fill_(0)

    evaluate(trainer_pieces.model, trainer_pieces.validation_dataset, trainer_pieces.iterator, -1, '')


if __name__ == '__main__':
    main()
