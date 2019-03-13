#!/usr/bin/env python
import argparse
from allennlp.common import Params

from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.util import evaluate


def parse_args():
    parser = argparse.ArgumentParser()  # TODO
    parser.add_argument('model', choices=['longest_answer', 'shortest_answer', 'most_similar_answer'],
                        help="model to run")
    return parser.parse_args()


def main():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from lqa_framework import LongestAnswer, LqaDatasetReader, MostSimilarAnswer, ShortestAnswer

    args = parse_args()

    reader = LqaDatasetReader()
    validation_dataset = reader.read(cached_path('data/lqa_dev.json'))

    vocab = Vocabulary.from_instances(validation_dataset)

    data_iterator = BasicIterator()
    data_iterator.index_with(vocab)

    if args.model == 'longest_answer':
        model = LongestAnswer(vocab)
    elif args.model == 'shortest_answer':
        model = ShortestAnswer(vocab)
    elif args.model == 'most_similar_answer':
        # Use from_params because it does some extra stuff __init__ doesn't.
        embedder = Embedding.from_params(vocab, Params({
          'pretrained_file': 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz',
          'embedding_dim': 300,
          'trainable': False,
        }))
        text_field_embedder = BasicTextFieldEmbedder({'tokens': embedder})

        question_encoder = BagOfEmbeddingsEncoder(1)
        answers_encoder = BagOfEmbeddingsEncoder(1)  # FIXME: use only one encoder?

        model = MostSimilarAnswer(vocab, text_field_embedder, question_encoder, answers_encoder)
    else:
        raise ValueError("Model name not recognized")

    evaluate(model, validation_dataset, data_iterator, -1, '')


if __name__ == '__main__':
    main()
