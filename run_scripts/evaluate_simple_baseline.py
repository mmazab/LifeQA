#!/usr/bin/env python
import argparse
import logging

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.util import cleanup_global_logging, prepare_environment, prepare_global_logging
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN, Vocabulary
from allennlp.modules import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.util import create_serialization_dir, evaluate

GLOVE_URL = 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz'


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

    from lqa_framework import LongestAnswer, LqaDatasetReader, MostSimilarAnswer, ShortestAnswer

    args = parse_args()

    prepare_environment(Params({}))

    # TODO: maybe the following code is useful to use `allennlp evaluate` instead of this file?
    # serialization_dir = f'models/{args.model}'
    # create_serialization_dir(Params({}), serialization_dir, False, True)
    # stdout_handler = prepare_global_logging(serialization_dir, False)

    token_indexers = {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}
    reader = LqaDatasetReader(token_indexers=token_indexers)
    validation_dataset = reader.read(cached_path('data/lqa_dev.json'))

    vocab = Vocabulary.from_instances(validation_dataset, pretrained_files={'tokens': GLOVE_URL},
                                      only_include_pretrained_words=True)

    data_iterator = BasicIterator()
    data_iterator.index_with(vocab)

    if args.model == 'longest_answer':
        model = LongestAnswer(vocab)
    elif args.model == 'shortest_answer':
        model = ShortestAnswer(vocab)
    elif args.model == 'most_similar_answer':
        # Use from_params because it does some extra stuff __init__ doesn't.
        embedder = Embedding.from_params(vocab, Params({
          'pretrained_file': GLOVE_URL,
          'embedding_dim': 300,
          'trainable': False,
        }))
        embedder.weight[vocab.get_token_index(DEFAULT_OOV_TOKEN, 'tokens')].fill_(0)
        embedder.weight[vocab.get_token_index(DEFAULT_PADDING_TOKEN, 'tokens')].fill_(0)

        text_field_embedder = BasicTextFieldEmbedder({'tokens': embedder})

        question_encoder = BagOfEmbeddingsEncoder(1)
        answers_encoder = BagOfEmbeddingsEncoder(1)  # FIXME: use only one encoder?

        model = MostSimilarAnswer(vocab, text_field_embedder, question_encoder, answers_encoder)
    else:
        raise ValueError("Model name not recognized")

    evaluate(model, validation_dataset, data_iterator, -1, '')

    # cleanup_global_logging(stdout_handler)


if __name__ == '__main__':
    main()
