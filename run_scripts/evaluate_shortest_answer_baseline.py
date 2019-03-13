#!/usr/bin/env python
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BasicIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.util import evaluate


def main():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from lqa_framework import LqaDatasetReader, ShortestAnswer

    reader = LqaDatasetReader()
    validation_dataset = reader.read(cached_path('data/lqa_dev.json'))

    vocab = Vocabulary.from_instances(validation_dataset)

    data_iterator = BasicIterator()
    data_iterator.index_with(vocab)

    model = ShortestAnswer(vocab)

    evaluate(model, validation_dataset, data_iterator, -1, '')


if __name__ == '__main__':
    main()
