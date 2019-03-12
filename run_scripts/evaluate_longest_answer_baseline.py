#!/usr/bin/env python
import json

from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.training.util import evaluate

from lqa_framework.dataset_readers.lifeqa import LqaDatasetReader
from lqa_framework.models.longest_answer_baseline import LongestAnswerBaseline


def main():
    reader = LqaDatasetReader()
    validation_dataset = reader.read(cached_path('data/lqa_dev.json'))

    vocab = Vocabulary.from_instances(validation_dataset)

    data_iterator = BasicIterator()
    data_iterator.index_with(vocab)

    model = LongestAnswerBaseline(vocab)

    print(json.dumps(evaluate(model, validation_dataset, data_iterator, -1, '')))


if __name__ == '__main__':
    main()
