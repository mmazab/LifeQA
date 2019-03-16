#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/most_similar_answer \
    --include-package lqa_framework \
    lqa_framework/experiments/most_similar_answer.jsonnet
allennlp evaluate \
    --include-package lqa_framework \
    models/most_similar_answer/model.tar.gz \
    data/lqa_dev.json
