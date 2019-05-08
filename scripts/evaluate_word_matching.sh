#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/word_matching \
    --include-package lqa_framework \
    training_config/word_matching.jsonnet
allennlp evaluate \
    --include-package lqa_framework \
    models/word_matching/model.tar.gz \
    data/lqa_dev.json
