#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/shortest_answer \
    --include-package lqa_framework \
    training_config/shortest_answer.jsonnet
allennlp evaluate \
    --include-package lqa_framework \
    models/shortest_answer/model.tar.gz \
    data/lqa_dev.json
