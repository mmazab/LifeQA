#!/usr/bin/env bash
allennlp train \
    --force \
    --serialization-dir models/longest_answer \
    --include-package lqa_framework \
    lqa_framework/experiments/longest_answer.jsonnet
allennlp evaluate \
    --include-package lqa_framework \
    models/longest_answer/model.tar.gz \
    data/lqa_dev.json
