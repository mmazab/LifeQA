allennlp train \
    --force \
    --serialization-dir models/longest_answer \
    --include-package lqa_framework \
    lqa_framework/experiments/text_longest_baseline.jsonnet
