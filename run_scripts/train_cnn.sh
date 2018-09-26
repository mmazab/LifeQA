#!/usr/bin/env bash
rm -r /scratch/mihalcea_fluxg/mazab/lifeqa/models/cnn
allennlp train lqa_framework/experiments/lstm_baseline.json -s /scratch/mihalcea_fluxg/mazab/lifeqa/models/cnn --include-package lqa_framework

