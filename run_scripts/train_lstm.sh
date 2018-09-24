#!/usr/bin/env bash
rm -r /scratch/mihalcea_fluxg/mazab/lifeqa/models
allennlp train lqa_framework/experiments/lstm_baseline.json -s /scratch/mihalcea_fluxg/mazab/lifeqa/models --include-package lqa_framework

