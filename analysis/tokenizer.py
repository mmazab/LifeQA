#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

from nltk import word_tokenize

with open('questions.txt') as _file:
    for line in _file:
        print(' '.join(word_tokenize(line)))
