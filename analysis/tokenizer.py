#!/usr/bin/env python
from nltk import word_tokenize


def main():
    with open('questions.txt') as _file:
        for line in _file:
            print(' '.join(word_tokenize(line)))


if __name__ == '__main__':
    main()
