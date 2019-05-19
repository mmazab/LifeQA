#!/usr/bin/env python
from nltk import word_tokenize


def main():
    with open('questions.txt') as file:
        for line in file:
            print(' '.join(word_tokenize(line)))


if __name__ == '__main__':
    main()
