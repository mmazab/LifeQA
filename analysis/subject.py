#!/usr/bin/env python
import argparse
from collections import Counter
import sys

from conllu import parse
from nltk.stem.wordnet import WordNetLemmatizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    debug = args.debug

    with open('output/questions') as questions_output_file:
        questions_output = questions_output_file.read()

    sentences = parse(questions_output)

    without_nn = 0
    whats = 0

    lemmatizer = WordNetLemmatizer()

    counter = Counter()

    for sentence in sentences:
        if sentence[0]['form'].lower() == 'what':
            if debug:
                print(' '.join(token['form'] for token in sentence))

            tree = sentence.to_tree()

            if debug:
                print("root: {}".format(tree.token['form']))

                # Inspect root's children.
                for child in tree.children:
                    token = child.token
                    print("    {}: {}".format(token['deprel'], token['form']))

            # All NNs.
            for token in sentence:
                if token['upostag'].startswith('NN'):
                    lemma = lemmatizer.lemmatize(token['form'].lower())
                    counter[lemma] += 1
                    print(lemma, end=' ')
            print('')

            # First NN.
            for token in sentence:
                if token['upostag'].startswith('NN'):
                    lemma = lemmatizer.lemmatize(token['form'].lower())
                    if debug:
                        print(lemma)
                    break
            else:
                without_nn += 1
                if debug:
                    print('')
            whats += 1

            if debug:
                print('')
        else:
            print('')

    print(counter.most_common(20))

    sys.stderr.write("There were {}/{} 'what' questions without NNs\n".format(without_nn, whats))


if __name__ == '__main__':
    main()
