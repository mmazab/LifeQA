#!/usr/bin/env python
import sys

from conllu import parse
from nltk.stem.wordnet import WordNetLemmatizer


def main():
    with open('output/questions') as questions_output_file:
        questions_output = questions_output_file.read()

    sentences = parse(questions_output)

    without_nn = 0
    whats = 0

    lemmatizer = WordNetLemmatizer()

    for sentence in sentences:
        if sentence[0]['form'].lower() == 'what':
            # print(' '.join(token['form'] for token in sentence))  # DEBUG
            # tree = sentence.to_tree()
            # print("root: {}".format(tree.token['form']))  # DEBUG

            # Inspect root's children.
            # for child in tree.children:
            #     token = child.token
            #     print("    {}: {}".format(token['deprel'], token['form']))

            # All NNs.
            # print('\t'.join(lemmatizer.lemmatize(token['form'].lower())
            #                 for token in sentence if token['upostag'].startswith('NN')))

            # First NN.
            for token in sentence:
                if token['upostag'].startswith('NN'):
                    print(lemmatizer.lemmatize(token['form'].lower()))
                    break
            else:
                without_nn += 1
                print('')
            whats += 1
            # print('')  # DEBUG
        else:
            print('')

    sys.stderr.write("There were {}/{} 'what' questions without NNs\n".format(without_nn, whats))


if __name__ == '__main__':
    main()
