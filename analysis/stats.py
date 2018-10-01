#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import Counter
import sys

from conllu import parse
from nltk.stem.wordnet import WordNetLemmatizer

with open('output/questions') as questions_output_file:
    questions_output = questions_output_file.read()

sentences = parse(questions_output)

lemmatizer = WordNetLemmatizer()

whats = [sentence for sentence in sentences if sentence[0]['form'].lower() == 'what']

whats_nn = [sentence for sentence in whats if sentence[1]['upostag'].startswith('NN')]
print("what NN[S]: {}/{}".format(len(whats_nn), len(whats)))
counter_nn = Counter(lemmatizer.lemmatize(sentence[1]['form'].lower()) for sentence in whats_nn)
print(counter_nn.most_common(20))

whats_do = [sentence for sentence in whats if sentence[1]['form'].lower() in ['do', 'does', 'did']]
print("what [does|do|did]: {}/{}".format(len(whats_do), len(whats)))
counter_do = Counter(lemmatizer.lemmatize(sentence.to_tree().token['form'].lower()) for sentence in whats_do)
print(counter_do.most_common(20))

whats_be = [sentence for sentence in whats if sentence[1]['form'].lower() in ['is', 'will', 'are', 'was', 'were']]
print("what BE [head-of-NP]: {}/{}".format(len(whats_be), len(whats)))
counter_be = Counter()
for sentence in whats_be:
    tree = sentence.to_tree()
    root = tree.token
    if root['upostag'].startswith('NN'):
        counter_be[lemmatizer.lemmatize(root['form'].lower())] += 1
    else: # The root It's probably a verb
        nsubj_id = [int(child.token['id']) for child in tree.children if child.token['deprel'].startswith('nsubj') and int(child.token['id']) > 1]
        if nsubj_id:
            counter_be[lemmatizer.lemmatize(sentence[min(nsubj_id) - 1]['form'].lower())] += 1
        else:
            prep_id = [int(child.token['id']) for child in tree.children if child.token['deprel'] == 'prep']
            if prep_id:
                prep_tree = [child for child in tree.children if int(child.token['id']) == min(prep_id)][0]
                pobj_id = [int(child.token['id']) for child in prep_tree.children if child.token['deprel'] == 'pobj']
                if pobj_id:
                    counter_be[lemmatizer.lemmatize(sentence[min(pobj_id) - 1]['form'].lower())] += 1
                else:
                    # There is 1 case that falls here.
                    # print(' '.join(token['form'] for token in sentence))
                    # sentence.to_tree().print_tree()
                    # print('')
                    pass
            else:
                # There are 5 cases that fall here.
                # print(' '.join(token['form'] for token in sentence))
                # sentence.to_tree().print_tree()
                # print('')
                pass
print(counter_be.most_common(20))
