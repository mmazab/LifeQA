# Matching baseline Model implemented from DREAM (Sun et. al 2018)
import json
import re
import operator
from random import choice



FILE = "data/lqa_dev.json"

def matching(question, answers):
    words = {}
    for w in re.sub(r'[^\w\s]','',question.lower()).split():
        words[w] = True

    matching_counts = [0,0,0,0]

    for i in range(len(answers)):
        answer = answers[i]
        for w in re.sub(r'[^\w\s]','',answer.lower()).split():
            if w in words:
                matching_counts[i] += 1

    mx = max(matching_counts)
    all_maxes = [x for x in matching_counts if x == mx]

    result = choice(all_maxes)
    return result



acc = 0
tot = 0
with open(FILE, 'r') as f:
    data = json.load(f)
    for video in data:
        for question in data[video]['questions']:
            if matching(question['question'], question['answers']) == question['correct_index']:
                acc += 1
            tot += 1


print("Accuracy of matching baseline on {} = {}".format(FILE, acc/tot))

