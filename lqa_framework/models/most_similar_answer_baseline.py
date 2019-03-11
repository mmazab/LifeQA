import pickle
import bcolz
import numpy as np
import string
from scipy import spatial
import json
import random
from random import randint
import re
import math 

print("Running most_similar_answer_baseline...")

glove_path = "glove.6B"
vectors = bcolz.open(f'{glove_path}/6B.300.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}
default = np.zeros(300)

print("Embeddings loaded! Running dev data...")


def choose_most_similar(q, answers):
    # This function will return the location of the question of closest similarity
    question_vec = np.mean(np.array([glove.get(a, default) for a in q.strip('?').lower().split()]),axis = 0)


    max_similarity = math.inf
    max_vec = []
    answ_vec = None
    # Find the minimum cosine value
    for index, ans in enumerate(answers):
        answ_vec = np.mean(np.array([glove.get(a, default) for a in ans.strip('?').lower().split()]))
        if (abs(spatial.distance.cosine(question_vec, answ_vec))) <= max_similarity:
            max_similarity = abs(spatial.distance.cosine(question_vec, answ_vec))
            max_vec.append(index)
    

    if max_similarity == 0:
        return randint(0, 3)
    return random.choice(max_vec)


filename = 'lqa_dev.json' 

with open('data/' + filename) as f:
    accuracy = 0
    total = 0
    data = json.load(f)
    for video, info in data.items():
        questions = info['questions']
        for question in questions:
            choice = choose_most_similar(question["question"], question["answers"])

            if choice == question['correct_index']: 
                accuracy += 1
            # Always add to the total
            total += 1

    print("Accuracy of most_similar_answer_baseline on {} = {}".format(filename, accuracy/ total))

