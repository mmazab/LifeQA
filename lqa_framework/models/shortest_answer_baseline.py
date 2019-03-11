# Implement simple shortest answer baseline using basic python techniques because rip Anaconda
import random
import json


def choose_shortest(answers):
    # This function will return the location of minimum length answer in answers
    min_num = len(min(answers, key=lambda x: len(x)))

    min_choices = []
    # Make a list of all the min length answers
    for index, ans in enumerate(answers):
        if len(ans) == min_num:
            min_choices.append(index)

    # Return a random min length answer
    return random.choice(min_choices)


filename = 'lqa_dev.json' 

with open('data/' + filename) as f:
    accuracy = 0
    total = 0
    data = json.load(f)
    for video, info in data.items():
        questions = info['questions']
        for question in questions:
            answers = question["answers"]
            choice = choose_shortest(answers)

            if choice == question['correct_index']: 
                accuracy += 1
            # Always add to the total
            total += 1

    print("Accuracy of shortest_answer_baseline on {} = {}".format(filename, accuracy/ total))
