"""Matching baseline Model implemented from DREAM (Sun et al., 2018)."""
import json
import random
import re

RE_SYMBOLS = re.compile(r'[^\w\s]')  # TODO: consider _?


def matching(question, answers):
    tokens = set(RE_SYMBOLS.sub('', question.lower()).split())  # TODO: multiset?

    matching_counts = [0, 0, 0, 0]

    for i, answer in enumerate(answers):
        for token in RE_SYMBOLS.sub('', answer.lower()).split():
            if token in tokens:
                matching_counts[i] += 1

    mx = max(matching_counts)

    return random.choice([x for x in matching_counts if x == mx])


def main():
    with open('data/lqa_dev.json') as file:
        video_dict = json.load(file)

    total = correct = 0
    for video in video_dict.values():
        for video_question in video['questions']:
            if matching(video_question['question'], video_question['answers']) == video_question['correct_index']:
                correct += 1
            total += 1

    print(f"Accuracy of the Matching baseline: {correct / total:.4f}")


if __name__ == '__main__':
    main()
