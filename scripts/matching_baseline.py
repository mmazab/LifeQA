"""Matching baseline Model implemented from DREAM (Sun et al., 2018)."""
import json
import random
import re

RE_SYMBOLS = re.compile(r'[^\w\s]')  # TODO: consider _?


def index_of_max_overlapping(question, answers):
    tokens = set(RE_SYMBOLS.sub('', question.lower()).split())

    overlap_counts = [sum(1 for token in RE_SYMBOLS.sub('', answer.lower()).split() if token in tokens)
                      for answer in answers]

    max_count = max(overlap_counts)

    return random.choice([count for count in overlap_counts if count == max_count])


def main():
    with open('data/folds/fold0_test.json') as file:
        video_dict = json.load(file)

    total = correct = 0
    for video in video_dict.values():
        for video_question in video['questions']:
            if index_of_max_overlapping(video_question['question'], video_question['answers']) \
                    == video_question['correct_index']:
                correct += 1
            total += 1

    print(f"Accuracy of the Matching baseline: {correct / total:.4f}")


if __name__ == '__main__':
    main()
