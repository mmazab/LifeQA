#!/usr/bin/env python
from collections import Counter, defaultdict
import json
from typing import Any, Dict, Iterator, List

from conllu import parse

MAX_INDEX = 2  # We suppose all sentences have length at least MAX_INDEX + 1.


def create_empty_counter(i: int = 0) -> Dict[str, Any]:
    empty_counter = {'counter': Counter()}
    if i < MAX_INDEX:
        empty_counter['sub_counters'] = defaultdict(lambda: create_empty_counter(i=i + 1))
    return empty_counter


def count(sentence: Iterator[str], counter_dict: Dict[str, Any]) -> None:
    token = next(sentence)

    counter_dict['counter'][token] += 1

    if 'sub_counters' in counter_dict:
        count(sentence, counter_dict['sub_counters'][token])


def to_plot_format(counter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    return_list = []

    counter_dict_counter = counter_dict['counter']
    for token, size in counter_dict_counter.most_common():
        token_dict = {'name': token}
        sub_counters = counter_dict.get('sub_counters')
        if sub_counters:
            token_dict['children'] = to_plot_format(sub_counters[token])
        else:
            token_dict['size'] = size
        return_list.append(token_dict)

    return return_list


def normalize(token: Dict[str, Any]) -> str:
    token_form = token['form'].lower()

    if token_form == '\'s':  # All the 's in the graph are actually "is".
        token_form = 'is'

    return token_form


def main():
    with open('output/questions') as file:
        questions_output = file.read()

    sentences = parse(questions_output)

    counter_dict = create_empty_counter()
    for sentence in sentences:
        count((normalize(token) for token in sentence), counter_dict)

    count_list = to_plot_format(counter_dict)

    with open('sunburst_plots/sunburst_plot_data.json', 'w') as file:
        json.dump({'name': '', 'children': count_list}, file, indent=2)


if __name__ == '__main__':
    main()
