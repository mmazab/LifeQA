#!/usr/bin/env python
import pandas as pd

import scripts.util


def main():
    data_dicts = scripts.util.load_data()

    df = pd.read_csv('data/answer_types.csv')

    q_id_to_answer_type = dict(zip(df['q_id'], df['type']))

    for data_dict in data_dicts:
        for video_dict in data_dict.values():
            for question_dict in video_dict['questions']:
                q_id = question_dict['q_id']
                question_dict['answer_type'] = q_id_to_answer_type[q_id]

    scripts.util.save_data(data_dicts)


if __name__ == '__main__':
    main()
