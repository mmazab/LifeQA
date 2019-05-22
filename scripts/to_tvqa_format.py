#!/usr/bin/env python
"""Script to convert LifeQA data to TVQA format."""
import json
import os
import pathlib
import pickle
from typing import Any, Dict, Iterable, List, Tuple

import _jsonnet
import jsonlines

TVQA_DATA_FOLDER = 'data/tvqa_format'
OBJECTS_DIR = 'data/lqa_objects'


def load_dataset(file_path: str) -> Dict[str, Any]:
    return json.loads(_jsonnet.evaluate_file(file_path))


def load_datasets() -> Iterable[Tuple[str, Dict[str, Any]]]:
    for filename_suffix in ['train', 'dev', 'test']:
        file_path = f'data/lqa_{filename_suffix}.json'
        yield file_path, load_dataset(file_path)


def to_tvqa_format(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{
        'vid_name': video_id,
        'qid': question_dict['q_id'],
        'q': question_dict['question'],
        'answer_idx': question_dict['correct_index'],
        'a0': question_dict['answers'][0],
        'a1': question_dict['answers'][1],
        'a2': question_dict['answers'][2],
        'a3': question_dict['answers'][3],
    } for video_id, video_dict in dataset.items() for question_dict in video_dict['questions']]


def save_tvqa_dataset(dataset: List[Dict[str, Any]], file_path: str) -> None:
    with jsonlines.open(file_path, 'w') as writer:
        writer.write_all(dataset)


def format_timedelta(total_seconds: float) -> str:
    hours, reminder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(reminder, 60)
    milliseconds = int(1000 * (total_seconds - seconds))
    return f'{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}'


def save_subtitles(dataset: Dict[str, Any], dir_path: str) -> None:
    for video_id, video_dict in dataset.items():
        with open(f'{os.path.join(dir_path, video_id)}.srt', 'w') as file:
            if video_dict['automatic_captions']:
                for i, caption_dict in enumerate(video_dict['automatic_captions']):
                    file.write(f'{i + 1}\n'
                               f'{format_timedelta(caption_dict["words"][0]["start"])} -->'
                               f' {format_timedelta(caption_dict["words"][-1]["end"])}\n'
                               f'{caption_dict["transcript"]}\n\n')


def save_objects(output_path: str) -> None:
    vcpt = {}
    for filepath in os.listdir(OBJECTS_DIR):
        video_id = filepath.split('.')[0]
        with open(os.path.join(OBJECTS_DIR, filepath)) as file:
            vcpt[video_id.encode()] = json.load(file)

    vcpt = {key: [' , '.join(objects).encode() for frame_index, objects in vcpt[key]] for key in vcpt}

    with open(output_path, 'wb') as file:
        pickle.dump(vcpt, file, protocol=2)


def main() -> None:
    pathlib.Path(TVQA_DATA_FOLDER).mkdir(exist_ok=True)
    pathlib.Path(TVQA_DATA_FOLDER, 'tvqa_qa_release').mkdir(exist_ok=True)
    pathlib.Path(TVQA_DATA_FOLDER, 'tvqa_subtitles').mkdir(exist_ok=True)

    for file_path, dataset in load_datasets():
        output_file_path = file_path \
            .replace('data', os.path.join(TVQA_DATA_FOLDER, 'tvqa_qa_release')) \
            .replace('json', 'jsonl')
        save_tvqa_dataset(to_tvqa_format(dataset), output_file_path)

        save_subtitles(dataset, os.path.join(TVQA_DATA_FOLDER, 'tvqa_subtitles'))

    save_objects(os.path.join(TVQA_DATA_FOLDER, 'det_visual_concepts_hq.pickle'))


if __name__ == '__main__':
    main()
