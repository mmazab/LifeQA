import json
import pathlib
import random
from typing import Any, Dict, Generator, Iterable, List, Optional

import _jsonnet
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, ListField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import h5py
import numpy as np
from overrides import overrides


class GeneratorWithSize:
    # See https://stackoverflow.com/a/7460929/1165181
    def __init__(self, gen: Generator, size: int) -> None:
        self.gen = gen
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Generator:
        return self.gen


def generator_with_size(func):
    def wrapper(*args):
        result = func(*args)
        size = next(result)
        if size:
            return GeneratorWithSize(result, size)
        else:
            return result

    return wrapper


@DatasetReader.register('lqa')
class LqaDatasetReader(DatasetReader):
    """Provides a dataset suitable for VideoQA, called LifeQA.

    Parameters
    ----------
    lazy : bool, optional (default=False)
        If this is true, ``read()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``read()`` returns a list.
    tokenizer : Optional[Tokenizer], optional (default=WordTokenizer())
        Tokenizer with which the question, answers and captions will be tokenized.
    token_indexers : Optional[Dict[str, TokenIndexer]], optional (default={'tokens': SingleIdTokenIndexer()})
        Dictionary of token indexers for the question, answers and captions.
    video_features_to_load : Optional[List[str]], optional (default=None)
        List of feature names to load. They will be concatenated.
    check_missing_video_features : Optional[bool], optional (default=True)
        If this is true, it will raise and exception if any video features cannot be loaded.
    frame_step : int, optional (default=1)
        Step to take frames. For example, frame step 4 means that one frame out of four will be taken. The start index
        is random.
    join_question_and_answers : bool, optional (default=False)
        If true, the ``read`` method returns the questions along with each of their answers in a TextField, instead of
        separate.
    small_sample : bool, optional (default=False)
        If true, it returns a small random sample of the dataset instead of all the instances.
    return_metadata : bool, optional (default=False)
        If true, it returns metadata such as the original question and answers texts along with the tokenized versions.
    unroll_captions : bool, optional (default=True)
        If true, it returns the caption lines as a single line (as if it were a single sentence).
    """

    FEATURES_PATH = pathlib.Path('data/features')
    MODEL_NAME_TO_PRETRAINED_FILE_DICT = {
        'c3d-conv5b': FEATURES_PATH / 'LifeQA_C3D_conv5b.hdf5',
        'c3d-fc6': FEATURES_PATH / 'LifeQA_C3D_fc6.hdf5',
        'c3d-fc7': FEATURES_PATH / 'LifeQA_C3D_fc7.hdf5',
        'i3d-avg-pool': FEATURES_PATH / 'LifeQA_I3D_avg_pool.hdf5',
        'resnet-pool5': FEATURES_PATH / 'LifeQA_RESNET_pool5.hdf5',
        'resnet-res5c': FEATURES_PATH / 'LifeQA_RESNET_res5c.hdf5',
        'resof': FEATURES_PATH / 'LifeQA_RESOF_pool5.hdf5',
    }
    SMALL_SAMPLE_VIDEO_COUNT = 5
    SMALL_SAMPLE_Q_PER_VIDEO = 3

    def __init__(self, lazy: bool = False, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 video_features_to_load: Optional[List[str]] = None, check_missing_video_features: bool = True,
                 frame_step: int = 1, join_question_and_answers: bool = False, small_sample: bool = False,
                 return_metadata: bool = False, unroll_captions: bool = True) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.video_features_to_load = video_features_to_load or []
        self.check_missing_video_features = check_missing_video_features
        self.frame_step = frame_step
        self.join_question_and_answers = join_question_and_answers
        self.small_sample = small_sample
        self.return_metadata = return_metadata
        self.unroll_captions = unroll_captions

    def _count_questions(self, video_dict: Dict[str, Any], features_files: Iterable[h5py.File]) -> int:
        return sum(min(len(video['questions']), self.SMALL_SAMPLE_Q_PER_VIDEO)
                   for video_id, video in video_dict.items()
                   if not self.video_features_to_load or self.check_missing_video_features
                   or video_id in features_files)

    @generator_with_size
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        features_files = [h5py.File(self.MODEL_NAME_TO_PRETRAINED_FILE_DICT[video_feature], 'r')
                          for video_feature in self.video_features_to_load]

        video_dict = json.loads(_jsonnet.evaluate_file(cached_path(file_path)))

        if self.small_sample:
            video_dict = {key: video_dict[key]
                          for key in random.sample(list(video_dict), self.SMALL_SAMPLE_VIDEO_COUNT)}

        yield self._count_questions(video_dict, features_files)

        for video_id, video in video_dict.items():
            if not self.video_features_to_load or self.check_missing_video_features or video_id in features_files:
                question_dicts = video['questions']

                if self.small_sample:
                    question_dicts = random.sample(question_dicts, self.SMALL_SAMPLE_Q_PER_VIDEO) \
                        if len(question_dicts) > self.SMALL_SAMPLE_Q_PER_VIDEO else question_dicts

                captions = video.get('manual_captions') or video['automatic_captions']
                parent_video_id = video['parent_video_id']

                if self.video_features_to_load:
                    initial_frame = random.randint(0, self.frame_step - 1)
                    video_features = np.concatenate([features_file[video_id][initial_frame::self.frame_step]
                                                     for features_file in features_files], axis=1)
                else:
                    video_features = None

                for question_dict in question_dicts:
                    question_text = question_dict['question']
                    answers = question_dict['answers']
                    correct_index = question_dict['correct_index']
                    yield self.text_to_instance(question_text, answers, parent_video_id, correct_index, captions,
                                                video_features)

        for features_file in features_files:
            features_file.close()

    @overrides
    def text_to_instance(self, question: str, answers: List[str], parent_video_id: str,
                         correct_index: Optional[int] = None, captions: Optional[List[Dict[str, Any]]] = None,
                         video_features: Optional[np.ndarray] = None) -> Instance:
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_answers = [self._tokenizer.tokenize(answer) for answer in answers]

        if captions:
            if self.unroll_captions:
                tokenized_captions = [self._tokenizer.tokenize(' '.join(caption['transcript'] for caption in captions))]
            else:
                tokenized_captions = (self._tokenizer.tokenize(caption['transcript']) for caption in captions)
        else:
            tokenized_captions = [self._tokenizer.tokenize('')]

        fields = {
            'captions': ListField([TextField(caption, self._token_indexers) for caption in tokenized_captions]),
            'parent_video_id': LabelField(parent_video_id, label_namespace='parent_video_id_labels'),
        }

        if self.return_metadata:
            fields['metadata'] = MetadataField({
                'original_question': question,
                'original_answers': answers,
                'tokenized_question': [token.text for token in tokenized_question],
                'tokenized_answers': [[token.text for token in tokenized_answer]
                                      for tokenized_answer in tokenized_answers],
            })

        if self.join_question_and_answers:
            fields['question_and_answers'] = ListField([TextField(tokenized_question + answer, self._token_indexers)
                                                        for answer in tokenized_answers])
        else:
            fields['question'] = TextField(tokenized_question, self._token_indexers)
            fields['answers'] = ListField([TextField(answer, self._token_indexers) for answer in tokenized_answers])

        if video_features is not None:
            fields['video_features'] = ArrayField(video_features)
            fields['frame_count'] = ArrayField(np.asarray(len(video_features), dtype=np.int32))

        if correct_index is not None:
            fields['label'] = LabelField(correct_index, skip_indexing=True)

        return Instance(fields)
