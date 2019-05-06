import json
import pathlib
import random
from typing import Any, Dict, Iterable, List, Optional

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
import h5py
import numpy as np
from overrides import overrides


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
    load_video_features : Optional[bool], optional (default=False)
        If this is true, it will load the features for the video frames.
    check_missing_video_features : Optional[bool], optional (default=True)
        If this is true, it will raise and exception if any video features cannot be loaded.
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

    def __init__(self, lazy: bool = False, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 video_features_to_load: Optional[List[str]] = None, check_missing_video_features: bool = True,
                 frame_step: int = 1, join_question_and_answers: bool = False, small_sample: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.video_features_to_load = video_features_to_load or []
        self.check_missing_video_features = check_missing_video_features
        self.frame_step = frame_step
        self.join_question_and_answers = join_question_and_answers
        self.small_sample = small_sample

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        features_files = [h5py.File(self.MODEL_NAME_TO_PRETRAINED_FILE_DICT[video_feature], 'r')
                          for video_feature in self.video_features_to_load]

        with open(cached_path(file_path)) as data_file:
            video_dict = json.load(data_file)

            if self.small_sample:
                video_dict = {key: video_dict[key] for key in random.sample(list(video_dict), 5)}

            for video_id in video_dict:
                if not self.video_features_to_load or self.check_missing_video_features or video_id in features_files:
                    question_dicts = video_dict[video_id]['questions']

                    if self.small_sample:
                        question_dicts = random.sample(question_dicts, 3) \
                            if len(question_dicts) > 3 else question_dicts

                    captions = video_dict[video_id].get('manual_captions') or video_dict[video_id]['automatic_captions']

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
                        yield self.text_to_instance(question_text, answers, correct_index, captions, video_features)

        for features_file in features_files:
            features_file.close()

    @overrides
    def text_to_instance(self, question: str, answers: List[str], correct_index: Optional[int] = None,
                         captions: Optional[List[Dict[str, Any]]] = None, video_features: Optional[np.ndarray] = None,
                         unroll: Optional[bool] = True) -> Instance:
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_answers = (self._tokenizer.tokenize(a) for a in answers)

        if captions:
            if unroll:
                tokenized_captions = [self._tokenizer.tokenize(' '.join(caption['transcript'] for caption in captions))]
            else:
                tokenized_captions = (self._tokenizer.tokenize(caption['transcript']) for caption in captions)
        else:
            tokenized_captions = [self._tokenizer.tokenize('')]

        fields = {'captions': ListField([TextField(caption, self._token_indexers) for caption in tokenized_captions])}

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
