import json
import logging
import pathlib
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

logger = logging.getLogger(__name__)


@DatasetReader.register('lqa')
class LqaDatasetReader(DatasetReader):
    """Reads a JSON file containing questions and answers, and creates a dataset suitable for QA. """

    FEATURES_PATH = pathlib.Path('data/features')
    MODEL_NAME_TO_PRETRAINED_FILE_DICT = {
        'c3d-fc6': FEATURES_PATH / 'LifeQA_C3D_fc6.hdf5',
        'c3d-fc7': FEATURES_PATH / 'LifeQA_C3D_fc7.hdf5',
        'i3d-avg-pool': FEATURES_PATH / 'LifeQA_I3D_avg_pool.hdf5',
        'resnet-pool5': FEATURES_PATH / 'LifeQA_RESNET_pool5.hdf5',
    }

    def __init__(self, lazy: bool = False, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 video_features_to_load: Optional[str] = None,
                 check_missing_video_features: Optional[bool] = True) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.video_features_to_load = video_features_to_load
        self.check_missing_video_features = check_missing_video_features

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        if self.video_features_to_load:
            logger.info("Reading video features of instances")
            features_file = h5py.File(self.MODEL_NAME_TO_PRETRAINED_FILE_DICT[self.video_features_to_load], 'r')

        with open(cached_path(file_path)) as data_file:
            logger.info("Reading instances in file at: %s", file_path)
            video_dict = json.load(data_file)
            for video_id in video_dict:
                # noinspection PyUnboundLocalVariable
                if not self.video_features_to_load or self.check_missing_video_features or video_id in features_file:
                    question_dicts = video_dict[video_id]['questions']
                    captions = video_dict[video_id]['captions']

                    if self.video_features_to_load:
                        # noinspection PyUnboundLocalVariable
                        video_features = features_file[video_id][()]
                    else:
                        video_features = None

                    for question_dict in question_dicts:
                        question_text = question_dict['question']
                        answers = question_dict['answers']
                        correct_index = question_dict['correct_index']
                        yield self.text_to_instance(question_text, answers, correct_index, captions, video_features)

        if self.video_features_to_load:
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

        question_field = TextField(tokenized_question, self._token_indexers)
        answers_field = ListField([TextField(answer, self._token_indexers) for answer in tokenized_answers])
        captions_field = ListField([TextField(caption, self._token_indexers) for caption in tokenized_captions])

        fields = {
            'question': question_field,
            'answers': answers_field,
            'captions': captions_field,
        }

        if video_features is not None:
            fields['video_features'] = ArrayField(video_features)
            fields['frame_count'] = ArrayField(np.asarray(len(video_features), dtype=np.int32))

        if correct_index is not None:
            fields['label'] = LabelField(correct_index, skip_indexing=True)

        return Instance(fields)
