import json
import logging
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

    def __init__(self, lazy: bool = False, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 load_video_features: Optional[bool] = False,
                 check_missing_video_features: Optional[bool] = True) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.load_video_features = load_video_features
        self.check_missing_video_features = check_missing_video_features

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        if self.load_video_features:
            logger.info("Reading video features of instances")
            features_file = h5py.File('data/features/LifeQA_RESNET_pool5.hdf5')

        with open(cached_path(file_path)) as data_file:
            logger.info("Reading instances in file at: %s", file_path)
            video_dict = json.load(data_file)
            for video_id in video_dict:
                # noinspection PyUnboundLocalVariable
                if not self.load_video_features or self.check_missing_video_features or video_id in features_file:
                    question_dicts = video_dict[video_id]['questions']
                    captions = video_dict[video_id]['captions']

                    if self.load_video_features:
                        # noinspection PyUnboundLocalVariable
                        video_features = features_file[video_id].value
                    else:
                        video_features = None

                    for question_dict in question_dicts:
                        question_text = question_dict['question']
                        answers = question_dict['answers']
                        correct_index = question_dict['correct_index']
                        yield self.text_to_instance(question_text, answers, correct_index, captions, video_features)

        if self.load_video_features:
            features_file.close()

    @overrides
    def text_to_instance(self, question: str, answers: List[str], correct_index: Optional[int] = None,
                         captions: Optional[Dict[str, Any]] = None, video_features: Optional[np.ndarray] = None,
                         unroll: Optional[bool] = True) -> Instance:
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_answers = (self._tokenizer.tokenize(a) for a in answers)

        if captions:
            if unroll:
                # noinspection PyTypeChecker
                tokenized_captions = [self._tokenizer.tokenize(' '.join(caption['transcript'] for caption in captions))]
            else:
                # noinspection PyTypeChecker
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
            fields['frame_count'] = ArrayField(np.asarray([len(video_features)], dtype=np.int32))

        if correct_index is not None:
            fields['label'] = LabelField(str(correct_index))

        return Instance(fields)
