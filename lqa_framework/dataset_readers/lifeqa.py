import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register('lqa')
class LqaDatasetReader(DatasetReader):
    """Reads a JSON file containing questions and answers, and creates a dataset suitable for QA. """

    def __init__(self, lazy: bool = False, tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path)) as data_file:
            videos = json.load(data_file)
            logger.info("Reading instances in file at: %s", file_path)
            for v in videos:
                questions = videos[v]['questions']
                captions = videos[v]['captions']
                for q in questions:
                    question = q['question']
                    q_id = q['q_id']
                    answers = q['answers']
                    correct_index = q['correct_index']
                    yield self.text_to_instance(q_id, question, answers, correct_index, captions)

    @overrides
    def text_to_instance(self, qid: str, question: str, answers: List[str], correct_index: Optional[int] = None,
                         captions: Optional[Dict[str, Any]] = None, unroll: Optional[bool] = True) -> Instance:
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_answers = (self._tokenizer.tokenize(a) for a in answers)

        if captions:
            if unroll:
                # noinspection PyTypeChecker
                tokenized_captions = [self._tokenizer.tokenize(' '.join(c['transcript'] for c in captions))]
            else:
                # noinspection PyTypeChecker
                tokenized_captions = (self._tokenizer.tokenize(c['transcript']) for c in captions)
        else:
            tokenized_captions = [self._tokenizer.tokenize('')]

        question_field = TextField(tokenized_question, self._token_indexers)
        answers_field = ListField([TextField(answer, self._token_indexers) for answer in tokenized_answers])
        captions_field = ListField([TextField(c, self._token_indexers) for c in tokenized_captions])

        fields = {'question': question_field, 'answers': answers_field, 'captions': captions_field}
        if correct_index is not None:
            fields['label'] = LabelField(str(correct_index))

        return Instance(fields)
