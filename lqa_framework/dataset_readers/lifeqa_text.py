from typing import Dict
import json
import logging
from overrides import overrides
import tqdm
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from IPython import embed


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("lqa_text")
class LqaTextDatasetReader(DatasetReader):
	"""
	Reads a JSON-lines file containing questions and answers, and creates a
	dataset suitable for question answering.
	"""
	def __init__(self,
				 lazy: bool = False,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None) -> None:
		super().__init__(lazy)
		self._tokenizer = tokenizer or WordTokenizer()
		self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

	@overrides
	def _read(self, file_path):
		with open(cached_path(file_path), "r") as data_file:
			videos = json.load(data_file)
			logger.info("Reading instances in file at: %s", file_path)
			for v in videos:
				questions = videos[v]['questions']
				captions  = videos[v]['captions']
				for q in questions:
					question = q['question']
					q_id	 = q['q_id']
					answers  = q['answers']
					correct	 = q['correct_index']
					yield self.text_to_instance(q_id, question, answers, captions, correct)

	@overrides
	def text_to_instance(self, qid: str, question: str, answers: list, captions:dict, correct: int) -> Instance:  # type: ignore
		# pylint: disable=arguments-differ

		tokenized_question = self._tokenizer.tokenize(question)
		tokenized_answers = [ self._tokenizer.tokenize(a) for a in answers]
		answers_field = []
		cc_field = []
		unroll = True

		if not captions:
			tokenized_cc = [ self._tokenizer.tokenize('') ]
		else:
			if unroll:	tokenized_cc = [self._tokenizer.tokenize( ' '.join([ c['transcript']  for c in captions ]) ) ]
			else:	tokenized_cc = [self._tokenizer.tokenize( c['transcript'] ) for c in captions ] 

		question_field	= TextField(tokenized_question, self._token_indexers)
		answers_field	= ListField ( [ TextField(answer, self._token_indexers)  for answer in tokenized_answers ] )
		cc_field		= ListField ( [ TextField(c, self._token_indexers) for c in tokenized_cc ] )
		
		fields = {'question': question_field, 'answers': answers_field, 'closed_captions': cc_field }
		if correct is not None:
			fields['label'] = LabelField(str(correct))

		return Instance(fields)

