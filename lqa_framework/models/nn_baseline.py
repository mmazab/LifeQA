from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from IPython import embed


@Model.register("lqa_baseline")
class LqaBaselineClassifier(Model):
	"""
	This ``Model`` performs question answering. We assume we're given the video/question/set of answers 
	and we predict the correct answer.

	The basic model structure: 
		we embed the question using lstm/cnn, same for the answers, and closed captions using a separate 
		Seq2VecEncoders getting a single vector representing the content of each.  
	We'll then fuse the questions and story vectors and the pass the result through a layer, the output of which should be closest to the correct answer
	"""
	def __init__(self, vocab: Vocabulary,
				 text_field_embedder: TextFieldEmbedder,
				 question_encoder: Seq2VecEncoder,
				 answers_encoder: Seq2VecEncoder,
				 captions_encoder: Seq2VecEncoder,
				 classifier_feedforward: FeedForward,
				 initializer: InitializerApplicator = InitializerApplicator(),
				 regularizer: Optional[RegularizerApplicator] = None) -> None:

		super(LqaBaselineClassifier, self).__init__(vocab, regularizer)

		self.text_field_embedder = text_field_embedder
		self.num_classes = self.vocab.get_vocab_size("labels")

		self.question_encoder = question_encoder
		self.answers_encoder  = TimeDistributed( answers_encoder )
		self.captions_encoder = TimeDistributed( captions_encoder)
		self.classifier_feedforward = classifier_feedforward
		#self.classifier_feedforward = TimeDistributed ( classifier_feedforward )

		self._encoding_dim = captions_encoder.get_output_dim()
		self.ques_cap_att = LinearMatrixAttention(self._encoding_dim, self._encoding_dim, 'x,y,x*y')

		self.metrics = { "accuracy": CategoricalAccuracy() }
		self.loss = torch.nn.CrossEntropyLoss()
		initializer(self)

	@overrides
	def forward(self,  # type: ignore
				question: Dict[str, torch.LongTensor],
				answers: Dict[str, torch.LongTensor],
				closed_captions: Dict[str, torch.LongTensor],
				label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
		# pylint: disable=arguments-differ
		"""
		Parameters
		----------
		questions : Dict[str, Variable], required	The output of ``TextField.as_array()``.
		answers : Dict[str, Variable], required		The output of ``TextField.as_array()``.
		captions : Dict[str, Variable], required 	The output of ``TextField.as_array()``.
		label : Variable, optional (default = None)	A variable representing the label for each instance in the batch.
		Returns
		-------
		An output dictionary consisting of:
		class_probabilities : torch.FloatTensor
			A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
			label classes for each instance.
		loss : torch.FloatTensor, optional
			A scalar loss to be optimised.
		"""
		embedded_question = self.text_field_embedder(question)
		question_mask = util.get_text_field_mask(question)
		encoded_question = self.question_encoder(embedded_question, question_mask)

		embedded_answers= self.text_field_embedder(answers)
		answers_mask	= util.get_text_field_mask(answers, num_wrapping_dims= 1 )
		encoded_answers = self.answers_encoder(embedded_answers, answers_mask)
		embedded_captions	= self.text_field_embedder(closed_captions)
		captions_mask		= util.get_text_field_mask(closed_captions, num_wrapping_dims= 1 )
		encoded_captions	= self.captions_encoder(embedded_captions, captions_mask)
		encoded_captions	= encoded_captions.squeeze (1)


		# TODO: should add attention between question and captions
		# TODO: should add attention between the output of the (questions and captions) and different answers

		fuse_cq	= self.classifier_feedforward(torch.cat([encoded_captions, encoded_question], dim=-1))
		logits = torch.bmm ( encoded_answers, fuse_cq.unsqueeze(2)  ).squeeze(2)
		
		output_dict = {'logits': logits}
		if label is not None:
			loss = self.loss(logits, label)
			for metric in self.metrics.values():
				metric(logits, label)
			output_dict["loss"] = loss

		return output_dict

	@overrides
	def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		"""
		Does a simple argmax over the class probabilities, converts indices to string labels, and
		adds a ``"label"`` key to the dictionary with the result.
		"""
		class_probabilities = F.softmax(output_dict['logits'], dim=-1)
		output_dict['class_probabilities'] = class_probabilities

		predictions = class_probabilities.cpu().data.numpy()
		argmax_indices = numpy.argmax(predictions, axis=-1)
		labels = [self.vocab.get_token_from_index(x, namespace="labels")
				  for x in argmax_indices]
		output_dict['label'] = labels
		return output_dict

	@overrides
	def get_metrics(self, reset: bool = False) -> Dict[str, float]:
		return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

