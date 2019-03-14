from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, TimeDistributed
from allennlp.modules.similarity_functions import CosineSimilarity
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
import torch

import lqa_framework.models


class SimpleBaseline(Model):
    def __init__(self, vocab: Vocabulary,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.loss = torch.nn.CrossEntropyLoss()

        self.metrics = {'accuracy': CategoricalAccuracy()}

        initializer(self)

    @overrides
    def forward(self, question: Dict[str, torch.LongTensor], answers: Dict[str, torch.LongTensor],
                label: Optional[torch.LongTensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        logits = self._compute_logits(question, answers)

        output_dict = {'logits': logits}

        if label is not None:
            output_dict['loss'] = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)

        return output_dict

    def _compute_logits(self, question: Dict[str, torch.LongTensor],
                        answers: Dict[str, torch.LongTensor]) -> torch.Tensor:
        raise NotImplementedError

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Does a simple argmax over the class probabilities, converts indices to string labels, and adds a ``'label'``
        key to the dictionary with the result. """
        logits = output_dict['logits']

        output_dict['class_probabilities'] = logits / logits.sum(dim=1)

        predicted_indices = torch.argmax(logits, dim=1)
        output_dict['label'] = torch.Tensor([self.vocab.get_token_from_index(token_index, namespace='labels')
                                             for token_index in predicted_indices])  # FIXME: namespace labels is wrong

        return output_dict

    @overrides
    def get_metrics(self, reset: Optional[bool] = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}


def answers_lengths(answers: Dict[str, torch.LongTensor], vocab: Vocabulary):
    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    return torch.tensor([[sum(0 if token_index == 0
                              else 1 + len(vocab.get_token_from_index(token_index.item()))
                              for token_index in answer)
                          for answer in instance]
                         for instance in answers['tokens']], dtype=torch.float)


@Model.register('longest_answer')
class LongestAnswer(SimpleBaseline):
    """This ``Model`` performs question answering. We assume we're given the video/question/set of answers and we
    predict the correct answer.

    The basic model structure: we take the answers and return the longest one (character-wise) as correct."""

    @overrides
    def _compute_logits(self, question: Dict[str, torch.LongTensor],
                        answers: Dict[str, torch.LongTensor]) -> torch.Tensor:
        return answers_lengths(answers, self.vocab)


@Model.register('shortest_answer')
class ShortestAnswer(SimpleBaseline):
    """This ``Model`` performs question answering. We assume we're given the video/question/set of answers and we
    predict the correct answer.

    The basic model structure: we take the answers and return the shortest one (character-wise) as correct."""

    @overrides
    def _compute_logits(self, question: Dict[str, torch.LongTensor],
                        answers: Dict[str, torch.LongTensor]) -> torch.Tensor:
        # noinspection PyProtectedMember
        return - lqa_framework.models.simple_baseline.answers_lengths(answers, self.vocab)


@Model.register('most_similar_answer')
class MostSimilarAnswer(SimpleBaseline):
    """This ``Model`` performs question answering. We assume we're given the video/question/set of answers and we
    predict the correct answer.

    The basic model structure: we take the answers and return the one closest in cosine similarity to the question
    (by averaging the word embeddings)."""

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder,
                 question_encoder: Seq2VecEncoder, answers_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, initializer, regularizer)
        self.text_field_embedder = text_field_embedder
        self.question_encoder = question_encoder
        self.answers_encoder = TimeDistributed(answers_encoder)
        self.cosine_similarity = TimeDistributed(CosineSimilarity())

    @overrides
    def _compute_logits(self, question: Dict[str, torch.LongTensor],
                        answers: Dict[str, torch.LongTensor]) -> torch.Tensor:
        embedded_question = self.text_field_embedder(question)
        question_mask = util.get_text_field_mask(question)
        encoded_question = self.question_encoder(embedded_question, question_mask)

        embedded_answers = self.text_field_embedder(answers)
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        encoded_answers = self.answers_encoder(embedded_answers, answers_mask)

        batch_size, embed_dim = encoded_question.shape
        repeated_encoded_question = encoded_question.view(batch_size, 1, embed_dim).expand(encoded_answers.size())
        logits = self.cosine_similarity(repeated_encoded_question, encoded_answers)
        logits[torch.isnan(logits)] = 0
        return logits
