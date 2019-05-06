from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, TimeDistributed
from allennlp.modules.similarity_functions import CosineSimilarity
from allennlp.nn import RegularizerApplicator, util
from overrides import overrides
import torch
import torch.nn

from .lqa import LqaModel


class SimpleBaseline(LqaModel):
    """A simple baseline base class for the LifeQA dataset that classifies based only on the question and/or the answers
    and that does not need training.
    """

    @overrides
    def forward(self, question: Dict[str, torch.Tensor], answers: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Computes the answer scores for the classification.

        It does not return a loss value even if ``label`` is provided because this baseline is supposed not to have
        trainable parameters.
        """
        scores = self._compute_scores(question, answers)

        output_dict = {'scores': scores}

        if label is not None:
            for metric in self.metrics.values():
                metric(scores, label)

        return output_dict

    def _compute_scores(self, question: Dict[str, torch.Tensor], answers: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the answer scores for the classification."""
        raise NotImplementedError


def answers_lengths(answers: Dict[str, torch.Tensor], vocab: Vocabulary):
    return torch.tensor([[sum(0 if token_index == 0
                              else 1 + len(vocab.get_token_from_index(token_index.item()))
                              for token_index in answer)
                          for answer in instance]
                         for instance in answers['tokens']], dtype=torch.float)


@Model.register('longest_answer')
class LongestAnswer(SimpleBaseline):
    """This ``Model`` returns the character-wise longest answer."""

    @overrides
    def _compute_scores(self, question: Dict[str, torch.Tensor], answers: Dict[str, torch.Tensor]) -> torch.Tensor:
        return answers_lengths(answers, self.vocab)


@Model.register('shortest_answer')
class ShortestAnswer(SimpleBaseline):
    """This ``Model`` returns the character-wise shortest answer."""

    @overrides
    def _compute_scores(self, question: Dict[str, torch.Tensor], answers: Dict[str, torch.Tensor]) -> torch.Tensor:
        return - answers_lengths(answers, self.vocab)


@Model.register('most_similar_answer')
class MostSimilarAnswer(SimpleBaseline):
    """This ``Model`` returns the answer closest to the question in cosine similarity."""

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder,
                 question_encoder: Seq2VecEncoder, answers_encoder: Seq2VecEncoder,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder

        oov_token_index = self.vocab.get_token_index(DEFAULT_OOV_TOKEN, 'tokens')
        # noinspection PyProtectedMember
        text_field_embedder._token_embedders['tokens'].weight[oov_token_index].fill_(0)

        self.question_encoder = question_encoder
        self.answers_encoder = TimeDistributed(answers_encoder)
        self.cosine_similarity = TimeDistributed(CosineSimilarity())

        # Note that no initializer is needed. There is one parameter: the embedding layer weights, but they are loaded
        # from a pretrained file and the OOV word embedding set to zeros. Also note that the latter cannot be done with
        # an ``Initializer`` because it's done after loading the pretrained file, so it'd set everything to zero. We
        # could possibly implement and register a custom initializer that does the same, but it's too much overkill;
        # it's better to just do it here.

    # noinspection PyCallingNonCallable
    @overrides
    def _compute_scores(self, question: Dict[str, torch.Tensor], answers: Dict[str, torch.Tensor]) -> torch.Tensor:
        embedded_question = self.text_field_embedder(question)
        question_mask = util.get_text_field_mask(question)
        encoded_question = self.question_encoder(embedded_question, question_mask)

        embedded_answers = self.text_field_embedder(answers)
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        encoded_answers = self.answers_encoder(embedded_answers, answers_mask)

        batch_size, embed_dim = encoded_question.shape
        repeated_encoded_question = encoded_question.view(batch_size, 1, embed_dim).expand(encoded_answers.size())
        scores = self.cosine_similarity(repeated_encoded_question, encoded_answers)
        scores[torch.isnan(scores)] = 0
        return scores
