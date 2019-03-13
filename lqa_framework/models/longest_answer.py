from typing import Dict

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from overrides import overrides
import torch

from .simple_baseline import SimpleBaseline


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
