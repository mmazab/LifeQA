from typing import Dict

from allennlp.models.model import Model
from overrides import overrides
import torch

from . import longest_answer
from .simple_baseline import SimpleBaseline


@Model.register('shortest_answer')
class ShortestAnswer(SimpleBaseline):
    """This ``Model`` performs question answering. We assume we're given the video/question/set of answers and we
    predict the correct answer.

    The basic model structure: we take the answers and return the shortest one (character-wise) as correct."""

    @overrides
    def _compute_logits(self, question: Dict[str, torch.LongTensor],
                        answers: Dict[str, torch.LongTensor]) -> torch.Tensor:
        # noinspection PyProtectedMember
        return - longest_answer.answers_lengths(answers, self.vocab)
