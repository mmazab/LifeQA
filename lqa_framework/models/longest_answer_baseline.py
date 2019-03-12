from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
import numpy
from overrides import overrides
import torch
import torch.nn.functional as F


def create_logits_longest(answers):
    # This function will return the location of max length answer in answers
    max_num = len(max(answers, key=lambda x: len(x)))

    logits = []
    max_choices = []
    # Make a list of all the max length answers
    for index, ans in enumerate(answers):
        logits.append(0)
        if len(ans) == max_num:
            max_choices.append(index)

    # Make the solution a random max length answer
    logits[random.choice(max_choices)] = 1
    return logits

@Model.register('longest_answer_baseline')
class LongestAnswerBaseline(Model):
    """This ``Model`` performs question answering. We assume we're given the video/question/set of answers and we
    predict the correct answer.

    The basic model structure: we take the answers and return the longest one as correct."""

    def __init__(self, vocab: Vocabulary,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.num_classes = self.vocab.get_vocab_size('labels')
        self.loss = torch.nn.CrossEntropyLoss()

        self.metrics = {'accuracy': CategoricalAccuracy()}
        initializer(self)

    @overrides
    def forward(self, answers: Dict[str, torch.LongTensor],
                label: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

        logits = create_logits_longest(answers)

        output_dict = {'logits': logits}

        if label is not None:
            output_dict['loss'] = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)

        return output_dict 

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Does a simple argmax over the class probabilities, converts indices to string labels, and adds a ``'label'``
        key to the dictionary with the result. """
        output_dict['class_probabilities'] = output_dict['logits']
        
        argmax_indices = numpy.argmax(output_dict['logits'], axis=-1)
        output_dict['label'] = torch.Tensor([self.vocab.get_token_from_index(x, namespace='labels')
                                             for x in argmax_indices])
        return output_dict


    @overrides
    def get_metrics(self, reset: Optional[bool] = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}