import random
from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
import numpy as np
from overrides import overrides
import torch


@Model.register('longest_answer_baseline')
class LongestAnswerBaseline(Model):
    """This ``Model`` performs question answering. We assume we're given the video/question/set of answers and we
    predict the correct answer.

    The basic model structure: we take the answers and return the longest one as correct."""

    def __init__(self, vocab: Vocabulary,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.loss = torch.nn.CrossEntropyLoss()

        self.metrics = {'accuracy': CategoricalAccuracy()}
        initializer(self)

    @overrides
    def forward(self, answers: Dict[str, torch.LongTensor], label: Optional[torch.LongTensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:

        logits = self.create_logits_longest(answers)

        output_dict = {'logits': logits}

        if label is not None:
            output_dict['loss'] = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)

        return output_dict

    def create_logits_longest(self, answers):
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        answers_lengths_in_words = util.get_lengths_from_binary_sequence_mask(answers_mask)

        # noinspection PyUnresolvedReferences
        logits = torch.zeros(answers['tokens'].shape[0:2])

        for i_instance in range(answers['tokens'].shape[0]):
            answer_lengths = []
            for i_answer in range(answers['tokens'].shape[1]):
                answer_length = 0
                # noinspection PyTypeChecker
                for i_token in range(answers_lengths_in_words[i_instance, i_answer]):
                    answer_length += len(self.vocab.get_token_from_index(answers['tokens'][i_instance, i_answer,
                                                                                           i_token].item())) + 1
                answer_lengths.append(answer_length)

            # This function will return the location of max length answer in answers
            max_num = max(answer_lengths)

            max_choices = []
            # Make a list of all the max length answers
            for index, ans in enumerate(answer_lengths):
                if ans == max_num:
                    max_choices.append(index)

            # Make the solution a random max length answer
            logits[i_instance, random.choice(max_choices)] = 1  # / len(max_choices)
        return logits

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Does a simple argmax over the class probabilities, converts indices to string labels, and adds a ``'label'``
        key to the dictionary with the result. """
        output_dict['class_probabilities'] = output_dict['logits']

        argmax_indices = np.argmax(output_dict['logits'], axis=-1)
        output_dict['label'] = torch.Tensor([self.vocab.get_token_from_index(x, namespace='labels')
                                             for x in argmax_indices])
        return output_dict

    @overrides
    def get_metrics(self, reset: Optional[bool] = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
