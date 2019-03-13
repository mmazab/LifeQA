from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
import torch


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
