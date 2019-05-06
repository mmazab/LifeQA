from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
import torch
import torch.nn


class LqaModel(Model):
    """Base model to perform Video Question Answering (VideoQA) on the LifeQA dataset. We assume we're given the
    video, question and set of candidate answers (among other things) and we predict the correct answer.
    """

    def __init__(self, vocab: Vocabulary,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        self.metrics = {'accuracy': CategoricalAccuracy()}

    @overrides
    def forward(self, question_and_answers: Dict[str, torch.Tensor], question: Dict[str, torch.Tensor],
                answers: Dict[str, torch.Tensor], captions: Dict[str, torch.Tensor],
                video_features: Optional[torch.Tensor] = None, frame_count: Optional[torch.Tensor] = None,
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Computes the answer scores for the classification, and optionally the loss value if the label is provided.

        Parameters
        ----------
        question_and_answers : Dict[str, Variable], required
            The tensor representation of the question along with every candidate answer, for every token indexer of the
            dataset reader. The method can either receive this parameter or `question` along with `answers`.
        question : Dict[str, Variable], required
            The tensor representation of the question for every token indexer of the dataset reader. The method can
            either receive this parameter along with `answers, or `question_and_answers`.
        answers : Dict[str, Variable], required
            The tensor representation of the answers for every token indexer of the dataset reader. The method can
            either receive this parameter along with `question, or `question_and_answers`.
        captions : Dict[str, Variable], required
            The tensor representation of the captions for every token indexer of the dataset reader.
        video_features : torch.Tensor, optional (default=None)
            The tensor representation of the video frames.
        frame_count : Optional[torch.Tensor], optional (default=None)
            The frame count. It must be provided if ``video_features`` is provided.
        label : Optional[torch.Tensor], optional (default=None)
            The index of the correct answer.

        Returns
        -------
        output_dict : Dict[str, torch.Tensor]
            An output dictionary consisting of an entry `scores` (a torch.Tensor of shape ``(batch_size, num_classes)``)
            representing the scores for each answer, and only if ``label`` is not None an entry `loss`
            (a scalar torch.Tensor) representing loss to be optimized.
        """
        raise NotImplementedError

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Takes the result of :func:`forward` and adds an entry `label` with the index of the predicted answer.

        Subclasses should override this method, calling super, and adding the class probabilities in an entry called
        `class_probabilities`.
        """
        output_dict['label'] = torch.argmax(output_dict['scores'], dim=1)
        return output_dict

    @overrides
    def get_metrics(self, reset: Optional[bool] = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
