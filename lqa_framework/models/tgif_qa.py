from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
import torch


@Model.register('tgif_qa')
class TgifQaClassifier(Model):
    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, video_encoder: Seq2VecEncoder,
                 question_encoder: Seq2VecEncoder, answers_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward, initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        # self.num_classes = self.vocab.get_vocab_size('labels')

        self.video_encoder = video_encoder
        self.question_encoder = question_encoder
        self.answers_encoder = TimeDistributed(answers_encoder)
        self.classifier_feedforward = classifier_feedforward

        self.metrics = {'accuracy': CategoricalAccuracy()}
        self.loss = torch.nn.HingeEmbeddingLoss()

        initializer(self)

    # noinspection PyUnresolvedReferences
    @overrides
    def forward(self, question: Dict[str, torch.LongTensor], answers: Dict[str, torch.LongTensor],
                captions: Dict[str, torch.LongTensor], video_features: torch.Tensor,
                label: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """Does the forward pass.

        Parameters
        ----------
        question : Dict[str, Variable], required	The output of ``TextField.as_array()``.
        answers : Dict[str, Variable], required		The output of ``TextField.as_array()``.
        captions : Dict[str, Variable], required 	The output of ``TextField.as_array()``.
        video_features : np.ndarray, required       The video features.
        label : Variable, optional (default = None)	A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        scores : torch.FloatTensor  FIXME
            A tensor of shape ``(batch_size, num_classes)`` representing the scores for each answer.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        encoded_video = self.video_encoder(video_features, mask=None)  # FIXME: should mask because it's a batch

        embedded_question = self.text_field_embedder(question)
        question_mask = util.get_text_field_mask(question)
        encoded_question = self.question_encoder(embedded_question, question_mask, hidden_state=encoded_video)

        embedded_answers = self.text_field_embedder(answers)
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        encoded_answers = self.answers_encoder(embedded_answers, answers_mask, hidden_state=encoded_question)

        scores = self.classifier_feedforward(encoded_answers)

        output_dict = {'scores': scores}
        if label is not None:
            output_dict['loss'] = self.loss(scores, labels)
            for metric in self.metrics.values():
                metric(logits, label)

        return output_dict

    # TODO: decode?

    @overrides
    def get_metrics(self, reset: Optional[bool] = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
