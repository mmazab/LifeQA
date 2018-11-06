from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F

from .time_distributed_rnn import TimeDistributedRNN


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
        self.answers_encoder = TimeDistributedRNN(answers_encoder)
        self.classifier_feedforward = TimeDistributed(classifier_feedforward)

        self.metrics = {'accuracy': CategoricalAccuracy()}
        self.loss = torch.nn.CrossEntropyLoss()  # TODO: HingeEmbeddingLoss()

        initializer(self)

    @overrides
    def forward(self, question: Dict[str, torch.LongTensor], answers: Dict[str, torch.LongTensor],
                captions: Dict[str, torch.LongTensor], video_features: torch.Tensor, frame_count: torch.Tensor,
                label: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """Does the forward pass.

        Parameters
        ----------
        question : Dict[str, Variable], required	The output of ``TextField.as_array()``.
        answers : Dict[str, Variable], required		The output of ``TextField.as_array()``.
        captions : Dict[str, Variable], required 	The output of ``TextField.as_array()``.
        video_features : torch.Tensor, required     The video features.
        frame_count : torch.Tensor, required        The frame count.
        label : Variable, optional (default = None)	A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        scores : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing the scores for each answer.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        batch_size = len(video_features)
        # noinspection PyProtectedMember
        num_layers = self.video_encoder._module.num_layers
        num_answers = answers['tokens'].shape[1]  # This supposes a fixed number of answers.

        frame_count = frame_count.reshape(-1)  # We have to do this because ArrayFields do not support scalars.

        video_features_mask = util.get_mask_from_sequence_lengths(frame_count, int(max(frame_count).item()))
        encoded_video = self.video_encoder(video_features, mask=video_features_mask)
        encoded_video = encoded_video.reshape(num_layers, batch_size, -1)

        # TODO: how to obtain all layers last hidden layer? Neither seq2vec nor seq2seq give it.

        embedded_question = self.text_field_embedder(question)
        question_mask = util.get_text_field_mask(question)
        # Passing the hidden state with LSTM doesn't work. This is because allennlp seq2vec wrapper only keeps h
        # (the hidden state) but not the cell state c. A workaround should be done if we want to chain LSTMs.
        encoded_question = self.question_encoder(embedded_question, question_mask, hidden_state=encoded_video)
        encoded_question = encoded_question.reshape(num_layers, batch_size, 1, -1)\
            .expand(-1, -1, num_answers, -1)

        embedded_answers = self.text_field_embedder(answers)
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        encoded_answers = self.answers_encoder(embedded_answers, answers_mask, hidden_state=encoded_question)

        scores = self.classifier_feedforward(encoded_answers).squeeze(2)

        output_dict = {'scores': scores}

        if label is not None:
            output_dict['loss'] = self.loss(scores, label)
            for metric in self.metrics.values():
                metric(scores, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Does a simple argmax over the class probabilities, converts indices to string labels, and adds a ``'label'``
        key to the dictionary with the result. """
        class_probabilities = F.softmax(output_dict['scores'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        output_dict['label'] = torch.Tensor([self.vocab.get_token_from_index(x, namespace='labels')
                                             for x in argmax_indices])
        return output_dict

    @overrides
    def get_metrics(self, reset: Optional[bool] = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
