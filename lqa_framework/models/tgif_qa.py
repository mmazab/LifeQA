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
    text_video_mode_options = ['video-text', 'text-video', 'parallel', 'text']
    loss_options = ['hinge', 'cross-entropy']

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder,
                 video_encoder: Optional[Seq2VecEncoder],
                 question_encoder: Seq2VecEncoder, answers_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward, initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, text_video_mode: str = 'video-text',
                 loss: str = 'hinge') -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder

        self.video_encoder = video_encoder
        self.question_encoder = question_encoder
        self.answers_encoder = TimeDistributedRNN(answers_encoder)
        self.classifier_feedforward = TimeDistributed(classifier_feedforward)

        # noinspection PyProtectedMember
        self.num_layers = self.question_encoder._module.num_layers

        if text_video_mode not in self.text_video_mode_options:
            raise ValueError(f"'text_video_mode' should be one of {self.text_video_mode_options}")
        self.text_video_mode = text_video_mode

        if text_video_mode == 'text-video':
            self.video_encoder = TimeDistributedRNN(self.video_encoder)

        if video_encoder is None and text_video_mode != 'text':
            raise ValueError("'video_encoder' can be None only if 'text_video_mode' is set to 'text'")

        encoded_size = question_encoder.get_output_dim()
        self.parallel_feedforward = TimeDistributed(FeedForward(input_dim=encoded_size * 2, num_layers=1,
                                                                hidden_dims=[encoded_size], activations=[lambda x: x]))

        self.metrics = {'accuracy': CategoricalAccuracy()}

        if loss not in self.loss_options:
            raise ValueError(f"'loss' should be one of {self.loss_options}")

        if loss == 'hinge':
            self.loss = torch.nn.HingeEmbeddingLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self, question: Dict[str, torch.LongTensor], answers: Dict[str, torch.LongTensor],
                captions: Dict[str, torch.LongTensor], video_features: Optional[torch.Tensor] = None,
                frame_count: Optional[torch.Tensor] = None,
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
        batch_size = list(question.values())[0].shape[0]  # Grabs any of the dict values available (note it keys depend
        #   on the actual implementation).
        num_answers = list(answers.values())[0].shape[1]  # This supposes a fixed number of answers, by grabbing
        #   any of the dict values available.

        # TODO: how to obtain all layers last hidden layer for more than 1 layer? Neither seq2vec nor seq2seq give it.

        if self.text_video_mode == 'video-text':
            encoded_video = self._encode_video(video_features, frame_count, batch_size)
            encoded_video = encoded_video.reshape(self.num_layers, *encoded_video.size())  # FIXME: shouldn't it
            #   *expand* num_layers instead of reshape?
            encoded_modalities = self._encode_text(question, answers, num_answers, batch_size,
                                                   hidden_state=encoded_video)
        elif self.text_video_mode == 'text-video':
            encoded_text = self._encode_text(question, answers, num_answers, batch_size)
            encoded_text = encoded_text.reshape(self.num_layers, *encoded_text.size())  # FIXME: shouldn't it
            #   *expand* num_layers instead of reshape?

            video_features = video_features.reshape(batch_size, 1, *video_features.size()[1:]) \
                .expand(-1, num_answers, -1, -1)

            encoded_modalities = self._encode_video(video_features, frame_count, batch_size, hidden_state=encoded_text,
                                                    time_expand_size=num_answers)
        elif self.text_video_mode == 'parallel':
            encoded_video = self._encode_video(video_features, frame_count, batch_size)
            encoded_video = encoded_video.reshape(self.num_layers, *encoded_video.size()) \
                .expand(-1, num_answers, -1)

            encoded_text = self._encode_text(question, answers, num_answers, batch_size)
            # noinspection PyUnresolvedReferences
            encoded_modalities = self.parallel_feedforward(torch.cat((encoded_video, encoded_text), 2))
        elif self.text_video_mode == 'text':
            encoded_modalities = self._encode_text(question, answers, num_answers, batch_size)
        else:
            raise ValueError(f"'text_video_mode' should be one of {self.text_video_mode_options}")

        scores = self.classifier_feedforward(encoded_modalities).squeeze(2)

        output_dict = {'scores': scores}

        if label is not None:
            output_dict['loss'] = self.loss(scores, label)
            for metric in self.metrics.values():
                metric(scores, label)

        return output_dict

    def _encode_video(self, video_features: torch.Tensor, frame_count: torch.Tensor, batch_size: int,
                      hidden_state: torch.Tensor = None, time_expand_size: int = 1) -> torch.Tensor:
        video_features_mask = util.get_mask_from_sequence_lengths(frame_count, int(max(frame_count).item()))
        if time_expand_size > 1:
            video_features_mask = video_features_mask.reshape(batch_size, 1, -1).expand(-1, time_expand_size, -1)
        return self.video_encoder(video_features, video_features_mask, hidden_state=hidden_state)

    def _encode_text(self, question: Dict[str, torch.LongTensor], answers: Dict[str, torch.LongTensor],
                     num_answers: int, batch_size: int, hidden_state: torch.Tensor = None) -> torch.Tensor:
        embedded_question = self.text_field_embedder(question)
        question_mask = util.get_text_field_mask(question)
        # Passing the hidden state with LSTM doesn't work. This is because allennlp seq2vec wrapper only keeps h
        # (the hidden state) but not the cell state c. A workaround should be done if we want to chain LSTMs.
        encoded_question = self.question_encoder(embedded_question, question_mask, hidden_state=hidden_state)
        encoded_question = encoded_question.reshape(self.num_layers, batch_size, 1, -1) \
            .expand(-1, -1, num_answers, -1)
        embedded_answers = self.text_field_embedder(answers)
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        return self.answers_encoder(embedded_answers, answers_mask, hidden_state=encoded_question)

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
