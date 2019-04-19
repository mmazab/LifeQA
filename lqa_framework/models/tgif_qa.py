from typing import Dict, List, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F

from ..modules.pytorch_seq2vec_wrapper_chain import PytorchSeq2VecWrapperChain


@Model.register('tgif_qa')
class TgifQaClassifier(Model):
    LOSS_OPTIONS = ['hinge', 'cross-entropy']

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder,
                 video_encoder: Optional[Seq2VecEncoder],
                 text_encoder: Seq2VecEncoder, classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, text_video_mode: str = 'video-text',
                 loss: str = 'hinge', spatial_attention: Optional[Attention] = None,
                 temporal_attention: Optional[Attention] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.answer_scorer = TimeDistributed(_TgifQaAnswerScorer(video_encoder=video_encoder,
                                                                 text_encoder=text_encoder,
                                                                 spatial_attention=spatial_attention,
                                                                 temporal_attention=temporal_attention,
                                                                 classifier_feedforward=classifier_feedforward,
                                                                 text_video_mode=text_video_mode))
        self.metrics = {'accuracy': CategoricalAccuracy()}

        if loss == 'hinge':
            self.loss = torch.nn.MultiMarginLoss()
        elif loss == 'cross-entropy':
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"'loss' should be one of {self.LOSS_OPTIONS}")

        initializer(self)

    @overrides
    def forward(self, question_and_answers: Dict[str, torch.LongTensor], video_features: Optional[torch.Tensor] = None,
                frame_count: Optional[torch.Tensor] = None,
                label: Optional[torch.LongTensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        question_and_answers : Dict[str, Variable], required	The output of ``TextField.as_array()``.
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
        # This supposes a fixed number of answers, by grabbing any of the dict values available.
        num_answers = list(question_and_answers.values())[0].shape[1]

        if video_features:
            video_features = self._expand_to_num_answers(video_features, num_answers)
            video_features_mask = self._expand_to_num_answers(
                util.get_mask_from_sequence_lengths(frame_count, video_features.shape[2]), num_answers)
        else:
            video_features_mask = None

        embedded_question_and_answers = self.text_field_embedder(question_and_answers)
        question_and_answers_mask = util.get_text_field_mask(question_and_answers, num_wrapping_dims=1)

        scores = self.answer_scorer(video_features, video_features_mask,
                                    embedded_question_and_answers, question_and_answers_mask)

        output_dict = {'scores': scores}

        if label is not None:
            output_dict['loss'] = self.loss(scores, label)
            for metric in self.metrics.values():
                metric(scores, label)

        return output_dict

    @staticmethod
    def _expand_to_num_answers(tensor: torch.Tensor, num_answers: int) -> torch.Tensor:
        return tensor.unsqueeze(1).expand(-1, num_answers, *[-1] * len(tensor.shape[1:]))

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


class _TgifQaAnswerScorer(torch.nn.Module):
    """Scores a single question-answer pair."""

    TEXT_VIDEO_MODE_OPTIONS = ['video-text', 'text-video', 'parallel', 'text']

    def __init__(self, video_encoder, text_encoder, spatial_attention, temporal_attention, classifier_feedforward,
                 text_video_mode):
        super().__init__()

        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.spatial_attention = spatial_attention
        self.temporal_attention = temporal_attention
        self.classifier_feedforward = classifier_feedforward
        self.text_video_mode = text_video_mode

        if text_video_mode == 'video-text':
            self.encoder = PytorchSeq2VecWrapperChain([video_encoder, text_encoder])
        elif text_video_mode == 'text-video':
            self.encoder = PytorchSeq2VecWrapperChain([text_encoder, video_encoder])
        elif text_video_mode == 'parallel':
            self.encoder = ParallelModalities(text_encoder.get_output_dim(), video_encoder, text_encoder)
        elif text_video_mode == 'text':
            # Use PytorchSeq2VecWrapperChain for consistency when time-distributing.
            self.encoder = PytorchSeq2VecWrapperChain([text_encoder])
        else:
            raise ValueError(f"'text_video_mode' should be one of {self.TEXT_VIDEO_MODE_OPTIONS}")

        if self.temporal_attention:
            # FIXME: first param?
            self.fc_temporal_attention = torch.nn.Linear(video_encoder.get_output_dim(), text_encoder.get_output_dim())

    @overrides
    def forward(self, video_features: Optional[torch.Tensor], video_features_mask: Optional[torch.Tensor],
                embedded_question_and_answers: torch.Tensor, question_and_answers_mask: torch.Tensor) -> torch.Tensor:
        if self.spatial_attention:
            encoded_question_and_answers = self.text_encoder(embedded_question_and_answers,
                                                             question_and_answers_mask)[0]

            video_shape = video_features.shape
            # In this case, video_features has shape (N, F, C, H, W)
            video_features = video_features \
                .reshape(*video_shape[:-2], video_shape[-2] * video_shape[-1]) \
                .transpose(-2, -1)

            video_features_mask_same_shape = video_features_mask.unsqueeze(-1).unsqueeze(-1)
            alpha = self.spatial_attention(encoded_question_and_answers, video_features, video_features_mask_same_shape)

            video_features = torch.sum(video_features * alpha, dim=2)

        if self.text_video_mode in ['video-text', 'parallel']:
            encoding_args = [video_features, embedded_question_and_answers]
            encoding_kwargs = {'masks': [video_features_mask, question_and_answers_mask]}
        elif self.text_video_mode == 'text-video':
            encoding_args = [embedded_question_and_answers, video_features]
            encoding_kwargs = {'masks': [question_and_answers_mask, video_features_mask]}
        elif self.text_video_mode == 'text':
            encoding_args = [embedded_question_and_answers]
            encoding_kwargs = {'masks': question_and_answers_mask}
        else:
            raise ValueError(f"'text_video_mode' should be one of {self.TEXT_VIDEO_MODE_OPTIONS}")

        encoded_modalities = self.encoder(*encoding_args, **encoding_kwargs)

        if self.temporal_attention:
            alpha = self.temporal_attention(encoded_modalities, self.video_encoder.last_layer_output)
            a = torch.tanh(self.fc_temporal_attention(self.video_encoder.last_layer_output * alpha))
            encoded_modalities = torch.tanh(a + encoded_modalities)

        return self.classifier_feedforward(encoded_modalities).squeeze(1)


@Attention.register('mlp')
class MlpAttention(Attention):
    def __init__(self, matrix_size: int, vector_size: int, hidden_size: int = 512) -> None:
        super().__init__(normalize=True)
        self.fc_vector = torch.nn.Linear(vector_size, hidden_size)
        self.fc_matrix = torch.nn.Linear(matrix_size, hidden_size)

        self.fc_output = torch.nn.Linear(hidden_size, 1)

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        :param vector: shape ``(N, vector_size)``
        :param matrix: shape ``(N, A, B, matrix_size)``
        :return:       shape ``(N, A, B)``
        """
        vector = vector.unsqueeze(1).unsqueeze(1)
        hidden_layer = torch.tanh(self.fc_matrix(matrix) + self.fc_vector(vector))
        return self.fc_output(hidden_layer)


class ParallelModalities(torch.nn.Module):
    def __init__(self, encoded_size, video_encoder, text_encoder):
        super().__init__()
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.parallel_feedforward = FeedForward(input_dim=encoded_size * 2, num_layers=1,
                                                hidden_dims=[encoded_size], activations=[lambda x: x])

    @overrides
    def forward(self, video_features, embedded_question_and_answers, masks: List[torch.Tensor]):
        assert len(masks) == 2
        encoded_video = self.video_encoder(video_features, masks[0])[0]
        encoded_text = self.text_encoder(embedded_question_and_answers, masks[1])[0]
        return self.parallel_feedforward(torch.cat((encoded_video, encoded_text), 2)).unsqueeze(0)
