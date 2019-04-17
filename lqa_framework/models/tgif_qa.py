from typing import Dict, List, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy
import numpy as np
from overrides import overrides
import torch
import torch.nn.functional as F

from lqa_framework.modules.pytorch_seq2vec_wrapper_chain import PytorchSeq2VecWrapperChain
from lqa_framework.modules.time_distributed_seq2vec_rnn_chain import TimeDistributedSeq2VecRNNChain


@Model.register('tgif_qa')
class TgifQaClassifier(Model):
    TEXT_VIDEO_MODE_OPTIONS = ['video-text', 'text-video', 'parallel', 'text']
    LOSS_OPTIONS = ['hinge', 'cross-entropy']

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder,
                 video_encoder: Optional[Seq2VecEncoder],
                 text_encoder: Seq2VecEncoder, classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, text_video_mode: str = 'video-text',
                 loss: str = 'hinge', use_spatial_attention: bool = False,
                 use_temporal_attention: bool = False) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.video_encoder = video_encoder
        self.text_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward
        self.text_video_mode = text_video_mode
        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_attention = use_temporal_attention

        if video_encoder is None and text_video_mode != 'text':
            raise ValueError("'video_encoder' can be None only if 'text_video_mode' is set to 'text'")

        if self.use_spatial_attention:
            self.fc_question = torch.nn.Linear(self.video_encoder.get_input_dim(), 512)

        if text_video_mode == 'video-text':
            self.encoder = PytorchSeq2VecWrapperChain([self.video_encoder, self.text_encoder])
        elif text_video_mode == 'text-video':
            self.encoder = PytorchSeq2VecWrapperChain([self.text_encoder, self.video_encoder])
        elif text_video_mode == 'parallel':
            self.encoder = ParallelModalities(text_encoder.get_output_dim(), self.video_encoder, self.text_encoder)
        elif text_video_mode == 'text':
            # Use PytorchSeq2VecWrapperChain for consistency when time-distributing.
            self.encoder = PytorchSeq2VecWrapperChain([self.text_encoder])
        else:
            raise ValueError(f"'text_video_mode' should be one of {self.TEXT_VIDEO_MODE_OPTIONS}")

        self.main_model = TimeDistributedSeq2VecRNNChain(MainModel(self.encoder, self.classifier_feedforward))

        self.metrics = {'accuracy': CategoricalAccuracy()}

        if loss == 'hinge':
            self.loss = torch.nn.MultiMarginLoss()
        elif loss == 'cross-entropy':
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"'loss' should be one of {self.LOSS_OPTIONS}")

        initializer(self)

    @overrides
    def forward(self, question_and_answers: Dict[str, torch.LongTensor],
                captions: Dict[str, torch.LongTensor], video_features: Optional[torch.Tensor] = None,
                frame_count: Optional[torch.Tensor] = None,
                label: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """Does the forward pass.

        Parameters
        ----------
        question_and_answers : Dict[str, Variable], required	The output of ``TextField.as_array()``.
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
        # This supposes a fixed number of answers, by grabbing any of the dict values available.
        num_answers = list(question_and_answers.values())[0].shape[1]

        video_features = video_features.unsqueeze(1).expand(-1, num_answers, -1, -1)
        video_features_mask = util.get_mask_from_sequence_lengths(frame_count, int(max(frame_count).item())) \
            .unsqueeze(1) \
            .expand(-1, num_answers, -1)

        embedded_question_and_answers = self.text_field_embedder(question_and_answers)
        question_and_answers_mask = util.get_text_field_mask(question_and_answers, num_wrapping_dims=1)

        if self.text_video_mode in ['video-text', 'parallel']:
            args = [video_features, embedded_question_and_answers]
            kwargs = {'masks': [video_features_mask, question_and_answers_mask]}
        elif self.text_video_mode == 'text-video':
            args = [embedded_question_and_answers, video_features]
            kwargs = {'masks': [question_and_answers_mask, video_features_mask]}
        elif self.text_video_mode == 'text':
            args = [embedded_question_and_answers]
            kwargs = {'masks': question_and_answers_mask}
        else:
            raise ValueError(f"'text_video_mode' should be one of {self.TEXT_VIDEO_MODE_OPTIONS}")

        scores = self.main_model(*args, **kwargs)

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


class MainModel(torch.nn.Module):
    def __init__(self, encoder, classifier_feedforward):
        super().__init__()
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward

    @overrides
    def forward(self, *inputs, **kwargs):
        encoded_modalities = self.encoder(*inputs, **kwargs)
        return self.classifier_feedforward(encoded_modalities).squeeze(1)


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
