from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from overrides import overrides
import torch
import torch.nn.functional as F

from .lqa import LqaClassifier


@Model.register('text_only')
class TextOnlyClassifier(LqaClassifier):
    """Text-only ``Model``.

    The basic model structure: we embed the question using LSTM/CNN, same for the answers and captions using a
    separate Seq2VecEncoders, getting a single vector representing the content of each. We'll then fuse the questions
    and captions vectors and the pass the result through a layer, the output of which should be closest to the correct
    answer.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 question_encoder: Seq2VecEncoder,
                 answers_encoder: Seq2VecEncoder,
                 captions_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size('labels')

        self.question_encoder = question_encoder
        self.answers_encoder = TimeDistributed(answers_encoder)
        self.captions_encoder = TimeDistributed(captions_encoder)
        self.classifier_feedforward = classifier_feedforward
        # self.classifier_feedforward = TimeDistributed(classifier_feedforward)

        self._encoding_dim = captions_encoder.get_output_dim()
        self.ques_cap_att = LinearMatrixAttention(self._encoding_dim, self._encoding_dim, 'x,y,x*y')

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self, question: Dict[str, torch.LongTensor], answers: Dict[str, torch.LongTensor],
                captions: Dict[str, torch.LongTensor], label: Optional[torch.LongTensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        embedded_question = self.text_field_embedder(question)
        question_mask = util.get_text_field_mask(question)
        encoded_question = self.question_encoder(embedded_question, question_mask)

        embedded_answers = self.text_field_embedder(answers)
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        encoded_answers = self.answers_encoder(embedded_answers, answers_mask)

        embedded_captions = self.text_field_embedder(captions)
        captions_mask = util.get_text_field_mask(captions, num_wrapping_dims=1)
        encoded_captions = self.captions_encoder(embedded_captions, captions_mask)
        encoded_captions = encoded_captions.squeeze(1)

        # TODO: should add attention between question and captions
        # TODO: should add attention between the output of the (questions and captions) and different answers

        fuse_cq = self.classifier_feedforward(torch.cat([encoded_captions, encoded_question], dim=-1))
        scores = torch.bmm(encoded_answers, fuse_cq.unsqueeze(2)).squeeze(2)

        output_dict = {'scores': scores}

        if label is not None:
            output_dict['loss'] = self.loss(scores, label)
            for metric in self.metrics.values():
                metric(scores, label)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = super().decode(output_dict)
        output_dict['class_probabilities'] = F.softmax(output_dict['scores'], dim=1)
        return output_dict
