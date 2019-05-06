from typing import Dict, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, FeedForward, similarity_functions, Seq2VecEncoder
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from overrides import overrides

from .lqa import LqaClassifier


@Model.register("bidaf_lqa")
class BidirectionalAttentionFlow(LqaClassifier):
    """
    This class was copied from allennlp.models.reading_comprehension.bidaf.BidirectionalAttentionFlow, and the following
    modifications were made:

    * The output is changed so it gives a score to every answer. For this, a Seq2Vec is used instead of a Seq2Seq
        (note that the original BiDAF needs a seq2seq to give scores to every possible start of the span).
    * It subclasses LqaClassifier, so the metrics are handled by it.
    * The docstrings are removed.
    * The code sections associated with spans are removed.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 phrase_layer: Seq2SeqEncoder,
                 modeling_layer: Seq2VecEncoder,
                 question_encoder: Seq2SeqEncoder,
                 answers_encoder: Seq2VecEncoder,
                 captions_encoder: Seq2SeqEncoder,
                 classifier_feedforward: FeedForward,
                 classifier_feedforward_answers: FeedForward,
                 num_highway_layers: int,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))

        self.classifier_feedforward = classifier_feedforward
        self.classifier_feedforward_answers = classifier_feedforward_answers

        self._phrase_layer = phrase_layer
        self._matrix_attention = LegacyMatrixAttention(similarity_functions.dot_product.DotProductSimilarity())
        self._modeling_layer = modeling_layer

        encoding_dim = phrase_layer.get_output_dim()
        
        self.answers_encoder = TimeDistributed(answers_encoder)
        self.captions_encoder = TimeDistributed(captions_encoder)
        self.question_encoder = question_encoder

        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(modeling_layer.get_input_dim(), 4 * encoding_dim,
                               "modeling layer input dim", "4 * encoding dim")
        check_dimensions_match(text_field_embedder.get_output_dim(), phrase_layer.get_input_dim(),
                               "text field embedder output dim", "phrase layer input dim")

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._mask_lstms = mask_lstms
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self, question: Dict[str, torch.LongTensor], answers: Dict[str, torch.LongTensor],
                captions: Dict[str, torch.LongTensor], label: Optional[torch.LongTensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        captions['tokens'] = captions['tokens'].squeeze()
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(captions))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(captions).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))

        encoded_passage = self._dropout(self._phrase_layer(embedded_passage.squeeze(), passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.masked_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))

        embedded_answers = self._highway_layer(self._text_field_embedder(answers))
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        encoded_answers = self.answers_encoder(embedded_answers, answers_mask)

        fuse_cq = self.classifier_feedforward(modeled_passage)
        logits = torch.bmm(encoded_answers, fuse_cq.unsqueeze(2)).squeeze(2)

        output_dict = {'logits': logits}
        if label is not None:
            output_dict['loss'] = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)

        return output_dict
