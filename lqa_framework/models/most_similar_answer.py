from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.modules.similarity_functions import CosineSimilarity
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from overrides import overrides
import torch

from .simple_baseline import SimpleBaseline


@Model.register('most_similar_answer')
class MostSimilarAnswer(SimpleBaseline):
    """This ``Model`` performs question answering. We assume we're given the video/question/set of answers and we
    predict the correct answer.

    The basic model structure: we take the answers and return the one closest in cosine similarity to the question
    (by averaging the word embeddings)."""

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder,
                 question_encoder: Seq2VecEncoder, answers_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, initializer, regularizer)
        self.text_field_embedder = text_field_embedder
        self.question_encoder = question_encoder
        self.answers_encoder = TimeDistributed(answers_encoder)
        self.cosine_similarity = TimeDistributed(CosineSimilarity())

    @overrides
    def _compute_logits(self, question: Dict[str, torch.LongTensor],
                        answers: Dict[str, torch.LongTensor]) -> torch.Tensor:
        embedded_question = self.text_field_embedder(question)
        question_mask = util.get_text_field_mask(question)
        encoded_question = self.question_encoder(embedded_question, question_mask)

        embedded_answers = self.text_field_embedder(answers)
        answers_mask = util.get_text_field_mask(answers, num_wrapping_dims=1)
        encoded_answers = self.answers_encoder(embedded_answers, answers_mask)

        repeated_encoded_question = encoded_question.repeat(encoded_answers.shape[1], 1)
        return self.cosine_similarity(tensor_1=repeated_encoded_question, tensor_2=encoded_answers,
                                      pass_through=['tensor_1'])\
            .view(encoded_answers.shape[1], encoded_question.shape[0])\
            .transpose(0, 1)
