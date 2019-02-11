# Implement simple longest answer baseline using basic python techniques because rip Anaconda
import random
import json



def choose_longest(answers):
    # This function will return the location of maximum length answer in answers
    max_num = len(max(answers, key=lambda x: len(x)))

    max_choices = []
    # Make a list of all the max length answers
    for index, ans in enumerate(answers):
        if len(ans) == max_num:
            max_choices.append(index)

    # Return a random max length answer
    return random.choice(max_choices)


filename = 'lqa_dev.json' 

with open('data/' + filename) as f:
    accuracy = 0
    total = 0
    data = json.load(f)
    for video, info in data.items():
        questions = info['questions']
        for question in questions:
            answers = question["answers"]
            choice = choose_longest(answers)

            if choice == question['correct_index']: 
                accuracy += 1
            # Always add to the total
            total += 1

    print("Accuracy of longest_answer_baseline on {} = {}".format(filename, float(accuracy)/ float(total)))







# from typing import Dict, Optional

# from allennlp.data import Vocabulary
# from allennlp.models.model import Model
# from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
# from allennlp.modules.matrix_attention.linear_matrix_attention import LinearMatrixAttention
# from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
# from allennlp.training.metrics import CategoricalAccuracy
# import numpy
# from overrides import overrides
# import torch
# import torch.nn.functional as F


# @Model.register('longest_answer_baseline')
# class LongestAnswerBaseline(Model):
#     """This ``Model`` performs question answering. We assume we're given the video/question/set of answers and we
#     predict the correct answer.

#     The basic model structure: we take the answers and return the longest one as correct."""

#     def __init__(self, vocab: Vocabulary,
#                  text_field_embedder: TextFieldEmbedder,
#                  regularizer: Optional[RegularizerApplicator] = None) -> None:
#         super().__init__(vocab, regularizer)

#         self.text_field_embedder = text_field_embedder
#         self.num_classes = self.vocab.get_vocab_size('labels')

#         self.metrics = {'accuracy': CategoricalAccuracy()}
#         initializer(self)

#     @overrides
#     def forward(self, question: Dict[str, torch.LongTensor], answers: Dict[str, torch.LongTensor],
#                 captions: Dict[str, torch.LongTensor],
#                 label: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

#         print(answers)
#         exit()


#     @overrides
#     def get_metrics(self, reset: Optional[bool] = False) -> Dict[str, float]:
#         return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}