from typing import Type

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules import Seq2VecEncoder
import torch

from .mlp_attention import MlpAttention
from .pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from .pytorch_seq2vec_wrapper_chain import PytorchSeq2VecWrapperChain
from .time_distributed import TimeDistributed
from .time_distributed_seq2vec_rnn import TimeDistributedSeq2VecRNN
from .time_distributed_seq2vec_rnn_chain import TimeDistributedSeq2VecRNNChain


class _Seq2VecWrapper:
    """Copy of ``allennlp.modules.seq2vec_encoders._Seq2VecWrapper`` to add support for the patched
    ``PytorchSeq2VecWrapper``.
    """
    PYTORCH_MODELS = [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]

    def __init__(self, module_class: Type[torch.nn.modules.RNNBase]) -> None:
        self._module_class = module_class

    def __call__(self, **kwargs) -> PytorchSeq2VecWrapper:
        return self.from_params(Params(kwargs))

    # Logic requires custom from_params
    def from_params(self, params: Params) -> PytorchSeq2VecWrapper:
        if not params.pop('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        if self._module_class in self.PYTORCH_MODELS:
            params['batch_first'] = True
        return_all_layers = params.pop('return_all_layers', False)
        return_all_hidden_states = params.pop('return_all_hidden_states', False)
        module = self._module_class(**params.as_dict())
        return PytorchSeq2VecWrapper(module, return_all_layers=return_all_layers,
                                     return_all_hidden_states=return_all_hidden_states)


Seq2VecEncoder.register('gru_patched')(_Seq2VecWrapper(torch.nn.GRU))
Seq2VecEncoder.register('lstm_patched')(_Seq2VecWrapper(torch.nn.LSTM))
Seq2VecEncoder.register('rnn_patched')(_Seq2VecWrapper(torch.nn.RNN))
