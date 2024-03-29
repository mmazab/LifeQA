from typing import List, Optional

from allennlp.modules import Seq2VecEncoder
from overrides import overrides
import torch

from .pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper


class PytorchSeq2VecWrapperChain(Seq2VecEncoder):
    def __init__(self, encoders: List[PytorchSeq2VecWrapper]) -> None:
        # Seq2VecEncoders cannot be stateful.
        super().__init__(stateful=False)
        if not encoders:
            raise ValueError("`encoders` cannot be empty")
        self.encoders = encoders

    @overrides
    def get_input_dim(self) -> int:
        return self.encoders[0].get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.encoders[-1].get_output_dim()

    @overrides
    def forward(self, *inputs, masks: Optional[List[Optional[torch.Tensor]]] = None,
                hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        n_iterations = len(self.encoders)

        if masks is None:
            masks = [None] * n_iterations

        for i, (encoder, input_, mask) in enumerate(zip(self.encoders, inputs, masks)):
            hidden_state = encoder(input_, mask, hidden_state)

            if i < n_iterations - 1:
                # noinspection PyProtectedMember
                hidden_size = encoder._module.hidden_size

                # It's better to obtain these numbers with respect to the last position,
                # because there may be an extra dimension at the beginning based on the value of
                # `return_all_hidden_states` for the encoder.
                batch_size = hidden_state.size()[-2]
                num_layers_times_directions_times_hidden_size = hidden_state.size()[-1]

                num_layers_times_directions = num_layers_times_directions_times_hidden_size // hidden_size

                hidden_state = hidden_state.reshape(-1, batch_size, num_layers_times_directions, hidden_size) \
                    .transpose(1, 2) \
                    .squeeze(0)  # It may be 1, and in that case the model expects only one state in hidden_state.
                if len(hidden_state.size()) == 4:
                    hidden_state = (hidden_state[0], hidden_state[1])

        if len(hidden_state.size()) == 3:
            return hidden_state[0]
        else:
            return hidden_state
