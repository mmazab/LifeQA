from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
import torch
import torch.nn


class PytorchSeq2MatrixWrapper(Seq2VecEncoder):
    def __init__(self, module: torch.nn.modules.RNNBase) -> None:
        # Seq2VecEncoders cannot be stateful.
        super().__init__(stateful=False)
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        try:
            is_bidirectional = self._module.bidirectional
        except AttributeError:
            is_bidirectional = False
        return self._module.hidden_size * (2 if is_bidirectional else 1)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.Tensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:

        if mask is None:
            # If a mask isn't passed, there is no padding in the batch of instances, so we can just
            # return the last sequence output as the state.  This doesn't work in the case of
            # variable length sequences, as the last state for each element of the batch won't be
            # at the end of the max sequence length, so we have to use the state of the RNN below.
            return self._module(inputs, hidden_state)[0][:, -1, :]

        batch_size = mask.size(0)

        _, state, restoration_indices, = \
            self.sort_and_run_forward(self._module, inputs, mask, hidden_state)

        num_layers_times_directions, num_valid, encoding_dim = state.size()
        # Add back invalid rows.
        if num_valid < batch_size:
            # batch size is the second dimension here, because pytorch
            # returns RNN state as a tensor of shape (num_layers * num_directions,
            # batch_size, hidden_size)
            zeros = state.new_zeros(num_layers_times_directions,
                                    batch_size - num_valid,
                                    encoding_dim)
            state = torch.cat([state, zeros], 1)

        # Restore the original indices and return the final state of the
        # top layer. Pytorch's recurrent layers return state in the form
        # (num_layers * num_directions, batch_size, hidden_size) regardless
        # of the 'batch_first' flag, so we transpose, extract the relevant
        # layer state (both forward and backward if using bidirectional layers)
        # and return them as a single (batch_size, self.get_output_dim()) tensor.

        # now of shape: (batch_size, num_layers * num_directions, hidden_size).
        unsorted_state = state.transpose(0, 1).index_select(0, restoration_indices)

        # Extract the last hidden vector, including both forward and backward states
        # if the cell is bidirectional. Then reshape by concatenation (in the case
        # we have bidirectional states) or just squash the 1st dimension in the non-
        # bidirectional case. Return tensor has shape (batch_size, hidden_size * num_directions).
        try:
            last_state_index = 2 if self._module.bidirectional else 1
        except AttributeError:
            last_state_index = 1
        last_layer_state = unsorted_state[:, -last_state_index:, :]
        return last_layer_state.contiguous().view([-1, self.get_output_dim()])
