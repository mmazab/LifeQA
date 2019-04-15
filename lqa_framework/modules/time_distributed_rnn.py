"""
A wrapper that unrolls the second (time) dimension of a tensor
into the first (batch) dimension, applies some other ``RNNBase``,
and then rolls the time dimension back up. Based on pytorch.modules.time_distributed.
"""
from typing import Tuple, Union

from overrides import overrides
import torch

from .time_distributed import TimeDistributed


class TimeDistributedRNN(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``RNNBase`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributedRNN`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.

    Unlike ``TimeDistributed``, it takes into account kwargs (such as hidden_state) and also that
    hidden_state expects the batch_size in the second place (the first is num_layers). hidden_state
    should be formatted like ``(num_layers * num_directions, batch_size, time_steps, [rest])``.

    Note that while the above gives shapes with ``batch_size`` first, this ``RNNBase`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
    def __init__(self, module):
        super().__init__()
        self._module = module
        self._time_distributed = TimeDistributed(module, reshape_output=False)

    # pylint: disable=arguments-differ
    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor,
                hidden_state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        if hidden_state is not None:
            if isinstance(hidden_state, tuple):  # LSTMs.
                hidden_state_size = hidden_state[0].size()
                hidden_state = hidden_state[0].reshape(hidden_state_size[0], -1, *hidden_state_size[3:]), \
                    hidden_state[1].reshape(hidden_state_size[0], -1, *hidden_state_size[3:])
            else:
                hidden_state_size = hidden_state.size()
                hidden_state = hidden_state.reshape(hidden_state_size[0], -1, *hidden_state_size[3:])

        output = self._time_distributed(inputs, mask, hidden_state=hidden_state, pass_through=['hidden_state'])

        if len(output.size()) == 3:
            output = output.reshape(output.size()[0], *inputs.size()[:2], *output.size()[2:])
        else:
            # Now get the output back into the right shape.
            # (batch_size, time_steps, **output_size)
            output = output.reshape(inputs.size()[:2] + output.size()[1:])

        return output
