"""
A wrapper that unrolls the second (time) dimension of a tensor
into the first (batch) dimension, applies some other ``RNNBase``,
and then rolls the time dimension back up. Based on pytorch.modules.time_distributed.
"""

import torch


class TimeDistributedRNN(torch.nn.Module):
    """
    Given an input shaped like ``(batch_size, time_steps, [rest])`` and a ``RNNBase`` that takes
    inputs like ``(batch_size, [rest])``, ``TimeDistributedRNN`` reshapes the input to be
    ``(batch_size * time_steps, [rest])``, applies the contained ``Module``, then reshapes it back.

    Unlike ``TimeDistributed``, it takes into account kwargs (such as hidden_state) and also that
    hidden_state expects the batch_size in the second place (the first is num_layers). hidden_state
    should be formatted like ``(num_layers, batch_size, time_steps, [rest])``.

    Note that while the above gives shapes with ``batch_size`` first, this ``RNNBase`` also works if
    ``batch_size`` is second - we always just combine the first two dimensions, then split them.
    """
    def __init__(self, module):
        super().__init__()
        self._module = module

    def forward(self, *inputs, hidden_state=None):  # pylint: disable=arguments-differ
        reshaped_inputs = []
        for input_tensor in inputs:
            input_size = input_tensor.size()
            if len(input_size) <= 2:
                raise RuntimeError("No dimension to distribute: " + str(input_size))

            # Squash batch_size and time_steps into a single axis; result has shape
            # (batch_size * time_steps, input_size).
            squashed_shape = [-1] + [x for x in input_size[2:]]
            reshaped_inputs.append(input_tensor.contiguous().view(*squashed_shape))

        if hidden_state is not None:
            hidden_state_size = hidden_state.size()
            hidden_state = hidden_state.contiguous().view(hidden_state_size[0], -1, *hidden_state_size[3:])

        reshaped_outputs = self._module(*reshaped_inputs, hidden_state=hidden_state)

        # Now get the output back into the right shape.
        # (batch_size, time_steps, [hidden_size])
        new_shape = [input_size[0], input_size[1]] + [x for x in reshaped_outputs.size()[1:]]
        outputs = reshaped_outputs.contiguous().view(*new_shape)

        return outputs
