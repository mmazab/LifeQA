from typing import List, Optional

from overrides import overrides
import torch

from .time_distributed import TimeDistributed


class TimeDistributedSeq2VecRNNChain(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self._module = module
        self._time_distributed = TimeDistributed(module, reshape_output=False)

    # pylint: disable=arguments-differ
    @overrides
    def forward(self, *inputs, masks: Optional[List[Optional[torch.Tensor]]] = None,
                hidden_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hidden_state is not None:
            if isinstance(hidden_state, tuple):  # LSTMs.
                hidden_state_size = hidden_state[0].size()
                hidden_state = hidden_state[0].reshape(hidden_state_size[0], -1, *hidden_state_size[3:]), \
                    hidden_state[1].reshape(hidden_state_size[0], -1, *hidden_state_size[3:])
            else:
                hidden_state_size = hidden_state.size()
                hidden_state = hidden_state.reshape(hidden_state_size[0], -1, *hidden_state_size[3:])

        if masks:
            masks = [mask.reshape(-1, *mask.size()[2:]) for mask in masks]

        output = self._time_distributed(*inputs, masks=masks, hidden_state=hidden_state, pass_through=['hidden_state'])

        if len(output.size()) == 3:
            output = output.reshape(output.size()[0], *inputs[0].size()[:2], *output.size()[2:])
        else:
            # Now get the output back into the right shape.
            # (batch_size, time_steps, **output_size)
            output = output.reshape(inputs[0].size()[:2] + output.size()[1:])

        return output
