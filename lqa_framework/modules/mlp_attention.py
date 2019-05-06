from allennlp.modules import Attention
from overrides import overrides
import torch


@Attention.register('mlp')
class MlpAttention(Attention):
    def __init__(self, matrix_size: int, vector_size: int, hidden_size: int = 512) -> None:
        super().__init__(normalize=True)
        self.fc_vector = torch.nn.Linear(vector_size, hidden_size)
        self.fc_matrix = torch.nn.Linear(matrix_size, hidden_size)

        self.fc_output = torch.nn.Linear(hidden_size, 1)

    @overrides
    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        :param vector: shape ``(N, vector_size)``
        :param matrix: shape ``(N, *shape, matrix_size)``
        :return:       shape ``(N, *shape)``
        """
        while vector.dim() < matrix.dim():
            vector = vector.unsqueeze(1)

        hidden_layer = torch.tanh(self.fc_matrix(matrix) + self.fc_vector(vector))
        return self.fc_output(hidden_layer)
