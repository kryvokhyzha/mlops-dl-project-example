import torch
from torch import nn


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        Parameters
        ----------
        input_size : int, optional
            The size of the input tensor, by default 784
        lin1_size : int, optional
            The size of the first linear layer, by default 256
        lin2_size : int, optional
            The size of the second linear layer, by default 256
        lin3_size : int, optional
            The size of the third linear layer, by default 256
        output_size : int, optional
            The size of the output tensor, by default 10
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.model(x)
