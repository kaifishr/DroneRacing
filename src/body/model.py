"""Contains neural network definitions.

The neural network represents the drone's brain.

"""
import torch
import torch.nn as nn

from src.config import Config


class NeuralNetwork(nn.Module):
    """Network class.

    Simple fully-connected neural network.

    Attributes:
        mutation_prob:
        mutation_rate:
        net:
    """

    def __init__(self, config: Config) -> None:
        """Initializes NeuralNetwork class."""
        super().__init__()

        self.mutation_prob = config.optimizer.mutation_probability
        self.mutation_rate = config.optimizer.mutation_rate

        config = config.env.drone.neural_network

        in_features = config.num_dim_in
        out_features = config.num_dim_out
        hidden_features = config.num_dim_hidden
        num_hidden = config.num_hidden

        layers = [
            nn.Flatten(start_dim=0),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.Tanh(),
        ]

        for _ in range(num_hidden):
            layers += [
                nn.Linear(in_features=hidden_features, out_features=hidden_features),
                nn.Tanh(),
            ]

        layers += [
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Sigmoid(),
        ]

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.4)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @torch.no_grad()  # add to NetLab as regularization tool if not used here.
    def _mutate_weights(self, module: nn.Module) -> None:
        """Mutates weights."""
        if isinstance(module, nn.Linear):
            mask = torch.rand_like(module.weight) < self.mutation_prob
            mutation = self.mutation_rate * torch.randn_like(module.weight)
            module.weight.add_(mask * mutation)
            if module.bias is not None:
                mask = torch.rand_like(module.bias) < self.mutation_prob
                mutation = self.mutation_rate * torch.randn_like(module.bias)
                module.bias.add_(mask * mutation)

    def mutate_weights(self) -> None:
        """Mutates the network's weights."""
        self.apply(self._mutate_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
