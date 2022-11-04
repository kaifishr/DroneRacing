"""Contains neural network definitions.

The neural network represents the drone's brain.

"""
import numpy
import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

from src.utils.config import Config


class NeuralNetwork:
    """Neural network written with Numpy.

    Attributes:
        mutation_prob:
        mutation_rate:
        weights:
        biases:
    """

    def __init__(self, config: Config) -> None:
        """Initializes NeuralNetwork."""

        self.mutation_prob = config.optimizer.mutation_probability
        self.mutation_rate = config.optimizer.mutation_rate

        config = config.env.drone.neural_network

        in_features = config.num_dim_in
        out_features = config.num_dim_out
        hidden_features = config.num_dim_hidden
        num_hidden_layers = config.num_hidden_layers

        # Input layer weights
        self.weights = [self._init_weights(size=(hidden_features, in_features))]
        self.biases = [np.zeros(shape=(hidden_features, 1))]

        # Hidden layer weights
        for _ in range(num_hidden_layers):
            self.weights += [
                self._init_weights(size=(hidden_features, hidden_features))
            ]
            self.biases += [np.zeros(shape=(hidden_features, 1))]

        # Output layer weights
        self.weights += [self._init_weights(size=(out_features, hidden_features))]
        self.biases += [np.zeros(shape=(out_features, 1))]

    def _init_weights(self, size: tuple[int, int]) -> None:
        """Initializes model weights."""
        return np.random.normal(loc=0.0, scale=0.5, size=size)

    def __call__(self, x: numpy.ndarray):
        return self.forward(x)

    def mutate_weights(self) -> None:
        """Mutates the network's weights."""
        for weight, bias in zip(self.weights, self.biases):
            mask = numpy.random.random(size=weight.shape) < self.mutation_prob
            mutation = self.mutation_rate * numpy.random.normal(size=weight.shape)
            weight[:] = weight[:] + mask * mutation
            # weight += mask * mutation

            mask = numpy.random.random(size=bias.shape) < self.mutation_prob
            mutation = self.mutation_rate * numpy.random.normal(size=bias.shape)
            bias[:] = bias[:] + mask * mutation
            # bias += mask * mutation

    @staticmethod
    def _sigmoid(x: numpy.ndarray) -> numpy.ndarray:
        # return 1.0 / (1.0 + np.exp(-x))
        return expit(x)

    def eval(self):
        pass

    def forward(self, x: numpy.ndarray):
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(np.matmul(x, weight.T) + bias.T)
        x = self._sigmoid(np.matmul(x, self.weights[-1].T) + self.biases[-1].T)[0, :]
        return x


class NeuralNetwork_(nn.Module):
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
        num_hidden_layers = config.num_hidden_layers

        layers = [
            nn.Flatten(start_dim=0),
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.Tanh(),
        ]

        for _ in range(num_hidden_layers):
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
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.5)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @torch.no_grad()  # TODO: add to NetLab as regularization tool if not used here.
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
