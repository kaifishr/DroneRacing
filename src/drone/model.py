"""Contains neural network definitions.

The drone's brain represented by a feedforward neural network.
"""
import numpy
from scipy.special import expit

from src.utils.config import Config
from src.utils.utils import load_checkpoint


def load_model(config: Config) -> numpy.ndarray:
    model = Model(config)
    if config.checkpoints.load_model:
        load_checkpoint(model=model, config=config)
    return model


class Model:

    def __init__(self, config: Config) -> None:
        """Initializes NeuralNetwork."""

        self.mutation_prob = config.optimizer.evo.mutation_probability
        self.mutation_rate = config.optimizer.evo.mutation_rate

        config = config.env.drone.neural_network

        in_features = config.num_dim_in
        out_features = config.num_dim_out
        hidden_features = config.num_dim_hidden
        num_hidden_layers = config.num_hidden_layers
        nonlinearity = config.nonlinearity

        # Create layers and initializes weights.

        # Input layer weights
        size = (in_features, hidden_features)
        self.weights = [self._init_weights(size=size)]
        self.biases = [numpy.zeros(shape=(1, hidden_features))]

        # Hidden layer weights
        size = (hidden_features, hidden_features)
        for _ in range(num_hidden_layers):
            self.weights += [self._init_weights(size=size)]
            self.biases += [numpy.zeros(shape=(1, hidden_features))]

        # Output layer weights
        size = (hidden_features, out_features)
        self.weights += [self._init_weights(size=size)]
        self.biases += [numpy.zeros(shape=(1, out_features))]

        if nonlinearity == "tanh":
            self._nonlinearity = numpy.tanh
        elif nonlinearity == "sigmoid":
            self._nonlinearity = expit
        elif nonlinearity == "relu":
            self._nonlinearity = self._relu
        else:
            raise NotImplementedError(f"Activation function '{nonlinearity}' not implemented.")

    @staticmethod
    def _init_weights(size: tuple[int, int]) -> None:
        parameters = numpy.random.normal(loc=0.0, scale=1e-4, size=size)
        parameters = numpy.clip(parameters, a_min=-3.0, a_max=3.0)
        return parameters

    @staticmethod
    def _relu(array: numpy.ndarray) -> numpy.ndarray:
        return array * (array > 0)

    def state_dict(self) -> dict:
        """Returns a dictionary containing the network's weights and biases.

        Return:
            state: State holding weights and biases of network.
        """
        state = {
            "weights": self.weights,
            "biases": self.biases,
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Loads state dict holding the network's weights and biases.

        Note: Ignores parameter dimension.
        """
        self.weights = state_dict["weights"]
        self.biases = state_dict["biases"]

    def forward(self, data: list):
        out = numpy.asarray(data).reshape(1, -1)
        *weights, weights_last = self.weights
        *biases, biases_last = self.biases
        for weight, bias in zip(weights, biases):
            out = self._nonlinearity(numpy.dot(out, weight) + bias)
        out = expit(numpy.dot(out, weights_last) + biases_last)
        return out[0, :]