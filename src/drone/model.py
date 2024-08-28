"""Contains neural network definitions.

The drone's brain represented by a feedforward neural network.
"""
import numpy
from scipy.special import expit

from src.utils.config import Config
from src.utils.utils import load_checkpoint


def load_model(config: Config) -> numpy.ndarray:
    """Loads NumPy neural network."""

    # Create model.
    model = Model(config)

    # Load pre-trained model
    if config.checkpoints.load_model:
        load_checkpoint(model=model, config=config)

    return model


class Model:
    """Neural network written with Numpy.

    TODO: Use params dict to hold weights and biases lists.

    Attributes:
        mutation_prob:
        mutation_rate:
        weights:
        biases:
    """

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
        size = (hidden_features, in_features)
        self.weights = [self._init_weights(size=size, nonlinearity=nonlinearity)]
        self.biases = [numpy.zeros(shape=(hidden_features, 1))]

        # Hidden layer weights
        size = (hidden_features, hidden_features)
        for _ in range(num_hidden_layers):
            self.weights += [self._init_weights(size=size, nonlinearity=nonlinearity)]
            self.biases += [numpy.zeros(shape=(hidden_features, 1))]

        # Output layer weights
        size = (out_features, hidden_features)
        self.weights += [self._init_weights(size=size, nonlinearity="sigmoid")]
        self.biases += [numpy.zeros(shape=(out_features, 1))]

        if nonlinearity == "tanh":
            self._nonlinearity = numpy.tanh
        elif nonlinearity == "sigmoid":
            self._nonlinearity = expit
        elif nonlinearity == "relu":
            self._nonlinearity = self._relu
        else:
            raise NotImplementedError(f"Activation function '{nonlinearity}' not implemented.")

    @staticmethod
    def _init_weights(size: tuple[int, int], nonlinearity: str) -> None:
        """Initializes model weights.

        Xavier normal initialization for feedforward neural networks described in
        'Understanding the difficulty of training deep feedforward neural networks'
        by Glorot and Bengio (2010).

            std = gain * (2 / (fan_in + fan_out)) ** 0.5
        """
        if nonlinearity == "tanh":
            gain = 5.0 / 3.0
        elif nonlinearity == "sigmoid":
            gain = 1.0
        elif nonlinearity == "relu":
            gain = 2.0**0.5
        else:
            raise NotImplementedError(f"Initialization for '{nonlinearity}' not implemented.")
        std = gain * (2.0 / sum(size)) ** 0.5
        parameters = numpy.random.normal(loc=0.0, scale=std, size=size)
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
        """Forwards observation data through network."""
        out = numpy.array(data)
        weights, biases = self.weights, self.biases
        for weight, bias in zip(weights[:-1], biases[:-1]):
            out = self._nonlinearity(numpy.matmul(out, weight.T) + bias.T)
        out = expit(numpy.matmul(out, weights[-1].T) + biases[-1].T)[0, :]
        return out
    