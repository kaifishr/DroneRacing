"""Evolutionary inspired black-box optimization algorithms.
"""
import copy

import numpy

from src.drone import Agent


class Optimizer:
    def __init__(self) -> None:
        """Initializes Optimizer base class."""


class GeneticOptimizer(Optimizer):
    def __init__() -> None:
        """ """
        super().__init__()


class EvolutionStrategy(Optimizer):
    def __init__(self, agents: list[Agent], learning_rate: float, sigma: float) -> None:
        """Initializes Evolution Strategies optimizer.

        Args:
            agents: Agents.
            learning_rate: Gradient descent learning rate.
            sigma: Noise level.
        """
        super().__init__()
        assert len(agents) > 1
        self.agents = agents

        self.parameters = [
            (agent.model.weights, agent.model.biases) 
            for agent in self.agents
        ]

        # self.shapes = [
        #     (w.shape, b.shape) 
        #     for w, b in 
        #     zip(self.agents[0].model.weights, self.agents[0].model.biases)
        # ]
        self.params = [
            (numpy.random.normal(0, 0.1, w.shape), numpy.random.normal(0, 0.1, b.shape))
            for w, b in 
            zip(self.agents[0].model.weights, self.agents[0].model.biases)
        ]
        # self.params = [
        #     *zip(copy.deepcopy(self.agents[0].model.weights), copy.deepcopy(self.agents[0].model.biases))
        # ]

        self.gradients = [
            (numpy.zeros_like(w), numpy.zeros_like(b)) 
            for w, b in 
            zip(self.agents[0].model.weights, self.agents[0].model.biases)
        ]

        self.learning_rate = learning_rate
        self.sigma = sigma

    def mutate_parameters(self) -> None:
        """Mutates the network's parameters."""
        for weights, biases in self.parameters:
            for weight, bias in zip(weights, biases):
                noise = numpy.random.normal(loc=0.0, scale=self.sigma, size=weight.shape)
                numpy.add(weight, noise, out=weight)
                noise = numpy.random.normal(loc=0.0, scale=self.sigma, size=bias.shape)
                numpy.add(bias, noise, out=bias)

    @staticmethod
    def softmax(x: numpy.array) -> numpy.array:
        x = numpy.exp(x - x.max())
        return x / x.sum()

    def step(self) -> None:
        """Performs single optimization step."""

        # Get rewards.
        rewards = numpy.array([agent.score for agent in self.agents])

        # Normalize rewards.
        rewards -= rewards.mean()
        rewards /= rewards.std() + 1e-5

        # Softmax
        # temp = 1.0
        # rewards = rewards / temp
        rewards = self.softmax(rewards)

        # Reset gradients.
        for (grad_weights, grad_biases) in self.gradients:
            numpy.multiply(grad_weights, 0.0, out=grad_weights)
            numpy.multiply(grad_biases, 0.0, out=grad_biases)

        # Compute gradients. (Or use first agent's parameters as buffer.)
        for agent, reward in zip(self.agents, rewards):
            for (weight, bias), (grad_weight, grad_bias) in zip(zip(*agent.model.state_dict().values()), self.gradients):
                # grad_weight += reward * weight
                # grad_bias += reward * bias
                # grad_weight[...] = grad_weight[...] + reward * weight
                # grad_bias[...] = grad_bias[...] + reward * bias
                numpy.add(grad_weight, reward * weight, out=grad_weight)
                numpy.add(grad_bias, reward * bias, out=grad_bias)

        # Perform gradient descent.
        for (weight, bias), (grad_weight, grad_bias) in zip(self.params, self.gradients):
            # weight += self.learning_rate * grad_weight 
            # bias += self.learning_rate * grad_bias 
            # weight[...] = grad_weight[...]
            # bias[...] = grad_bias[...]
            # weight = copy.deepcopy(grad_weight)
            # bias = copy.deepcopy(grad_bias)
            numpy.add(weight, self.learning_rate * grad_weight, out=weight)
            numpy.add(bias, self.learning_rate * grad_bias, out=bias)

        # Broadcast new parameters.
        for agent in self.agents:
            weights, biases = zip(*self.params)
            agent.model.weights = copy.deepcopy(weights)
            agent.model.biases = copy.deepcopy(biases)

        # Mutate parameters.
        self.mutate_parameters()