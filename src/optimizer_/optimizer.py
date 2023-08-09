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

        self.params = [
            (numpy.zeros_like(w), numpy.zeros_like(b)) 
            for w, b in 
            zip(self.agents[0].model.weights, self.agents[0].model.biases)
        ]

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
                weight[:] = weight[:] + noise
                # weight += mask * mutation

                noise = numpy.random.normal(loc=0.0, scale=self.sigma, size=bias.shape)
                bias[:] = bias[:] + noise
                # bias += mask * mutation

    def step(self) -> None:
        """Performs single optimization step."""

        # Get rewards.
        rewards = numpy.array([agent.score for agent in self.agents])

        # Normalize rewards.
        rewards -= rewards.mean()
        rewards /= rewards.std() + 1e-5

        # Compute gradients. (Or use first agent's parameters as buffer.)
        for agent, reward in zip(self.agents, rewards):
            for (weight, bias), (grad_weight, grad_bias) in zip(zip(*agent.model.state_dict().values()), self.gradients):
                grad_weight += reward * weight
                grad_bias += reward * bias

        # Perform gradient descent.
        for agent in self.agents:
            for (weight, bias), (grad_weight, grad_bias) in zip(zip(*agent.model.state_dict().values()), self.gradients):
                # weight += self.learning_rate * grad_weight 
                # bias += self.learning_rate * grad_bias 
                weight[...] = grad_weight
                bias[...] = grad_bias 

        # Reset gradients.
        # self.gradients = [
        #     (dw.fill(0.0), db.fill(0.0)) 
        #     for dw, db in self.gradients
        # ]
        for (dw, db) in self.gradients:
            dw *= 0.0
            db *= 0.0

        # Scatter new gradients.
        # TODO

        # Mutate parameters.
        self.mutate_parameters()
