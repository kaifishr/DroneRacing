"""Evolutionary inspired black-box optimization algorithms.
"""
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
        self.parameters = [agent.model.state_dict for agent in self.agents]
        self.learning_rate = learning_rate
        self.sigma = sigma

    def step(self) -> None:
        """Performs single optimization step."""

        # Get rewards.
        rewards = numpy.array([agent.score for agent in self.agents])

        # Normalize rewards.
        rewards -= rewards.mean()
        rewards /= rewards.std() + 1e-5

        # Compute gradients.
        # TODO

        # Scatter new gradients.
        # TODO
