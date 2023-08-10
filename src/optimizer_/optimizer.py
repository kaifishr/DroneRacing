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

        assert len(agents) > 1, f"Number of agents must be larger than 1."

        self.agents = agents
        self.learning_rate = learning_rate
        self.sigma = sigma

        model = agents[0].model
        self.parameters = (copy.deepcopy(model.weights), copy.deepcopy(model.biases))

        self._initialize()

    def _initialize(self) -> None:
        """Initializes agents by broadcasting same parameters and adding noise."""
        self.rewards = None
        self.gradients = None
        self._register_noise()
        self._broadcast_parameters()
        self._create_noise()
        self._mutate_parameters()
        self._initialize_gradients()

    def _register_noise(self) -> None:
        """Registers noise in model of each agent."""
        for agent in self.agents:
            agent.model.eps_weights = copy.deepcopy(agent.model.weights)
            agent.model.eps_biases = copy.deepcopy(agent.model.biases)

    def _broadcast_parameters(self) -> None:
        """Broadcasts parameters to agents."""
        weights, biases = self.parameters
        for agent in self.agents:
            agent.model.weights = copy.deepcopy(weights)
            agent.model.biases = copy.deepcopy(biases)

    def _create_noise(self) -> None:
        """Creates noise for each parameter."""
        # Zero noise.
        for agent in self.agents:
            for eps_weights, eps_biases in zip(agent.model.eps_weights, agent.model.eps_biases):
                eps_weights.fill(0.0)
                eps_biases.fill(0.0)

        for agent in self.agents:
            # model = agent.model # TODO
            for eps_weights, eps_biases in zip(agent.model.eps_weights, agent.model.eps_biases):
                noise = numpy.random.normal(loc=0.0, scale=self.sigma, size=eps_weights.shape)
                numpy.add(eps_weights, noise, out=eps_weights) 
                noise = numpy.random.normal(loc=0.0, scale=self.sigma, size=eps_biases.shape)
                numpy.add(eps_biases, noise, out=eps_biases) 

    def _mutate_parameters(self) -> None:
        """Mutates models's parameters of each agent by adding Gaussian noise."""
        for agent in self.agents:
            model = agent.model
            for (weights, biases), (eps_weights, eps_biases) in zip(zip(model.weights, model.biases), zip(model.eps_weights, model.eps_biases)):
                numpy.add(weights, eps_weights, out=weights)
                numpy.add(biases, eps_biases, out=biases)

    def _initialize_gradients(self) -> None:
        """Initializes gradients."""
        weights, biases = self.parameters
        self.gradients = [
            (numpy.zeros_like(w), numpy.zeros_like(b)) 
            for w, b in zip(weights, biases)
        ]
        
    def _zero_gradients(self) -> None:
        """Sets all gradients to zero."""
        for dw, db in self.gradients:
            dw.fill(0.0)
            db.fill(0.0)

    def _compute_gradients(self) -> None:
        """Computes approxiamte gradients."""
        for agent, reward in zip(self.agents, self.rewards):
            for (grad_weights, grad_biases), (eps_weights, eps_biases) in zip(self.gradients, zip(agent.model.eps_weights, agent.model.eps_biases)):
                numpy.add(grad_weights, reward * eps_weights, out=grad_weights)
                numpy.add(grad_biases, reward * eps_biases, out=grad_biases)

    def _gradient_descent(self) -> None:
        """Performs gradient descent step."""
        for (weights, biases), (grad_weights, grad_biases) in zip(zip(*self.parameters), self.gradients):
            numpy.subtract(weights, self.learning_rate * grad_weights, out=weights)
            numpy.subtract(biases, self.learning_rate * grad_biases, out=biases)

    def _gather_rewards(self) -> None:
        """Gathers rewards from all agents."""
        # Get rewards.
        rewards = numpy.array([agent.score for agent in self.agents])

        # Normalize rewards.
        rewards -= rewards.mean()
        rewards /= rewards.std() + 1e-5

        self.rewards = rewards

        # Softmax
        # temp = 1.0
        # rewards = rewards / temp
        # rewards = self._softmax(rewards)

    @staticmethod
    def _softmax(x: numpy.array) -> numpy.array:
        x = numpy.exp(x - x.max())
        return x / x.sum()
    
    def step(self) -> None:
        """Performs single optimization step."""
        self._gather_rewards()
        self._zero_gradients()
        self._compute_gradients()
        self._gradient_descent()
        self._broadcast_parameters()
        self._create_noise()
        self._mutate_parameters()