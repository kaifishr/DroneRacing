"""Evolutionary inspired black-box optimization algorithms.
"""
import copy
import numpy
from src.drone import Agent


class Optimizer:
    """Optimizer base class."""

    # Value added to the denominator for numerical stability.
    eps: float = 1e-5

    def __init__(self, agents: list[Agent]) -> None:
        """Initializes Optimizer base class."""

        # assert len(agents) > 1, f"Number of agents must be larger than 1."

        self.agents = agents


class GeneticOptimizer(Optimizer):
    """Genetic optimizer class."""

    def __init__(self, agents: list[Agent], mutation_probability: float, mutation_rate=float) -> None:
        """ """
        super().__init__(agents=agents)

        assert 0.0 <= mutation_probability <= 1.0
        assert mutation_rate > 0.0

        self.mutation_probability: float = mutation_probability
        self.mutation_rate: float = mutation_rate
        self.mutation_type: str = "random_normal"  # "random_uniform", "random_normal"
        self.index_best: int = None

    def _select(self) -> None:
        """Selects best agent."""
        rewards = numpy.array([agent.score for agent in self.agents])
        self.index_best = numpy.argmax(rewards)

    def _mutate_parameters(self) -> None:
        """Mutates models's parameters of each agent by adding Gaussian noise."""

        mutation_rate = self.mutation_rate
        mutation_prob = self.mutation_probability

        for agent in self.agents:
            for weights, biases in zip(agent.model.weights, agent.model.biases):
                noise = numpy.random.uniform(low=-1.0 * mutation_rate, high=mutation_rate, size=weights.shape)
                mask = numpy.random.random(size=weights.shape) < mutation_prob
                numpy.add(weights, mask * noise, out=weights)

                noise = numpy.random.uniform(low=-1.0 * mutation_rate, high=mutation_rate, size=biases.shape)
                mask = numpy.random.random(size=biases.shape) < mutation_prob
                numpy.add(biases, mask * noise, out=biases)

    def _broadcast_parameters(self) -> None:
        """Broadcasts parameters of best agent to all other agents."""
        model = self.agents[self.index_best].model
        for i, agent in enumerate(self.agents):
            if i != self.index_best:
                agent.model.weights = copy.deepcopy(model.weights)
                agent.model.biases = copy.deepcopy(model.biases)

    def step(self) -> None:
        self._select()
        self._broadcast_parameters()
        self._mutate_parameters()


class EvolutionStrategy(Optimizer):
    """Natural evolution strategies class."""

    def __init__(self, agents: list[Agent], learning_rate: float, sigma: float) -> None:
        """Initializes Evolution Strategies optimizer.

        Args:
            agents: Agents.
            learning_rate: Gradient descent learning rate.
            sigma: Noise level.
        """
        super().__init__(agents=agents)
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
            for eps_weights, eps_biases in zip(agent.model.eps_weights, agent.model.eps_biases):
                numpy.add(
                    eps_weights,
                    numpy.random.normal(loc=0.0, scale=self.sigma, size=eps_weights.shape),
                    out=eps_weights,
                )
                numpy.add(
                    eps_biases,
                    numpy.random.normal(loc=0.0, scale=self.sigma, size=eps_biases.shape),
                    out=eps_biases,
                )

    def _mutate_parameters(self) -> None:
        """Mutates models's parameters of each agent by adding Gaussian noise."""
        for agent in self.agents:
            # model = agent.model
            for (weights, biases), (eps_weights, eps_biases) in zip(
                zip(agent.model.weights, agent.model.biases),
                zip(agent.model.eps_weights, agent.model.eps_biases),
            ):
                numpy.add(weights, eps_weights, out=weights)
                numpy.add(biases, eps_biases, out=biases)

    def _initialize_gradients(self) -> None:
        """Initializes gradients."""
        weights, biases = self.parameters
        self.gradients = [(numpy.zeros_like(w), numpy.zeros_like(b)) for w, b in zip(weights, biases)]

    def _zero_gradients(self) -> None:
        """Sets all gradients to zero."""
        for d_w, d_b in self.gradients:
            d_w.fill(0.0)
            d_b.fill(0.0)

    def _compute_gradients(self) -> None:
        """Computes approxiamte gradients."""
        for agent, reward in zip(self.agents, self.rewards):
            for (grad_weights, grad_biases), (eps_weights, eps_biases) in zip(
                self.gradients, zip(agent.model.eps_weights, agent.model.eps_biases)
            ):
                numpy.add(grad_weights, reward * eps_weights, out=grad_weights)
                numpy.add(grad_biases, reward * eps_biases, out=grad_biases)

    def _gradient_descent(self) -> None:
        """Performs gradient descent step."""
        for (weights, biases), (grad_weights, grad_biases) in zip(zip(*self.parameters), self.gradients):
            numpy.add(weights, self.learning_rate * grad_weights, out=weights)
            numpy.add(biases, self.learning_rate * grad_biases, out=biases)

    def _gather_rewards(self) -> None:
        """Gathers rewards from all agents."""
        # Get the rewards.
        rewards = numpy.array([agent.score for agent in self.agents])

        # Normalize rewards.
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        self.rewards = rewards

    def step(self) -> None:
        """Performs single optimization step."""
        self._gather_rewards()
        self._zero_gradients()
        self._compute_gradients()
        self._gradient_descent()
        self._broadcast_parameters()
        self._create_noise()
        self._mutate_parameters()



class ContinuousEvolutionStrategy(Optimizer):
    """Single agent natural evolution strategies class

    Population-free natural evolution strategies optimizer.
    Uses momentum to approximate a population for gradient computation.
    """

    def __init__(
            self, 
            agents: list[Agent], 
            learning_rate: float, 
            sigma: float, 
            momentum: float
        ) -> None:
        """Initializes Evolution Strategies optimizer.

        Args:
            agents: Agents.
            learning_rate: Gradient descent learning rate.
            sigma: Noise level.
        """
        super().__init__(agents=agents)
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.momentum = momentum

        self.model = agents[0].model
        self.parameters = (copy.deepcopy(self.model.weights), copy.deepcopy(self.model.biases))
        self._initialize()

    def _initialize(self) -> None:
        """Initializes agents by broadcasting same parameters and adding noise."""
        self._register_noise()
        self._create_noise()
        self._mutate_parameters()
        self._initialize_gradients()

    def _register_noise(self) -> None:
        """Registers noise in model of each agent."""
        weights, biases = self.model.weights, self.model.biases
        self.model.noise = [(numpy.zeros_like(w), numpy.zeros_like(b)) for w, b in zip(weights, biases)]

    def _create_noise(self) -> None:
        """Creates noise for each parameter."""
        # Zero noise.
        for eps_w, eps_b in self.model.noise:
            eps_w.fill(0.0)
            eps_b.fill(0.0)

        for eps_w, eps_b in self.model.noise:
            noise_w = numpy.random.normal(loc=0.0, scale=self.sigma, size=eps_w.shape) 
            numpy.add(eps_w, noise_w, out=eps_w)
            noise_b = numpy.random.normal(loc=0.0, scale=self.sigma, size=eps_b.shape) 
            numpy.add(eps_b, noise_b, out=eps_b)

    def _mutate_parameters(self) -> None:
        """Mutates models's parameters of each agent by adding Gaussian noise."""
        for (w, b), (eps_w, eps_b) in zip(zip(self.model.weights, self.model.biases), self.model.noise):
            numpy.add(w, eps_w, out=w)
            numpy.add(b, eps_b, out=b)

    def _initialize_gradients(self) -> None:
        """Initializes gradients."""
        weights, biases = self.parameters
        self.grad = [(numpy.zeros_like(w), numpy.zeros_like(b)) for w, b in zip(weights, biases)]

    def _compute_gradients(self) -> None:
        """Computes approxiamte gradients."""
        for (grad_w, grad_b), (eps_w, eps_b) in zip(self.grad, self.model.noise):
            numpy.multiply(grad_w, (1 - self.momentum), out=grad_w)
            numpy.multiply(grad_b, (1 - self.momentum), out=grad_b)
            numpy.add(grad_w, self.momentum * (eps_w * (self.reward / self.sigma)), out=grad_w)
            numpy.add(grad_b, self.momentum * (eps_b * (self.reward / self.sigma)), out=grad_b)
            # numpy.add(grad_w, self.momentum * (self.reward * eps_w) / self.sigma, out=grad_w)
            # numpy.add(grad_b, self.momentum * (self.reward * eps_b) / self.sigma, out=grad_b)

    def _gradient_descent(self) -> None:
        """Performs gradient descent step."""
        for (w, b), (grad_w, grad_b) in zip(zip(*self.parameters), self.grad):
            numpy.add(w, self.learning_rate * grad_w, out=w)
            numpy.add(b, self.learning_rate * grad_b, out=b)

    def _gather_rewards(self) -> None:
        """Gathers reward from agent."""
        self.reward = self.agents[0].score

    def _broadcast_parameters(self) -> None:
        """Broadcasts parameters to agent."""
        weights, biases = self.parameters
        self.model.weights = copy.deepcopy(weights)
        self.model.biases = copy.deepcopy(biases)

    def step(self) -> None:
        """Performs single optimization step."""
        self._gather_rewards()
        self._compute_gradients()
        self._gradient_descent()
        self._broadcast_parameters()
        self._create_noise()
        self._mutate_parameters()
