"""Environment class.

The Environment class holds the worlds objects that 
can interact with each other. Currently, these are

    - Drones
    - Domain walls

The Environment class also wraps the Framework class 
that calls the pyhsics engine and rendering engines.
"""
import numpy as np

from Box2D.Box2D import b2Vec2

from src.body import Domain, Drone
from src.utils.config import Config
from src.framework import Framework


class Environment(Framework):
    """Environment holding all world bodies.

    This class holds the world's bodies as well as methods
    to control the drones.

    Attributes:
        domain: Domain walls.
        drones:
    """

    def __init__(self, config: Config) -> None:
        """Initializes environment class."""
        super().__init__(config=config)

        num_agents = config.optimizer.num_agents

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)
        Domain(world=self.world, config=config)
        self.drones = [
            Drone(world=self.world, config=config) for _ in range(num_agents)
        ]

        # Add reference of drones to world class for easier rendering handling.
        setattr(self.world, "drones", self.drones)

        # Index of current fittest agent
        self.idx_best = 0

    def reset(self) -> None:
        """Resets all drones."""
        for drone in self.drones:
            drone.reset()

    def ray_casting(self) -> None:
        """Runs ray casting for each drone"""
        for drone in self.drones:
            drone.ray_casting()

    def collision_detection(self) -> None:
        """Detects collision of drone."""
        for drone in self.drones:
            drone.detect_collision()

    def comp_score(self) -> None:
        """Computes fitness score of each drones."""
        for drone in self.drones:
            drone.comp_score()

    def comp_action(self) -> None:
        """Computes next set of actions."""
        for drone in self.drones:
            drone.comp_action()

    def apply_action(self) -> None:
        """Applies action coming from neural network to all drones."""
        for drone in self.drones:
            drone.apply_action()

    def select(self) -> float:
        """Selects best agent for reproduction."""
        scores = [drone.score for drone in self.drones]
        self.idx_best = np.argmax(scores)
        return scores[self.idx_best]

    def mutate(self) -> None:
        """Mutates network parameters of each drone."""
        # Get network of fittest drone to reproduce.
        model = self.drones[self.idx_best].model

        # Pass best model to other drones and mutate their weights.
        for drone in self.drones:
            drone.mutate(model)
