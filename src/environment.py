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
from src.config import Config
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
        super().__init__()

        num_agents = config.optimizer.num_agents

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)
        self.domain = Domain(world=self.world, config=config)
        self.drones = [Drone(world=self.world, config=config) for _ in range(num_agents)]

        # Add reference of drones to world class for easier rendering handling.
        setattr(self.world, "drones", self.drones)

    def reset(self) -> None:
        """Resets all drones."""
        for drone in self.drones:
            drone.reset()

    def ray_casting(self) -> None:
        """Runs ray casting for each drone"""
        for drone in self.drones:
            drone.ray_casting()

    def apply_action(self) -> None:
        """Applies action coming from neural network to all drones."""
        for drone in self.drones:
            drone.apply_action()

    def run_odometer(self) -> None:
        """Runs odometer to compute distance covered by each drone."""
        for drone in self.drones:
            drone.odometer()

    def comp_action(self) -> None:
        """Computes next set of actions."""
        for drone in self.drones:
            drone.comp_action()

    def mutate(self, idx_best: int) -> None:
        """Mutates network parameters of each drone.

        Args:
            idx_best: Index of best drone.
        """
        # Get network of fittest drone.
        model = self.drones[idx_best].model

        # Pass best model to other drones and mutate weights.
        for drone in self.drones:
            drone.mutate(model)

    def get_distance(self) -> None:
        """Gets distance traveled by drones."""
        distances = [drone.distance for drone in self.drones]
        idx_best = np.argmax(distances)
        return idx_best, distances[idx_best]