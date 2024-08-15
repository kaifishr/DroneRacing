"""Environment class.

The Environment class holds the worlds objects that can interact with each
other.

The Environment class also wraps the Framework class that calls the physics
engine and rendering engines.
"""
import random

import numpy
from Box2D.Box2D import b2Vec2

from src.domain import Domain
from src.drone import Drone
from src.snitch import Snitch
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

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)

        Domain(world=self.world, config=config)

        # Create agents.
        num_agents = config.optimizer.num_agents
        self.drones = [Drone(world=self.world, config=config) for _ in range(num_agents)]
        self.world.drones = self.drones

        # Create moving target.
        # self.target = Snitch(world=self.world, config=config)
        # self.world.target = self.target
        
        # Domain
        self.x_max = config.env.domain.limit.x_max
        self.x_min = config.env.domain.limit.x_min
        self.y_max = config.env.domain.limit.y_max
        self.y_min = config.env.domain.limit.y_min

        self.phi = 0.95

    def next_target(self) -> None:
        for drone in self.world.drones:
            distance = (drone.target - drone.body.position).length
            if distance < self.config.env.target.distance_threshold:
                # drone.idx_target = (drone.idx_target + 1) % len(drone.targets)
                drone.idx_target = drone.idx_target + 1
                drone.idx_target = drone.idx_target % len(drone.targets)
                drone.target = drone.targets[drone.idx_target]

    def reset(self) -> None:
        """Resets Drone to initial position and velocity."""

        if self.config.env.drone.respawn.is_random:
            init_position_rand = b2Vec2(
                random.uniform(a=self.phi * self.x_min, b=self.phi * self.x_max),
                random.uniform(a=self.phi * self.y_min, b=self.phi * self.y_max),
            )

        for drone in self.drones:
            if self.config.env.drone.respawn.is_random:
                drone.body.position = init_position_rand
            elif self.config.env.drone.respawn.is_all_random:
                # Respawn each drones at different location in map.
                pos_x = random.uniform(a=self.phi * self.x_min, b=self.phi * self.x_max)
                pos_y = random.uniform(a=self.phi * self.y_min, b=self.phi * self.y_max)
                position = b2Vec2(pos_x, pos_y)
                drone.body.position = position
            else:
                drone.body.position = drone.init_position

            drone.body.linearVelocity = drone.init_linear_velocity
            drone.body.angularVelocity = drone.init_angular_velocity
            drone.body.angle = drone.init_angle

            # Reset fitness score for next generation.
            drone.score = 0.0

            # Reset to first target.
            drone.idx_target = 0
            drone.target = drone.targets[drone.idx_target]

            # Reactivate drone after collision in last generation.
            drone.body.active = True


    def fetch_data(self) -> None:
        """Fetches data for each drone"""
        for drone in self.drones:
            drone.fetch_data()

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

    def apply_action(self, noise: float = None) -> None:
        """Applies action coming from neural network to all drones.

        Args:
            noise: Amount of noise added to action of each drone. Default: None
        """
        for drone in self.drones:
            drone.apply_action(noise=noise)

    def is_done(self) -> bool:
        """Checks if episode is done.

        Simulation is done if there is no active agent.
        """
        for drone in self.drones:
            if drone.body.active:
                return False
        return True

    def comp_mean_reward(self) -> dict[str, float]:
        """Computes mean reward over all agents.

        TODO: Move to trainer.

        Returns:
            Dictionary holding reward metrics.
        """
        results = {}
        rewards = numpy.array([agent.score for agent in self.drones])
        results["min_reward"] = rewards.min()
        results["max_reward"] = rewards.max()
        results["mean_reward"] = rewards.mean()
        distances = numpy.array([agent.distance_to_target for agent in self.drones])
        results["min_distance"] = distances.min()
        results["max_distance"] = distances.max()
        results["mean_distance"] = distances.mean()
        return results

    def index_best_agent(self) -> int:
        """Computes best agent based on rewards.

        Returns:
            Index of agent.
        """
        return numpy.argmax([agent.score for agent in self.drones])
