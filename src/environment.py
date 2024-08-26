"""Environment class.

The Environment class holds the worlds objects that can interact with each
other.

The Environment class also wraps the Framework class that calls the physics
engine and rendering engines.
"""
import random

import numpy
from Box2D.Box2D import b2Vec2

from src.track.track import Track 
from src.drone import Drone
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

        self.world.gravity = b2Vec2(0.0, 0.0)

        # Race track
        self.world.track = Track(world=self.world, config=config)

        # Create agents.
        num_agents = config.optimizer.num_agents
        self.drones = [Drone(world=self.world, config=config) for _ in range(num_agents)]
        self.world.drones = self.drones

        # Domain
        self.x_max = config.env.domain.limit.x_max
        self.x_min = config.env.domain.limit.x_min
        self.y_max = config.env.domain.limit.y_max
        self.y_min = config.env.domain.limit.y_min

        self.phi = 0.95

    def next_target(self) -> None:
        for drone in self.world.drones:
            if drone.body.active:
                distance = (drone.next_target.position - drone.body.position).length
                if distance < drone.next_target.gate_size:
                    drone.idx_next_target = (drone.idx_next_target + 1) % len(drone.targets)
                    drone.next_target = drone.targets[drone.idx_next_target]

    def reset(self) -> None:
        """Resets Drone to initial position and velocity."""

        if self.config.env.drone.respawn.is_random:
            # Respawn drone at random location between two gates on track.
            num_gates = len(self.world.track.gates)
            idx_gate_1 = random.randint(0, num_gates - 1)
            idx_gate_2 = (idx_gate_1 + 1) % num_gates
            gate_1 = self.world.track.gates[idx_gate_1]
            gate_2 = self.world.track.gates[idx_gate_2]

            x_1 = gate_1.position.x
            y_1 = gate_1.position.y
            x_2 = gate_2.position.x
            y_2 = gate_2.position.y

            if x_1 == x_2:
                x_random = x_1
                y_random = random.uniform(a=y_1, b=y_2)
            else:
                x_random = random.uniform(a=x_1, b=x_2)
                slope = (y_2 - y_1) / (x_2 - x_1)
                y_random = slope * (x_random - x_1) + y_1

            init_position_rand = b2Vec2(x_random, y_random)

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
            drone.idx_next_target = idx_gate_2
            drone.next_target = drone.targets[drone.idx_next_target]

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

    def comp_reward(self) -> None:
        for drone in self.drones:
            drone.comp_reward()

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
        rewards = numpy.array([agent.reward for agent in self.drones])
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
        return numpy.argmax([agent.reward for agent in self.drones])
