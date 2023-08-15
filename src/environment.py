"""Environment class.

The Environment class holds the worlds objects that 
can interact with each other. Currently, these are

    - Drones
    - Domain walls

The Environment class also wraps the Framework class 
that calls the physics engine and rendering engines.
"""
import random
import numpy

from Box2D.Box2D import b2Vec2

from src.domain import Domain
from src.domain import StaticTarget
# from src.environment import Target
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

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)

        Domain(world=self.world, config=config)

        # Create agents.
        num_agents = config.optimizer.num_agents
        self.drones = [
            Drone(world=self.world, config=config) for _ in range(num_agents)
        ]

        # Domain
        self.x_max = config.env.domain.limit.x_max
        self.x_min = config.env.domain.limit.x_min
        self.y_max = config.env.domain.limit.y_max
        self.y_min = config.env.domain.limit.y_min

        self.target = StaticTarget(world=self.world, config=config)
        self.target

        # Add reference of drones to world class for easier rendering handling.
        self.world.drones = self.drones
        self.world.target = self.target

    def _move_target(self) -> None:
        """Moves target to random position."""
        x_pos = random.uniform(self.x_min, self.x_max)
        y_pos = random.uniform(self.y_min, self.y_max)
        self.target.body.position = b2Vec2(x_pos, y_pos)

    def reset(self) -> None:
        """Resets Drone to initial position and velocity."""

        self._move_target()

        # # Set new target
        # x_pos = random.uniform(self.x_min, self.x_max)
        # y_pos = random.uniform(self.y_min, self.y_max)
        # self.target.body.position = b2Vec2(x_pos, y_pos)

        if self.config.env.drone.respawn.is_random:
            # Respawn drones every generation at different
            # predefined location in map.
            phi = 0.9
            # respawn_points = [
            #     b2Vec2(phi * self.x_max, phi * self.y_max),
            #     b2Vec2(phi * self.x_min, phi * self.y_max),
            #     b2Vec2(phi * self.x_min, phi * self.y_min),
            #     b2Vec2(phi * self.x_max, phi * self.y_min),
            # ]
            # init_position_rand = random.choice(respawn_points)
            init_position_rand = b2Vec2(
                random.uniform(a=phi * self.x_min, b=phi * self.x_max),
                random.uniform(a=phi * self.y_min, b=phi * self.y_max),
            )

        for drone in self.drones:
            if self.config.env.drone.respawn.is_random:
                drone.body.position = init_position_rand
            else:
                drone.body.position = drone.init_position

            drone.body.linearVelocity = drone.init_linear_velocity
            drone.body.angularVelocity = drone.init_angular_velocity
            drone.body.angle = drone.init_angle

            # Reset fitness score for next generation.
            drone.score = 0.0

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

    def apply_action(self) -> None:
        """Applies action coming from neural network to all drones."""
        for drone in self.drones:
            drone.apply_action()

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
        results["mean_reward"] = rewards.mean()
        results["min_reward"] = rewards.min()
        results["max_reward"] = rewards.max()
        return results
