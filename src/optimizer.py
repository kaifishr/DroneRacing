import time
import numpy as np

from Box2D.Box2D import b2Vec2, b2Color
from src.framework import Framework

from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.body import Domain, Drone


class Environment(Framework):

    name = "Drones"
    description = "Learning environment."

    color_raycast_line = b2Color(0, 0, 1)
    color_raycast_head = b2Color(1, 0, 1)

    def __init__(self, config: Config) -> None:
        """Initializes environment class."""
        super().__init__()

        self.config = config

        n_agents = config.optimizer.n_agents

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)
        self.flyers = [Drone(world=self.world, config=config) for _ in range(n_agents)]
        self.domain = Domain(world=self.world, config=config)

        setattr(self.world, "flyers", self.flyers)

    def reset(self) -> None:
        """Resets all flyers."""
        for flyer in self.flyers:
            flyer.reset()

    def ray_casting(self) -> None:
        """Runs ray casting for each Flyer"""
        for flyer in self.flyers:
            flyer.ray_casting()

    def mutate(self, idx_best: int) -> None:
        """Mutates vertices of box."""
        # Get network of best fitest flyer
        model = self.flyers[idx_best].model
        # Pass best model to other flyers and mutate weights
        for flyer in self.flyers:
            flyer.mutate(model)

    def apply_action(self) -> None:
        """Applies action to all boxes."""
        for flyer in self.flyers:
            flyer.apply_action()

    def run_odometer(self) -> None:
        """Computes distances traveled by Flyer."""
        for flyer in self.flyers:
            flyer.odometer()

    def get_distance(self) -> None:
        """Gets distance traveled by Flyers."""
        distances = [flyer.distance for flyer in self.flyers]
        idx_best = np.argmax(distances)
        return idx_best, distances[idx_best]

    def comp_action(self) -> None:
        """Computes next set of actions.

        Next steps of action are computed by feeding obstacle data
        to the neural network.
        """
        for flyer in self.flyers:
            flyer.comp_action()


class Optimizer:
    """Optimizer class for Flyer"""

    def __init__(self, config: Config):

        self.config = config
        self.n_max_iterations = config.optimizer.n_max_iterations

        self.env = Environment(config=config)

        self.writer = SummaryWriter()
        self.iteration = 0
        self.generation = 0

    def run(self) -> None:

        is_running = True

        while is_running:

            # Physics and rendering
            self.env.step()

            # Ray casting -> Change order. Move before step()?
            self.env.ray_casting()

            # Method that run every simulation step
            self.env.comp_action()
            self.env.apply_action()
            self.env.run_odometer()

            # Method that run at end of simulation
            if (self.iteration + 1) % self.n_max_iterations == 0:

                # Get index of agent who traveled the farthest
                idx_best, distance = self.env.get_distance()

                self.env.mutate(idx_best)
                self.env.reset()

                self.iteration = 0
                self.generation += 1

                self.writer.add_scalar("Distance", distance, self.generation)
                print(f"Generation {self.generation}")

            self.iteration += 1
