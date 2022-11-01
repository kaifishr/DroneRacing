"""Optimizer class for genetic optimization."""
import time

from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.environment import Environment


class Optimizer:
    """Optimizer class.

    Optimizer uses genetic optimization.

    Attributes:
        config:
        env:
        writer:
    """

    def __init__(self, config: Config) -> None:
        """Initializes Optimizer"""
        self.config = config
        self.env = Environment(config=config)
        self.writer = SummaryWriter()

    def run(self) -> None:
        """Runs genetic optimization."""

        num_max_steps = self.config.optimizer.num_max_steps
        step = 0
        generation = 0

        is_running = True
        t0 = time.time()

        while is_running:

            # Physics and rendering.
            self.env.step()

            # Ray casting and positioning.
            self.env.ray_casting()

            # Run neural network prediction.
            self.env.comp_action()

            # Apply network predictions to drone.
            self.env.apply_action()

            # Compute distance covered so far.
            self.env.run_odometer()

            # Method that run at end of simulation.
            if (step + 1) % num_max_steps == 0:

                # Select fittest agent based on distance traveled.
                distance = self.env.select()

                # Reproduce and mutate weights of best agent.
                self.env.mutate()

                # Reset drones to start over again.
                self.env.reset()

                step = 0
                generation += 1

                # Write stats to Tensorboard.
                self.writer.add_scalar("Distance", distance, generation)
                self.writer.add_scalar("seconds/generation", time.time() - t0, generation)
                print(f"{generation = }")

                t0 = time.time()

            step += 1
