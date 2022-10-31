"""Optimizer class for genetic optimization."""

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

                # Get index of agent who traveled the farthest.
                idx_best, distance = self.env.get_distance()

                # Pass networks weights of best agent to next generation
                # and mutate their weights.
                self.env.mutate(idx_best)

                # Reset drones to start over again.
                self.env.reset()

                step = 0
                generation += 1

                # Write distance traveled to Tensorboard.
                self.writer.add_scalar("Distance", distance, generation)
                print(f"{generation = }")

            step += 1
