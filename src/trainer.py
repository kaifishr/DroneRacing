"""Optimizer class for genetic optimization."""
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.environment import Environment
from src.optimizer import Optimizer
from src.utils.config import Config
from src.utils.utils import save_checkpoint


class Trainer:
    """Optimizer class.

    Optimizer uses genetic optimization.

    Attributes:
        config:
        env:
        writer:
    """

    def __init__(self, env: Environment, optimizer: Optimizer, config: Config) -> None:
        """Initializes Optimizer"""
        self.env = env
        self.optimizer = optimizer
        self.config = config

        self.writer = SummaryWriter()

        # Save config file
        file_path = Path(self.writer.log_dir) / "config.txt"
        with open(file_path, "w") as file:
            file.write(self.config.__str__())

    def run(self) -> None:
        """Runs genetic optimization."""

        num_max_steps = self.config.optimizer.num_max_steps  # max_episode_length
        step = 0
        generation = 0
        best_score = 0.0

        is_running = True
        t0 = time.time()

        while is_running:
            # Physics and rendering.
            self.env.step()

            # Fetch data for neural network.
            self.env.fetch_data()

            # Detect collisions with other bodies
            if not self.config.env.allow_collision_domain:
                self.env.collision_detection()

            # Compute current fitness / score of drone
            self.env.comp_score()

            # Run neural network prediction
            self.env.comp_action()

            # Apply network predictions to drone
            self.env.apply_action()

            # Method that run at end of simulation.
            if ((step + 1) % num_max_steps == 0) or self.env.is_done():
                self.optimizer.step()

                # Select fittest agent based on distance traveled.
                results = self.env.comp_mean_reward()

                # Reset drones to start over again.
                self.env.reset()

                # Write stats to Tensorboard.
                for result_name, result_value in results.items():
                    self.writer.add_scalar(result_name, result_value, generation)
                self.writer.add_scalar("seconds_episode", time.time() - t0, generation)

                # Save model
                if self.config.checkpoints.save_model:
                    if results["mean_reward"] > best_score:
                        model = self.env.drones[self.env.idx_best].model
                        save_checkpoint(model=model, config=self.config)
                        best_score = results["mean_reward"]

                step = 0
                generation += 1
                print(f"{generation = }")

                t0 = time.time()

            step += 1
