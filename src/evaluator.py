"""Optimizer class for genetic optimization."""
from src.environment import Environment
from src.utils.config import Config


class Eval:
    """Evaluation class.

    Attributes:
        config:
        env:
        writer:
    """

    def __init__(self, env: Environment, config: Config) -> None:
        """Initializes class."""
        self.env = env
        self.config = config

    def run(self) -> None:
        """Runs genetic optimization."""

        step = 0
        generation = 0

        self.env.reset()
        is_running = True

        while is_running:
            # Physics and rendering.
            self.env.step()

            if (step + 1) % 300 == 0:
                self.env._move_target()  # TODO: Make _move_target() part of a callback.

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
            if self.env.is_done():
                # Reset drones to start over again.
                self.env.reset()

                step = 0
                generation += 1
                print(f"{generation = }")

            step += 1
