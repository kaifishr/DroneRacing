from src.environment import Environment
from src.utils.config import Config


class Eval:

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

            # Fetch data for neural network.
            self.env.fetch_data()

            # Detect collisions with other bodies
            if not self.config.env.allow_collision_domain:
                self.env.collision_detection()

            # Run neural network prediction
            self.env.comp_action()

            # Apply network predictions to drone
            self.env.apply_action(noise=5.0)

            # Select next target.
            self.env.next_target()

            # Method that run at end of simulation.
            if self.env.is_done():
                # Reset drones to start over again.
                self.env.reset()

                step = 0
                generation += 1
                print(f"{generation = }")

            step += 1
