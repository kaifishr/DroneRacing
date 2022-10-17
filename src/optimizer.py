import numpy as np

from Box2D.Box2D import b2Vec2, b2Color
from src.framework import SimpleFramework as Framework

from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.asset import Domain, Flyer


class Environment(Framework):

    name = "Flyer"
    description = "Learning environment."

    color_raycast_line = b2Color(0, 0, 1)
    color_raycast_head = b2Color(1, 0, 1)

    def __init__(self, config: Config) -> None:
        """Initializes environment class."""
        super().__init__()

        self.config = config

        n_agents = config.optimizer.n_agents

        self.world.gravity = b2Vec2(config.env.gravity.x, config.env.gravity.y)
        self.flyers = [Flyer(world=self.world, config=config) for _ in range(n_agents)]
        self.domain = Domain(world=self.world, config=config)

    def render_force(self):
        """Displays force applied to Flyer.

        Purely cosmetic but helps with debugging. Arrows point towards
        direction the force is coming from.
        """
        alpha = self.config.renderer.scale_force  # Scaling factor
        self.line_color = (1, 0, 0)

        for flyer in self.flyers:
            
            f_left, f_right, f_up, f_down = flyer.forces

            # Left
            local_point_left = b2Vec2(-0.5 * flyer.diam, 0.0)
            force_direction = (-alpha * f_left, 0.0)
            p1 = flyer.body.GetWorldPoint(localPoint=local_point_left)
            p2 = p1 + flyer.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

            # Right
            local_point_right = b2Vec2(0.5 * flyer.diam, 0.0)
            force_direction = (alpha * f_right, 0.0)
            p1 = flyer.body.GetWorldPoint(localPoint=local_point_right)
            p2 = p1 + flyer.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

            # Up
            local_point_up = b2Vec2(0.0, 0.5 * flyer.diam)
            force_direction = (0.0, alpha * f_up)
            p1 = flyer.body.GetWorldPoint(localPoint=local_point_up)
            p2 = p1 + flyer.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))
            
            # Down
            local_point_down = b2Vec2(0.0, -0.5 * flyer.diam)
            force_direction = (0.0, -alpha * f_down)
            p1 = flyer.body.GetWorldPoint(localPoint=local_point_down)
            p2 = p1 + flyer.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

    def reset(self) -> None:
        """Resets all flyers."""
        for flyer in self.flyers:
            flyer.reset() 

    def render_raycast(self):
        """TODO: Add to renderer."""
        for flyer in self.flyers:
            for p1, p2, callback in zip(flyer.p1, flyer.p2, flyer.callbacks):
                p1 = self.renderer.to_screen(p1)
                p2 = self.renderer.to_screen(p2)
                if callback.hit:
                    cb_point = callback.point
                    cb_point = self.renderer.to_screen(cb_point)
                    self.renderer.DrawPoint(cb_point, 5.0, self.color_raycast_head)
                    self.renderer.DrawSegment(p1, cb_point, self.color_raycast_line)
                else:
                    self.renderer.DrawSegment(p1, p2, self.color_raycast_line)

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
            # TODO: move render methods to environment()
            # Physics and rendering
            self.env.step()
            # Optimization

            self.env.ray_casting()
            # if self.is_render:
            self.env.render_raycast()
            self.env.render_force() 
        
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
