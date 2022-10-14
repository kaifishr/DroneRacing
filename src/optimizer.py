DEBUG = True

import time

import numpy as np

from Box2D.Box2D import b2Vec2, b2Color

if DEBUG:
    from Box2D.examples.framework import Framework
else:
    from src.framework import SimpleFramework as Framework

from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.asset import Domain, Flyer


class Optimizer(Framework):
    """Optimizer class for Flyer"""

    name = "Flyer"
    description = "Simple learning environment."

    color_raycast_line = b2Color(0, 0, 1)
    color_raycast_head = b2Color(1, 0, 1)

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        n_agents = config.optimizer.n_agents
        self.n_max_iterations = config.optimizer.n_max_iterations

        self.world.gravity = (self.config.env.gravity.x, self.config.env.gravity.y)
        self.boxes = [
            Flyer(world=self.world, config=self.config) for _ in range(n_agents)
        ]
        self.domain = Domain(world=self.world, config=self.config)

        self.writer = SummaryWriter()
        self.iteration = 0
        self.generation = 0

    def reset(self) -> None:
        """Resets all wheels to initial parameter."""
        for box in self.boxes:
            box.reset()     # todo: Reset distance traveled

    def is_awake(self) -> bool:
        """Checks if box in simulation are awake.

        Returns:
            True if at least one body is awake.
        """
        for box in self.boxes:
            if box.body.awake:
                return True

        return False

    def mutate(self, idx_best: int) -> None:
        """Mutates vertices of box."""
        # Get network of best fitest flyer
        model = self.boxes[idx_best].model
        # Pass best model to other flyers and mutate weights 
        for box in self.boxes:
            box.mutate(model)

    def apply_action(self) -> None:
        """Applies action to all boxes."""
        for box in self.boxes:
            box.apply_action()

    def _render_force(self):
        """Displays force applied to Flyer.

        Purely cosmetic but helps with debugging. Arrows point towards
        direction the force is coming from.
        """
        alpha = self.config.renderer.scale_force  # Scaling factor
        self.line_color = (1, 0, 0)

        for box in self.boxes:
            
            f_left, f_right, f_up, f_down = box.forces

            # Left
            local_point_left = b2Vec2(-0.5 * box.diam, 0.0)
            force_direction = (-alpha * f_left, 0.0)
            p1 = box.body.GetWorldPoint(localPoint=local_point_left)
            p2 = p1 + box.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

            # Right
            local_point_right = b2Vec2(0.5 * box.diam, 0.0)
            force_direction = (alpha * f_right, 0.0)
            p1 = box.body.GetWorldPoint(localPoint=local_point_right)
            p2 = p1 + box.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

            # Up
            local_point_up = b2Vec2(0.0, 0.5 * box.diam)
            force_direction = (0.0, alpha * f_up)
            p1 = box.body.GetWorldPoint(localPoint=local_point_up)
            p2 = p1 + box.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))
            
            # Down
            local_point_down = b2Vec2(0.0, -0.5 * box.diam)
            force_direction = (0.0, -alpha * f_down)
            p1 = box.body.GetWorldPoint(localPoint=local_point_down)
            p2 = p1 + box.body.GetWorldVector(force_direction)
            self.renderer.DrawSegment(self.renderer.to_screen(p1), self.renderer.to_screen(p2), b2Color(*self.line_color))

    def run_odometer(self) -> None:
        """Computes distances traveled by Flyer."""
        for box in self.boxes:
            box.odometer()

    def get_distance(self) -> None:
        """Gets distance traveled by Flyers."""
        distances = [box.distance for box in self.boxes]
        idx_best = np.argmax(distances)
        return idx_best, distances[idx_best]

    def comp_action(self) -> None:
        """Computes next set of actions.
        
        Next steps of action are computed by feeding obstacle data
        to the neural network.
        """
        for box in self.boxes:
            box.comp_action()

    def _step(self) -> None:
        """Performs single optimization step."""
        t_0 = time.time()

        # Method that run every simulation step
        self.comp_action()
        self.apply_action()
        self.run_odometer()

        # Method that run at end of simulation 
        if not self.is_awake() or (self.iteration + 1) % self.n_max_iterations == 0:

            # Get index of agent who traveled the farthest
            idx_best, distance = self.get_distance()

            self.mutate(idx_best)
            self.reset()

            self.writer.add_scalar("Distance", distance, self.generation)
            self.writer.add_scalar("Time_Generation", time.time() - t_0, self.generation)

            self.iteration = 0
            self.generation += 1

        self.iteration += 1

    def _render_raycast(self, cb_, p1_, p2_):
        """TODO: Add to renderer."""
        for p1, p2, callback in zip(p1_, p2_, cb_):
            p1 = self.renderer.to_screen(p1)
            p2 = self.renderer.to_screen(p2)
            if callback.hit:
                # print("Hit:", callback.point)
                cb_point = callback.point
                cb_point = self.renderer.to_screen(cb_point)
                self.renderer.DrawPoint(cb_point, 5.0, self.color_raycast_head)
                self.renderer.DrawSegment(p1, cb_point, self.color_raycast_line)
            else:
                # print("No hit.")
                self.renderer.DrawSegment(p1, p2, self.color_raycast_line)

    def _ray_casting(self) -> None:
        """Decouple ray_casting() from _render_raycast()
        """
        for box in self.boxes:
            cb_, p1_, p2_ = box.ray_casting()
            self._render_raycast(cb_, p1_, p2_)

    def Step(self, settings):
        super(Optimizer, self).Step(settings)
        self._render_force()
        self._ray_casting()
        self._step()

    def run(self) -> None:
        if DEBUG:
            super().run()
        else:
            while True:
                # Physics and rendering
                self.step()
                # Optimization
                self._step()
