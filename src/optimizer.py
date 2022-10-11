DEBUG = True

import pygame

import math
import time
import random

import numpy as np

from Box2D.Box2D import b2Vec2, b2Color, b2RayCastCallback

if DEBUG:
    from Box2D.examples.framework import Framework
else:
    from src.framework import SimpleFramework as Framework

from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.body import Domain, BoosterBox 


class RayCastCallback(b2RayCastCallback):
    """Callback detects closest hit.

    See also this example for more information about ray casting in PyBox2D:
    https://github.com/pybox2d/pybox2d/blob/master/library/Box2D/examples/raycast.py
    
    """
    def __init__(self, **kwargs) -> None:
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction) -> float:
        """Reports hit fixture.
        """
        self.hit = True
        self.fixture = fixture          # Fixture of the hit body. Interesting for multi-agent environments.
        self.point = b2Vec2(point)      # Point of contact of body.
        self.normal = b2Vec2(normal)    # Normal vector at point of contact. Perpendicular to body surface.
        return fraction


class Optimizer(Framework):
    """Optimizer class for BoxScout"""

    name = "BoosterBox"
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
            BoosterBox(world=self.world, config=self.config) for _ in range(n_agents)
        ]
        self.domain = Domain(world=self.world, config=self.config)

        self.writer = SummaryWriter()
        self.iteration = 0
        self.generation = 0

        self.callback = RayCastCallback

    def reset(self) -> None:
        """Resets all wheels to initial parameter."""
        for box in self.boxes:
            box.reset()

    def comp_fitness(self) -> float:
        """Computes maximum fitness of box.

        The fitness is determined by the vertical
        distance traveled by the box.

        This method is called after box have stoped
        moving or if maximum number of iterations is
        reached.

        Returns:
            List holding fitness scores.
        """
        scores = [box.body.position.x for box in self.boxes]
        idx_best = np.argmax(scores)
        return idx_best, scores[idx_best]

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
        # Get vertices of best box:
        vertices = self.boxes[idx_best].vertices
        # Pass best vertices to all box 
        for box in self.boxes:
            box.mutate(vertices)

    def apply_action(self) -> None:
        """Applies action to all boxes."""
        for box in self.boxes:
            box.apply_action()

    def _render_force(self):
        """Displays force applied to BoosterBox.

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

    def _step(self) -> None:
        """Performs single optimization step."""
        t_0 = time.time()

        self.apply_action()

        if not self.is_awake() or (self.iteration + 1) % self.n_max_iterations == 0:
            idx_best, max_score = self.comp_fitness()
            self.mutate(idx_best)
            self.reset()

            self.writer.add_scalar("Score", max_score, self.generation)
            self.writer.add_scalar(
                "Time_Generation", time.time() - t_0, self.generation
            )

            self.iteration = 0
            self.generation += 1

        self.iteration += 1

    def render_raycast(self, p1, p2, callback):
        """Add to renderer."""
        p1 = self.renderer.to_screen(p1)
        p2 = self.renderer.to_screen(p2)
        if callback.hit:
            print("Hit:", callback.point)
            cb_point = callback.point
            cb_point = self.renderer.to_screen(cb_point)
            self.renderer.DrawPoint(cb_point, 10.0, self.color_raycast_head)
            self.renderer.DrawSegment(p1, cb_point, self.color_raycast_line)
        else:
            print("No hit.")
            self.renderer.DrawSegment(p1, p2, self.color_raycast_line)

    def Step(self, settings):
        super(Optimizer, self).Step(settings)
        self._step()
        self._render_force()

        ###########
        # Raycast #
        ###########

        # Set up the raycast line
        ray_length = 7      # Determines how far we can see.
        points = [b2Vec2(0, ray_length), b2Vec2(-ray_length, 0), b2Vec2(0, -ray_length), b2Vec2(ray_length, 0)]

        for p2 in points:
            p1 = b2Vec2(0, 0)
            callback = self.callback()
            self.world.RayCast(callback, p1, p2)
            self.render_raycast(p1, p2, callback)

    def run(self) -> None:
        if DEBUG:
            super().run()
        else:
            while True:
                # Physics and rendering
                self.step()
                # Optimization
                self._step()
