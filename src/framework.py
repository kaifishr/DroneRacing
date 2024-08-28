"""Minimal pygame-based framework for Box2D.

For more information about frameworks see also:
https://github.com/pybox2d/pybox2d/tree/master/library/Box2D/examples/backends

"""

import sys
import datetime
import pygame
from Box2D.b2 import world

from src.renderer import Renderer
from src.utils.config import Config

# from src.utils.utils import capture_screen


class Framework:
    """A simple framework for PyBox2D with PyGame backend."""

    target_fps: int = 60
    velocity_iterations: int = 10  # Iterations to compute next velocity.
    position_iterations: int = 10  # Iterations to compute next position.

    def __init__(self, config: Config) -> None:
        """Initializes Framework."""
        self.config = config

        # Physics simulation parameters.
        self.time_step = 1.0 / self.target_fps

        # Instantiating world.
        self.world = world()
        self.world.CreateBody()

        # Pygame initialization.
        pygame.init()
        pygame.display.set_caption(f"DroneRacer {datetime.datetime.now()}")
        self.clock = pygame.time.Clock()

        # Set screen properties.
        screen_width = self.config.framework.screen.width
        screen_height = self.config.framework.screen.height
        screen = pygame.display.set_mode((screen_width, screen_height))

        # Rendering.
        self.renderer = Renderer(screen=screen, config=config)
        self.is_rendering = True

        self.iteration = 0

    def _set_rendering(self) -> None:
        """Turns rendering on or off."""
        if self.is_rendering:
            self.is_rendering = False
        else:
            self.is_rendering = True

    def step(self) -> None:
        """Catches events, performs simulation step, and renders world."""

        # Catch events.
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._set_rendering()
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._set_rendering()
                if event.key == pygame.K_r:
                    self.renderer.set_render_rays()
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        self.world.Step(self.time_step, self.velocity_iterations, self.position_iterations)
        self.world.ClearForces()

        if self.is_rendering:
            self.renderer.render(self.world)
            pygame.display.flip()
            self.clock.tick(self.target_fps)
            print(f"{self.clock.get_fps():.1f} FPS", flush=True, end="\r")

            # Capture screen
            # capture_screen(screen=self.renderer.screen, iteration=self.iteration)
            # self.iteration += 1

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None
