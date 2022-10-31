"""Minimal pygame-based framework for Box2D."""
import pygame
from Box2D import b2Vec2
from Box2D.b2 import world

from src.renderer import Renderer

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 640
TARGET_FPS = 60
TIMESTEP = 1.0 / TARGET_FPS
VEL_ITERS = 10  # Iterations to compute next velocity
POS_ITERS = 10  # Iterations to compute next position


class Framework:
    """A simple framework for pybox2d with pygame."""

    name = ""
    description = ""

    def __init__(self):

        self.world = world()
        self.groundbody = self.world.CreateBody()

        # Pygame Initialization
        pygame.init()
        pygame.display.set_caption(self.name)

        # Screen and debug draw
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.font = pygame.font.Font(None, 15)
        self.renderer = Renderer(self.screen)

        self.is_render = True
        self.clock = pygame.time.Clock()

    def _set_render(self):
        """Sets rendering on or off."""
        self.is_render = False if self.is_render else True

    def step(self):
        """Catches events, performs simulation step, and renders world."""

        # Catch events
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._set_render()
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._set_render()
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

        # Step the world
        self.world.Step(TIMESTEP, VEL_ITERS, POS_ITERS)
        self.world.ClearForces()

        if self.is_render:
            self.screen.fill((0, 0, 0))  # TODO: move this to renderer?
            self.renderer.render(self.world)
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)
            self.fps = self.clock.get_fps()

        print(f"FPS {self.fps:.0f}", flush=True, end="\r")

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None
