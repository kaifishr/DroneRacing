"""Minimal pygame-based framework for Box2D.

See also: https://github.com/pybox2d/pybox2d/blob/f8617198555d2e1539b0be82f0ee6e7db4d26085/library/Box2D/examples/backends/opencv_draw.py
for renderer.
"""
import pygame
from pygame.locals import QUIT, KEYDOWN 
from Box2D import b2Vec2, b2Color
from Box2D.b2 import (
    world,
    staticBody,
    dynamicBody,
    kinematicBody,
    polygonShape,
    edgeShape,
)

from src.asset.flyer import Flyer

PPM = 15.0  # ZOOM, pixels per meter
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 640
SCREEN_OFFSETX, SCREEN_OFFSETY = 0.5 * SCREEN_WIDTH, 0.5 * SCREEN_HEIGHT
colors = {
    staticBody: (0, 0, 0, 255),
    dynamicBody: (127, 127, 127, 255),
    kinematicBody: (127, 127, 230, 255),
}
# Target frames per second.
TARGET_FPS = 60
TIMESTEP = 1.0 / TARGET_FPS
# Iterations to compute next velocity
VEL_ITERS = 10
# Iterations to compute next position
POS_ITERS = 10

viewZoom = 15.0
viewOffset = b2Vec2(-0.5 * SCREEN_WIDTH, -0.5 * SCREEN_HEIGHT)
screenSize = b2Vec2(SCREEN_WIDTH, SCREEN_HEIGHT)

class Renderer: # Drawer

    def __init__(self, screen) -> None:
        self.screen = screen
        # Installing drawing methods
        edgeShape.draw = self._draw_edge
        polygonShape.draw = self._draw_polygon

        self.flip_x = False
        self.flip_y = True

        self.view_zoom = None # --> self.ppm or self.pixel_per_meter

    def render(self, world):

        # Render bodies
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)

        # Render forces
        for flyer in world.flyers:
            self._draw_force(flyer)

        # Render rays
        for flyer in world.flyers:
            self._draw_raycast(flyer)

    def _to_screen(self, point: b2Vec2) -> tuple:
        """Transforms point from simulation to screen coordinates."""
        x = point.x * viewZoom - viewOffset.x
        if self.flip_x: x = screenSize.x - x
        y = point.y * viewZoom - viewOffset.y 
        if self.flip_y: y = screenSize.y - y
        return (int(x), int(y))

    def _draw_point(self, point, size, color):
        """Draws point in specified size and color."""
        self._draw_circle(point, size / viewZoom, color, width=0)

    def _draw_circle(self, center, radius, color, width=1):
        """Draws circle in specified size and color."""
        radius *= viewZoom
        radius = 1 if radius < 1 else int(radius)
        pygame.draw.circle(self.screen, color.bytes, center, radius, width)

    def _draw_segment(self, p1, p2, color):
        """Draws line from points p1 to p2 in specified color."""
        pygame.draw.aaline(self.screen, color.bytes, p1, p2)

    def _draw_raycast(self, flyer: Flyer) -> None:
        for p1, p2, callback in zip(flyer.p1, flyer.p2, flyer.callbacks):
            p1 = self._to_screen(p1)
            p2 = self._to_screen(p2)
            # DEBUG >
            self._draw_point((10, 10), 5.0, b2Color(0, 0, 0))
            self._draw_point((600, 600), 10.0, b2Color(0, 0, 0))
            # DEBUG <
            if callback.hit:
                cb_point = callback.point
                cb_point = self._to_screen(cb_point)
                self._draw_point(cb_point, 2.0, b2Color(0.2, 0.3, 0.5))
                self._draw_segment(p1, cb_point, b2Color(0.2, 0.3, 0.5))
            else:
                self._draw_segment(p1, p2, b2Color(0.2, 0.3, 0.5))

    def _draw_force(self, flyer: Flyer) -> None:
        """Draws force vectors.

        Purely cosmetic but helps with debugging. 
        Arrows point towards direction the force is coming from.
        """
        # alpha = self.config.renderer.scale_force  # Scaling factor
        alpha = 1.0  # Scaling factor
        line_color = b2Color(1, 0, 0)

        f_left, f_right, f_up, f_down = flyer.forces

        # Left
        local_point_left = b2Vec2(-0.5 * flyer.diam, 0.0)
        force_direction = (-alpha * f_left, 0.0)
        p1 = flyer.body.GetWorldPoint(localPoint=local_point_left)
        p2 = p1 + flyer.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p1), self._to_screen(p2), line_color)

        # Right
        local_point_right = b2Vec2(0.5 * flyer.diam, 0.0)
        force_direction = (alpha * f_right, 0.0)
        p1 = flyer.body.GetWorldPoint(localPoint=local_point_right)
        p2 = p1 + flyer.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p1), self._to_screen(p2), line_color)

        # Up
        local_point_up = b2Vec2(0.0, 0.5 * flyer.diam)
        force_direction = (0.0, alpha * f_up)
        p1 = flyer.body.GetWorldPoint(localPoint=local_point_up)
        p2 = p1 + flyer.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p1), self._to_screen(p2), line_color)
        
        # Down
        local_point_down = b2Vec2(0.0, -0.5 * flyer.diam)
        force_direction = (0.0, -alpha * f_down)
        p1 = flyer.body.GetWorldPoint(localPoint=local_point_down)
        p2 = p1 + flyer.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p1), self._to_screen(p2), line_color)

    @staticmethod
    def _fix_vertices(vertices):
        return [(int(SCREEN_OFFSETX + v[0]), int(SCREEN_OFFSETY - v[1])) for v in vertices]

    def _draw_polygon(self, body, fixture):
        polygon = fixture.shape
        transform = body.transform
        vertices = self._fix_vertices([transform * v * PPM for v in polygon.vertices])
        pygame.draw.polygon(self.screen, [c / 2.0 for c in colors[body.type]], vertices, 0)      # Frame
        pygame.draw.polygon(self.screen, colors[body.type], vertices, 1)                         # Face color

    def _draw_edge(self, body, fixture):
        edge = fixture.shape
        vertices = self._fix_vertices([body.transform * edge.vertex1 * PPM, body.transform * edge.vertex2 * PPM])
        pygame.draw.line(self.screen, colors[body.type], vertices[0], vertices[1])


class SimpleFramework:
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
        """Sets rendering on / off."""
        self.is_render = False if self.is_render else True

    def step(self):
        """Performs single simulation step.

        Updates the world and then the screen.
        """
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._set_render()
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._set_render()
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

        # Step the world
        self.world.Step(TIMESTEP, VEL_ITERS, POS_ITERS)
        self.world.ClearForces()

        if self.is_render:
            self.screen.fill((255, 255, 255))
            self.renderer.render(self.world)
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)
            self.fps = self.clock.get_fps()

        print(f"FPS {self.fps:.0f}", flush=True, end="\r")

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None
