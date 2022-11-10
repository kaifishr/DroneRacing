"""Renderer for pygame-based framework."""
import pygame

from Box2D import b2Vec2, b2World
from Box2D.b2 import (
    staticBody,
    dynamicBody,
    kinematicBody,
    polygonShape,
    edgeShape,
)

from src.drone.drone import Drone
from src.utils.config import Config


class Renderer:
    """Renderer class for Box2D world.

    Attributes:
        config:
        screen

    """

    # Screen background color
    color_background = (0, 0, 0, 0)

    # World body colors
    colors = {
        staticBody: (220, 220, 220, 255),
        dynamicBody: (127, 127, 127, 255),
        kinematicBody: (127, 127, 230, 255),
    }

    # Raycast colors
    color_raycast_line = (128, 0, 128, 255)
    color_raycast_head = (255, 0, 255, 255)

    # Force vector color
    color_force_line = (255, 0, 0, 255)

    def __init__(self, screen: pygame.Surface, config: Config) -> None:
        """Initializes Renderer."""
        self.screen = screen
        self.config = config

        self.ppm = self.config.renderer.ppm
        screen_width = self.config.framework.screen.width
        screen_height = self.config.framework.screen.height

        self.screen_offset = b2Vec2(-0.5 * screen_width, -0.5 * screen_height)
        self.screen_size = b2Vec2(screen_width, screen_height)

        self.flip_x = False
        self.flip_y = True

        self._install()

    def _install(self):
        """Installs drawing methods for world objects."""
        edgeShape.draw = self._draw_edge
        polygonShape.draw = self._draw_polygon

    def render(self, world: b2World) -> None:
        """Renders world."""
        self.screen.fill(self.color_background)  # TODO: move this to renderer?

        # Render rays.
        for drone in world.drones:
            self._draw_raycast(drone)

        # Render force vectors.
        for drone in world.drones:
            self._draw_force(drone)

        # Render bodies.
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)

    def _transform_vertices(self, vertices: tuple):
        """Transforms points of vertices to pixel coordinates."""
        return [self._to_screen(vertex) for vertex in vertices]

    def _to_screen(self, point: b2Vec2) -> tuple:
        """Transforms point from simulation to screen coordinates.

        Args:
            point: Point to be transformed to pixel coordinates.
        """
        x = point.x * self.ppm - self.screen_offset.x
        y = point.y * self.ppm - self.screen_offset.y

        if self.flip_x:
            x = self.screen_size.x - x
        if self.flip_y:
            y = self.screen_size.y - y

        return (int(x), int(y))

    def _draw_point(self, point, size, color):
        """Draws point in specified size and color."""
        self._draw_circle(point, size / self.ppm, color, width=0)

    def _draw_circle(self, center, radius, color, width=1):
        """Draws circle in specified size and color."""
        radius *= self.ppm
        radius = 1 if radius < 1 else int(radius)
        pygame.draw.circle(self.screen, color, center, radius, width)

    def _draw_segment(self, p1, p2, color):
        """Draws line from points p1 to p2 in specified color."""
        pygame.draw.aaline(self.screen, color, p1, p2)

    def _draw_raycast(self, drone: Drone) -> None:
        """Draws rays"""

        for p1, p2, callback in zip(drone.p1, drone.p2, drone.callbacks):
            p1 = self._to_screen(p1)
            p2 = self._to_screen(p2)
            if callback.hit:
                cb_point = callback.point
                cb_point = self._to_screen(cb_point)
                self._draw_point(cb_point, 3.0, self.color_raycast_head)
                self._draw_segment(p1, cb_point, self.color_raycast_line)
            else:
                self._draw_segment(p1, p2, self.color_raycast_line)

    def _draw_force(self, drone: Drone) -> None:
        """Draws force vectors.

        Purely cosmetic but helps with debugging.
        Arrows point towards direction the force is coming from.
        """
        scale_force = self.config.renderer.scale_force
        color = self.color_force_line

        f_left, f_right, f_up, f_down = drone.forces

        # Left
        local_point_left = b2Vec2(-0.5 * drone.diam, 0.0)
        force_direction = (-scale_force * f_left, 0.0)
        p1 = drone.body.GetWorldPoint(localPoint=local_point_left)
        p2 = p1 + drone.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p1), self._to_screen(p2), color)

        # Right
        local_point_right = b2Vec2(0.5 * drone.diam, 0.0)
        force_direction = (scale_force * f_right, 0.0)
        p1 = drone.body.GetWorldPoint(localPoint=local_point_right)
        p2 = p1 + drone.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p1), self._to_screen(p2), color)

        # Up
        local_point_up = b2Vec2(0.0, 0.5 * drone.diam)
        force_direction = (0.0, scale_force * f_up)
        p1 = drone.body.GetWorldPoint(localPoint=local_point_up)
        p2 = p1 + drone.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p1), self._to_screen(p2), color)

        # Down
        local_point_down = b2Vec2(0.0, -0.5 * drone.diam)
        force_direction = (0.0, -scale_force * f_down)
        p1 = drone.body.GetWorldPoint(localPoint=local_point_down)
        p2 = p1 + drone.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p1), self._to_screen(p2), color)

    def _draw_polygon(self, body, fixture):
        """Draws polygon to screen."""
        polygon = fixture.shape
        transform = body.transform
        vertices = [transform * vertex for vertex in polygon.vertices]
        vertices = self._transform_vertices(vertices)
        edge_color = [0.5 * c for c in self.colors[body.type]]
        pygame.draw.polygon(self.screen, edge_color, vertices, 0)  # edge
        pygame.draw.polygon(self.screen, self.colors[body.type], vertices, 1)  # face

    def _draw_edge(self, body, fixture):
        """Draws edge to screen."""
        edge = fixture.shape
        vertices = [body.transform * edge.vertex1, body.transform * edge.vertex2]
        vertex1, vertex2 = self._transform_vertices(vertices)
        pygame.draw.line(self.screen, self.colors[body.type], vertex1, vertex2)
