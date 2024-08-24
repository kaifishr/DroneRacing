"""Renderer for pygame-based framework."""
import pygame

from Box2D import b2Vec2, b2World
from Box2D.b2 import staticBody
from Box2D.b2 import dynamicBody
from Box2D.b2 import kinematicBody
from Box2D.b2 import polygonShape
from Box2D.b2 import circleShape
from Box2D.b2 import edgeShape

from src.drone.drone import Drone
from src.utils.config import Config


class Renderer:
    """Renderer class for Box2D world.

    Attributes:
        config:
        screen

    """

    # Screen background color.
    color_background = (0, 0, 0, 0)

    # World body colors.
    colors = {
        staticBody: (220, 220, 220, 255),
        dynamicBody: (127, 127, 127, 255),
        kinematicBody: (127, 127, 230, 255),
    }

    # Raycast colors.
    color_raycast_line = (128, 0, 128, 255)
    color_raycast_head = (255, 0, 255, 255)

    color_force_vector = (255, 0, 0, 255)
    color_gate = (0, 255, 0, 255)

    # Flip axes.
    flip_x = False
    flip_y = True

    def __init__(self, screen: pygame.Surface, config: Config) -> None:
        """Initializes Renderer."""
        self.screen = screen
        self.config = config

        self.ppm = self.config.renderer.ppm
        screen_width = self.config.framework.screen.width
        screen_height = self.config.framework.screen.height

        self.screen_offset = b2Vec2(-0.5 * screen_width, -0.5 * screen_height)
        self.screen_size = b2Vec2(screen_width, screen_height)

        self._install()

        self.render_rays = True

    def _install(self):
        """Installs drawing methods for world objects."""
        edgeShape.draw = self._draw_edge
        polygonShape.draw = self._draw_polygon
        circleShape.draw = self._draw_circle2

    def set_render_rays(self) -> None:
        """Turns rendering of ray casting lines on or off."""
        if self.render_rays:
            self.render_rays = False
        else:
            self.render_rays = True

    def render(self, world: b2World) -> None:
        """Renders world."""
        self.screen.fill(self.color_background)

        # Render rays.
        if self.render_rays:
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

    def _to_screen(self, point: b2Vec2) -> tuple[int, int]:
        """Transforms point from simulation to screen coordinates.

        Args:
            point: Point to be transformed to pixel coordinates.
        """
        pos_x = point.x * self.ppm - self.screen_offset.x
        pos_y = point.y * self.ppm - self.screen_offset.y

        if self.flip_x:
            pos_x = self.screen_size.x - pos_x
        if self.flip_y:
            pos_y = self.screen_size.y - pos_y

        return int(pos_x), int(pos_y)

    def _draw_point(self, point, size, color):
        """Draws point in specified size and color."""
        self._draw_circle(center=point, radius=size / self.ppm, color=color, width=0)

    def _draw_circle(self, center, radius, color, width=1):
        """Draws circle in specified size and color."""
        radius *= self.ppm
        radius = 1 if radius < 1 else int(radius)
        pygame.draw.circle(self.screen, color, center, radius, width)

    def _draw_circle2(self, body, fixture, color=None, width: int = 1) -> None:
        position = self._to_screen(body.position)
        radius = self.ppm * fixture.shape.radius
        pygame.draw.circle(self.screen, color or self.color_gate, position, radius, width=width)

    def _draw_segment(self, p_1, p_2, color: tuple[int]):
        """Draws line from points p_1 to p_2 in specified color."""
        pygame.draw.aaline(self.screen, color, p_1, p_2)

    def _draw_raycast(self, drone: Drone) -> None:
        """Draws rays"""

        for p_1, p_2, callback in zip(drone.p1, drone.p2, drone.callbacks):
            p_1 = self._to_screen(p_1)
            p_2 = self._to_screen(p_2)
            if callback.hit:
                cb_point = callback.point
                cb_point = self._to_screen(cb_point)
                self._draw_point(cb_point, 3.0, self.color_raycast_head)
                self._draw_segment(p_1, cb_point, self.color_raycast_line)
            else:
                self._draw_segment(p_1, p_2, self.color_raycast_line)

    def _draw_force(self, drone: Drone) -> None:
        """Draws force vectors.

        Purely cosmetic but helps with debugging.
        Arrows point towards direction the force is coming from.
        """
        scale_force = self.config.renderer.scale_force
        color = self.color_force_vector

        f_left, f_right, f_up, f_down = drone.forces

        # Left
        local_point_left = b2Vec2(-0.5 * drone.diam, 0.0)
        force_direction = (-scale_force * f_left, 0.0)
        p_1 = drone.body.GetWorldPoint(localPoint=local_point_left)
        p_2 = p_1 + drone.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p_1), self._to_screen(p_2), color)

        # Right
        local_point_right = b2Vec2(0.5 * drone.diam, 0.0)
        force_direction = (scale_force * f_right, 0.0)
        p_1 = drone.body.GetWorldPoint(localPoint=local_point_right)
        p_2 = p_1 + drone.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p_1), self._to_screen(p_2), color)

        # Up
        local_point_up = b2Vec2(0.0, 0.5 * drone.diam)
        force_direction = (0.0, scale_force * f_up)
        p_1 = drone.body.GetWorldPoint(localPoint=local_point_up)
        p_2 = p_1 + drone.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p_1), self._to_screen(p_2), color)

        # Down
        local_point_down = b2Vec2(0.0, -0.5 * drone.diam)
        force_direction = (0.0, -scale_force * f_down)
        p_1 = drone.body.GetWorldPoint(localPoint=local_point_down)
        p_2 = p_1 + drone.body.GetWorldVector(force_direction)
        self._draw_segment(self._to_screen(p_1), self._to_screen(p_2), color)

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
