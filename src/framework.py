"""Minimal pygame-based framework for Box2D.

See also: https://github.com/pybox2d/pybox2d/blob/f8617198555d2e1539b0be82f0ee6e7db4d26085/library/Box2D/examples/backends/opencv_draw.py
for renderer.
"""
import pygame
from pygame.locals import QUIT, KEYDOWN 
from Box2D import b2DrawExtended, b2Vec2, b2Color
from Box2D.b2 import (
    world,
    staticBody,
    dynamicBody,
    kinematicBody,
    polygonShape,
    circleShape,
    edgeShape,
    loopShape,
)


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
viewCenter = b2Vec2(0.5 * SCREEN_WIDTH, 0.5 * SCREEN_HEIGHT)    # TODO: has no effect
viewOffset = b2Vec2(-0.5 * SCREEN_WIDTH, -0.5 * SCREEN_HEIGHT)
screenSize = b2Vec2(SCREEN_WIDTH, SCREEN_HEIGHT)


class PygameDraw(b2DrawExtended):
    """
    This debug draw class accepts callbacks from Box2D (which specifies what to
    draw) and handles all of the rendering.

    If you are writing your own game, you likely will not want to use debug
    drawing.  Debug drawing, as its name implies, is for debugging.
    """
    surface = None
    axisScale = 10.0

    def __init__(self, test=None, **kwargs):
        b2DrawExtended.__init__(self, **kwargs)
        self.flipX = False
        self.flipY = True
        self.convertVertices = False
        self.test = test

        self.zoom = viewZoom
        self.center = viewCenter
        self.offset = viewOffset
        self.screenSize = screenSize

    def DrawPoint(self, p, size, color):
        """
        Draw a single point at point p given a pixel size and color.
        """
        self.DrawCircle(p, size / self.zoom, color, drawwidth=0)

    def DrawCircle(self, center, radius, color, drawwidth=1):
        """
        Draw a wireframe circle given the center, radius, axis of orientation
        and color.
        """
        radius *= self.zoom
        if radius < 1:
            radius = 1
        else:
            radius = int(radius)

        pygame.draw.circle(self.surface, color.bytes, center, radius, drawwidth)

    def DrawSegment(self, p1, p2, color):
        """
        Draw the line segment from p1-p2 with the specified color.
        """
        pygame.draw.aaline(self.surface, color.bytes, p1, p2)

    def DrawPolygon(self, vertices, color):
        """
        Draw a wireframe polygon given the screen vertices with the specified color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color.bytes, vertices[0], vertices)
        else:
            pygame.draw.polygon(self.surface, color.bytes, vertices, 1)

    def DrawSolidPolygon(self, vertices, color):
        """
        Draw a filled polygon given the screen vertices with the specified color.
        """
        if not vertices:
            return

        if len(vertices) == 2:
            pygame.draw.aaline(self.surface, color.bytes, vertices[0], vertices[1])
        else:
            pygame.draw.polygon(self.surface, (color / 2).bytes + [127], vertices, 0)
            pygame.draw.polygon(self.surface, color.bytes, vertices, 1)


class Renderer: # Drawer

    def __init__(self, screen) -> None:
        self.screen = screen
        # Installing drawing methods
        edgeShape.draw = self._draw_edge
        polygonShape.draw = self._draw_polygon

        self.flip_x = False
        self.flip_y = True

        self.view_zoom = None # --> self.ppm or self.pixel_per_meter

        ###################
        self.pd = PygameDraw(surface=self.screen)
        ###################

    def render(self, world):

        # Render bodies
        for body in world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)

        # Render forces
        for flyer in world.flyers:
            pass

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

    def _draw_raycast(self, flyer):
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
        self.renderer_ = Renderer(self.screen)

        self.renderer = PygameDraw(surface=self.screen, test=self)
        # self.world.renderer = self.renderer

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

            # draw_world(self.screen, self.world)
            self.renderer_.render(self.world)

            pygame.display.flip()

            self.clock.tick(TARGET_FPS)
            self.fps = self.clock.get_fps()

            self.screen.fill((255, 255, 255))


        print(f"FPS {self.fps:.0f}", flush=True, end="\r")

        self.world.contactListener = None
        self.world.destructionListener = None
        self.world.renderer = None
