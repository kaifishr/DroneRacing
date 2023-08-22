"""The drone's engine.

TODO: Move apply_action / force here?
"""
import math

from Box2D import b2FixtureDef, b2PolygonShape
from Box2D.Box2D import b2Vec2, b2Filter, b2Body

from src.utils.config import Config


class Engines:
    """Engine class.

    Class adds engines to the drone.

    Attributes:
        body: The drone's body object.
        config:
        drone:
        density:
        friction:
    """

    # Engine design parameters
    height = 0.4
    width_min = 0.1
    width_max = 0.3

    def __init__(self, body: b2Body, config: Config) -> None:
        """Initializes engines class."""
        self.body = body
        self.config = config

        self._add_engines()

    def _engine_nozzle(self, mount_point: b2Vec2, theta: float):
        """Adds three engines to booster."""
        height = self.height
        width_min = self.width_min
        width_max = self.width_max

        def rotate(x: float, y: float, theta: float) -> b2Vec2:
            theta = theta * math.pi / 180.0
            x_ = x * math.cos(theta) - y * math.sin(theta)
            y_ = x * math.sin(theta) + y * math.cos(theta)
            return x_, y_

        r_0 = mount_point + rotate(-0.5 * width_min, 0.0, theta)
        r_1 = mount_point + rotate(-0.5 * width_max, height, theta)
        r_2 = mount_point + rotate(0.5 * width_max, height, theta)
        r_3 = mount_point + rotate(0.5 * width_min, 0.0, theta)

        return r_0, r_1, r_2, r_3

    def _add_engines(self) -> None:
        """Adds engines to booster."""

        density = self.config.env.drone.engine.density
        friction = self.config.env.drone.engine.friction
        diam = self.config.env.drone.diam

        mount_points = [
            b2Vec2(0.0, 0.5 * diam),  # top
            b2Vec2(-0.5 * diam, 0.0),  # left
            b2Vec2(0.0, -0.5 * diam),  # bottom
            b2Vec2(0.5 * diam, 0.0),  # right
        ]

        for mount_point, theta in zip(mount_points, [0.0, 90.0, 180.0, 270.0]):
            engine_vertices = self._engine_nozzle(mount_point=mount_point, theta=theta)
            engine_polygon = b2PolygonShape(vertices=engine_vertices)

            engine_fixture_def = b2FixtureDef(
                shape=engine_polygon,
                density=density,
                friction=friction,
                filter=b2Filter(groupIndex=-1),
            )

            self.body.CreateFixture(engine_fixture_def)
