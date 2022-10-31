"""Domain class.

The domain defines the physical space the drone can interact with.
"""
from Box2D import b2EdgeShape
from Box2D.Box2D import b2World

from src.config import Config


class Domain:
    """Defines domain with obstacles."""

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes the inclined plane."""
        self.config = config.env.domain

        self.x_min = self.config.limit.x_min
        self.x_max = self.config.limit.x_max
        self.y_min = self.config.limit.y_min
        self.y_max = self.config.limit.y_max

        shapes = []

        shapes += self._get_domain_boundary()

        map_ = self.config.map
        if map_ == "cross":
            shapes += self._get_map_cross()
        elif map_ == "dead_end":
            raise NotImplementedError(f"Map '{map_}' not implemented.")
        elif map_ == "maze":
            raise NotImplementedError(f"Map '{map_}' not implemented.")
        else:
            raise NotImplementedError(f"Map '{map_}' is not a valid map.")

        world.CreateStaticBody(shapes=shapes)

    def _get_domain_boundary(self) -> list:
        """Creates the domain boundary."""
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        domain_boundary = [
            b2EdgeShape(vertices=[(x_max, y_max), (x_min, y_max)]),
            b2EdgeShape(vertices=[(x_min, y_max), (x_min, y_min)]),
            b2EdgeShape(vertices=[(x_min, y_min), (x_max, y_min)]),
            b2EdgeShape(vertices=[(x_max, y_min), (x_max, y_max)]),
        ]

        return domain_boundary

    def _get_map_cross(self) -> list:
        """Creates a horizontal bar."""

        fraction = 0.75

        x_min = self.x_min
        x_max = self.x_max
        y_max = self.y_max

        x_0 = 0.5 * (x_min + x_max)
        y_0 = y_max
        x_1 = x_0
        y_1 = fraction * y_max

        obstacle = [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return obstacle
