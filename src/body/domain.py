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
        if map_ == "empty":
            pass
        elif map_ == "block":
            shapes += self._get_map_block()
        elif map_ == "cross":
            shapes += self._get_map_cross()
        elif map_ == "track":
            shapes += self._get_map_track()
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

    def _get_map_block(self, fraction: float = 0.5) -> list:
        """Creates a block in the center of the domain.

        Args:
            fraction: Scalar defining length of map elements.
        """
        x_min = self.x_min
        x_max = self.x_max
        y_min = self.y_min
        y_max = self.y_max

        boundary = []

        x_0 = fraction * x_max
        y_0 = fraction * y_max
        x_1 = fraction * x_min
        y_1 = fraction * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = fraction * x_min
        y_0 = fraction * y_max
        x_1 = fraction * x_min
        y_1 = fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = fraction * x_min
        y_0 = fraction * y_min
        x_1 = fraction * x_max
        y_1 = fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = fraction * x_max
        y_0 = fraction * y_min
        x_1 = fraction * x_max
        y_1 = fraction * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary

    def _get_map_cross(self, fraction: float = 0.5) -> list:
        """Creates a cross in the center of the domain.

        Args:
            fraction: Scalar defining length of map elements.
        """
        x_min = self.x_min
        x_max = self.x_max
        y_min = self.y_min
        y_max = self.y_max

        boundary = []

        x_0 = fraction * x_min
        y_0 = 0.0
        x_1 = fraction * x_max
        y_1 = 0.0
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = 0.0
        y_0 = fraction * y_min
        x_1 = 0.0
        y_1 = fraction * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = x_max
        y_0 = y_max
        x_1 = (1.0 - fraction) * x_max
        y_1 = (1.0 - fraction) * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = x_min
        y_0 = y_max
        x_1 = (1.0 - fraction) * x_min
        y_1 = (1.0 - fraction) * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = x_min
        y_0 = y_min
        x_1 = (1.0 - fraction) * x_min
        y_1 = (1.0 - fraction) * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = x_max
        y_0 = y_min
        x_1 = (1.0 - fraction) * x_max
        y_1 = (1.0 - fraction) * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary

    def _get_map_track(self, fraction: float = 0.5) -> list:
        """Creates a simple track.

        Args:
            fraction: Scalar defining length of map elements.
        """
        x_min = self.x_min
        x_max = self.x_max
        y_min = self.y_min
        y_max = self.y_max

        boundary = []

        x_0 = 0.0
        y_0 = y_max
        x_1 = 0.0
        y_1 = (1.0 - fraction) * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = 0.0
        y_0 = y_min
        x_1 = 0.0
        y_1 = (1.0 - fraction) * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = fraction * x_min
        y_0 = 0.0
        x_1 = fraction * x_max
        y_1 = 0.0
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = fraction * x_min
        y_0 = fraction * y_max
        x_1 = fraction * x_min
        y_1 = fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0 = fraction * x_max
        y_0 = fraction * y_max
        x_1 = fraction * x_max
        y_1 = fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary
