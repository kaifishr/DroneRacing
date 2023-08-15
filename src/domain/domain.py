"""Domain class.

The domain defines the physical space the drone can interact with.
"""
import random

from Box2D import b2EdgeShape
from Box2D import b2FixtureDef
from Box2D import b2PolygonShape
from Box2D.Box2D import b2Filter
from Box2D.Box2D import b2Vec2
from Box2D.Box2D import b2World

from src.utils.config import Config


class Target:
    """Target base class."""

    def __init__(self, world: b2World) -> None:
        """Initializes Target base class."""
        self.world = world


class StaticTarget(Target):
    """Static target class.

    Static targets remain at position during one episode.
    """

    vertices = [
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
    ]

    def __init__(self, world: b2World, config: Config) -> None:
        super().__init__(world=world)

        cfg = config.env.domain
        self.x_min = cfg.limit.x_min
        self.x_max = cfg.limit.x_max
        self.y_min = cfg.limit.y_min
        self.y_max = cfg.limit.y_max

        self.x_diam = self.x_max - self.x_min
        self.y_diam = self.y_max - self.y_min

        # world.CreateStaticBody(shapes=shapes)

        self.body = world.CreateDynamicBody(
            bullet=False,
            allowSleep=False,
            position=b2Vec2(0, 0),
            linearVelocity=b2Vec2(0, 0),
            angularVelocity=0,
            angle=0,
            fixedRotation=False,
        )

        fraction = 0.5
        vertices = [(fraction * x, fraction * y) for (x, y) in self.vertices]

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices),
            density=config.env.drone.density,
            friction=config.env.drone.friction,
            filter=b2Filter(
                groupIndex=-1
            ),  # group_index: -1, 0 (negative groups never collide)
        )
        self.body.CreateFixture(fixture_def)

    def _get_target(self, fraction: float = 0.05) -> list[b2EdgeShape]:
        """Creates a small block as target.

        Args:
            fraction: Scalar defining size of target in map.

        Returns:
            List of edge shapes.
        """
        x_pos = random.uniform(a=self.x_min, b=self.x_max)
        y_pos = random.uniform(a=self.y_min, b=self.y_max)

        x0, y0 = fraction * x_pos, fraction * y_pos
        x1, y1 = x0 + 1, y0 + 1

        shapes = [b2EdgeShape(vertices=[(x0, y0), (x1, y1)])]

        return shapes


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
        elif map_ == "locks":
            shapes += self._get_map_locks()
        elif map_ == "track":
            shapes += self._get_map_track()
        elif map_ == "blade":
            shapes += self._get_map_blade()
        elif map_ == "smile":
            shapes += self._get_map_smile()
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

    def _get_map_block(self, fraction: float = 0.2) -> list:
        """Creates a block in the center of the domain.

        Args:
            fraction: Scalar defining length of map elements.
        """
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        boundary = []

        x_0, y_0 = fraction * x_max, fraction * y_max
        x_1, y_1 = fraction * x_min, fraction * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, fraction * y_max
        x_1, y_1 = fraction * x_min, fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, fraction * y_min
        x_1, y_1 = fraction * x_max, fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_max, fraction * y_min
        x_1, y_1 = fraction * x_max, fraction * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary

    def _get_map_locks(self, fraction: float = 0.3) -> list:
        """Creates a locks around a block in the center of the domain.

        Args:
            fraction: Scalar defining length of map elements.
        """
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        boundary = []

        boundary += self._get_map_block(fraction=fraction)

        # Top
        x_0, y_0 = 0.0, fraction * y_max
        x_1, y_1 = 0.0, 1.0 / 3.0 * y_max * (1.0 + 2.0 * fraction)
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = 0.0, y_max
        x_1, y_1 = 0.0, 1.0 / 3.0 * y_max * (2.0 + fraction)
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        # Left
        x_0, y_0 = fraction * x_min, 0.0
        x_1, y_1 = 1.0 / 3.0 * x_min * (1.0 + 2.0 * fraction), 0.0
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = x_min, 0.0
        x_1, y_1 = 1.0 / 3.0 * x_min * (2.0 + fraction), 0.0
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        # Bottom
        x_0, y_0 = 0.0, fraction * y_min
        x_1, y_1 = 0.0, 1.0 / 3.0 * y_min * (1.0 + 2.0 * fraction)
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = 0.0, y_min
        x_1, y_1 = 0.0, 1.0 / 3.0 * y_min * (2.0 + fraction)
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        # Right
        x_0, y_0 = fraction * x_max, 0.0
        x_1, y_1 = 1.0 / 3.0 * x_max * (1.0 + 2.0 * fraction), 0.0
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = x_max, 0.0
        x_1, y_1 = 1.0 / 3.0 * x_max * (2.0 + fraction), 0.0
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary

    def _get_map_blade(self, fraction: float = 0.3) -> list:
        """Saw blade map.

        Args:
            fraction: Scalar defining length of map elements.
        """
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        boundary = []

        boundary += self._get_map_block(fraction=fraction)

        x_0, y_0 = fraction * x_max, fraction * y_max
        x_1, y_1 = fraction * x_max, 0.5 * (1.0 + fraction) * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, fraction * y_max
        x_1, y_1 = 0.5 * (1.0 + fraction) * x_min, fraction * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, fraction * y_min
        x_1, y_1 = fraction * x_min, 0.5 * (1.0 + fraction) * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_max, fraction * y_min
        x_1, y_1 = 0.5 * (1.0 + fraction) * x_max, fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = x_max, fraction * y_max
        x_1, y_1 = 0.5 * (1.0 + fraction) * x_max, fraction * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, y_max
        x_1, y_1 = fraction * x_min, 0.5 * (1.0 + fraction) * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = x_min, fraction * y_min
        x_1, y_1 = 0.5 * (1.0 + fraction) * x_min, fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_max, y_min
        x_1, y_1 = fraction * x_max, 0.5 * (1.0 + fraction) * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary

    def _get_map_track(self, fraction: float = 0.5) -> list:
        """Creates a simple track.

        Args:
            fraction: Scalar defining length of map elements.
        """
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        boundary = []

        x_0, y_0 = 0.0, y_max
        x_1, y_1 = 0.0, (1.0 - fraction) * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = 0.0, y_min
        x_1, y_1 = 0.0, (1.0 - fraction) * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, 0.0
        x_1, y_1 = fraction * x_max, 0.0
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, fraction * y_max
        x_1, y_1 = fraction * x_min, fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_max, fraction * y_max
        x_1, y_1 = fraction * x_max, fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary

    def _get_map_smile(self, fraction: float = 0.5) -> list:
        """Smile map.

        Args:
            fraction: Scalar defining length of map elements.
        """
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max

        boundary = []

        x_0, y_0 = 0.0, y_max
        x_1, y_1 = 0.0, y_min * (2.0 * fraction - 1.0)
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, fraction * y_max
        x_1, y_1 = fraction * x_min, fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_min, fraction * y_min
        x_1, y_1 = fraction * x_max, fraction * y_min
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        x_0, y_0 = fraction * x_max, fraction * y_min
        x_1, y_1 = fraction * x_max, fraction * y_max
        boundary += [b2EdgeShape(vertices=[(x_0, y_0), (x_1, y_1)])]

        return boundary
