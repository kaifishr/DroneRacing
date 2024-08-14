import random

from Box2D import b2EdgeShape
from Box2D import b2FixtureDef
from Box2D import b2PolygonShape
from Box2D.Box2D import b2Filter
from Box2D.Box2D import b2Vec2
from Box2D.Box2D import b2World

from src.utils.config import Config


class Snitch:
    """Snitch class.

    The Snitch acts as a target.
    """

    vertices = [
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
    ]

    def __init__(self, world: b2World, config: Config) -> None:
        self.world = world
        self.config = config

        cfg = config.env.domain
        self.x_min = cfg.limit.x_min
        self.x_max = cfg.limit.x_max
        self.y_min = cfg.limit.y_min
        self.y_max = cfg.limit.y_max

        self.x_diam = self.x_max - self.x_min
        self.y_diam = self.y_max - self.y_min

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
            filter=b2Filter(groupIndex=-1),  # Negative groups never collide.
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

    def step(self) -> None:
        """Moves target to next random position if one drone comes close enough.
        Allows for incremental learning. 
        """
        for drone in self.world.drones:
            dist = (self.body.position - drone.body.position).length
            if dist < self.config.env.snitch.distance_threshold:
                x_pos = random.uniform(self.x_min, self.x_max)
                y_pos = random.uniform(self.y_min, self.y_max)
                self.body.position = b2Vec2(x_pos, y_pos)
                break
