from Box2D import b2EdgeShape
from Box2D.Box2D import b2World

from src.config import Config


class Domain:
    """Defines domain with obstacles for Flyer."""

    def __init__(self, world: b2World, config: Config):
        """Initializes the inclined plane."""
        cfg = config.env.domain
        x_min, x_max = cfg.x_min, cfg.x_max
        y_min, y_max = cfg.y_min, cfg.y_max

        walls = [
            b2EdgeShape(vertices=[(x_max, y_max), (x_min, y_max)]),
            b2EdgeShape(vertices=[(x_min, y_max), (x_min, y_min)]),
            b2EdgeShape(vertices=[(x_min, y_min), (x_max, y_min)]),
            b2EdgeShape(vertices=[(x_max, y_min), (x_max, y_max)]),
        ]

        obstacle = [
            b2EdgeShape(vertices=[(4, 4), (6, 4)]),
            b2EdgeShape(vertices=[(6, 4), (6, 6)]),
            b2EdgeShape(vertices=[(6, 6), (4, 6)]),
            b2EdgeShape(vertices=[(4, 6), (4, 4)]),
        ]

        world.CreateStaticBody(
            shapes=[
                *walls,
                # *obstacle
            ]
        )
