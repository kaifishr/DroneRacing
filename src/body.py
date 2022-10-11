import math
import random

from Box2D import b2EdgeShape, b2FixtureDef, b2PolygonShape
from Box2D.Box2D import b2World, b2Vec2, b2Filter, b2Body

from src.config import Config


class Domain:
    """Defines domain for box to reside.

    """

    def __init__(self, world: b2World, config: Config):
        """Initializes the inclined plane."""
        cfg = config.env.domain
        x_min, x_max = cfg.x_min, cfg.x_max
        y_min, y_max = cfg.y_min, cfg.y_max

        world.CreateStaticBody(
            shapes=[
                b2EdgeShape(vertices=[(x_max, y_max), (x_min, y_max)]),
                b2EdgeShape(vertices=[(x_min, y_max), (x_min, y_min)]),
                b2EdgeShape(vertices=[(x_min, y_min), (x_max, y_min)]),
                b2EdgeShape(vertices=[(x_max, y_min), (x_max, y_max)]),
            ]
        )


class BoosterBox:
    """BoosterBox class.

    Creates a box with four boosters.
    """

    _vertices = [
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
    ]

    class Engines:
        """Engine class.
        """

        # Parameters for engine
        density = 1.0
        height = 0.4
        width_min = 0.1
        width_max = 0.3

        def __init__(self, body: b2Body, box) -> None:
            """Initializes engines class."""
            self.body = body
            self.box = box
            self._add_engines()

        def _add_engines(self) -> None:
            """Adds engines to booster."""

            def engine_nozzle(mount_point: b2Vec2, theta: float):
                """Adds three engines to booster."""

                def rotate(x:float , y: float, theta: float) -> b2Vec2:
                    theta = theta * math.pi / 180.0
                    return x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)

                r_0 = mount_point + rotate(-0.5 * self.width_min, 0.0, theta)
                r_1 = mount_point + rotate(-0.5 * self.width_max, self.height, theta)
                r_2 = mount_point + rotate(0.5 * self.width_max, self.height, theta)
                r_3 = mount_point + rotate(0.5 * self.width_min, 0.0, theta)

                return r_0, r_1, r_2, r_3

            mount_points = [
                b2Vec2(0.0, 0.5 * self.box.diam),   # up
                b2Vec2(-0.5 * self.box.diam, 0.0),  # left
                b2Vec2(0.0, -0.5 * self.box.diam),  # down
                b2Vec2(0.5 * self.box.diam, 0.0),   # right
            ]

            for mount_point, theta in zip(mount_points, [0.0, 90.0, 180.0, 270.0]):

                engine_polygon = b2PolygonShape(
                    vertices=engine_nozzle(mount_point=mount_point, theta=theta)
                )

                engine_fixture_def = b2FixtureDef(
                    shape=engine_polygon, 
                    density=self.density,
                    filter=b2Filter(groupIndex=-1),  # negative groups never collide
                )

                self.body.CreateFixture(engine_fixture_def)

    def __init__(self, world: b2World, config: Config):
        """Initializes the wheel class."""

        self.config = config
        self.max_force = self.config.env.box.engine.max_force
        self.diam = self.config.env.box.diam
        self.density = self.config.env.box.density
        self.friction = self.config.env.box.friction

        self.init_position = b2Vec2(
            config.env.box.init_position.x, config.env.box.init_position.y
        )
        self.init_linear_velocity = b2Vec2(
            config.env.box.init_linear_velocity.x,
            config.env.box.init_linear_velocity.y,
        )
        self.init_angular_velocity = config.env.box.init_angular_velocity
        self.init_angle = (config.env.box.init_angle * math.pi) / 180.0

        self.body = world.CreateDynamicBody(
            bullet=False,
            allowSleep=True,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle=self.init_angle,
        )

        self.vertices = self.get_vertices()
        self.fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=self.vertices),
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1),
        )

        self.fixture = self.body.CreateFixture(self.fixture_def)

        self.engines = self.Engines(
            body=self.body,
            box=self
        )
        
        self.force = None

    def get_vertices(self) -> list:
        """Creates base vertices for wheel."""
        return [(self.diam * x, self.diam * y) for (x, y) in self._vertices]

    def reset(self, noise: bool = False) -> None:
        """Resets wheel to initial position and velocity."""
        init_position = self.init_position
        init_linear_velocity = self.init_linear_velocity
        init_angular_velocity = self.init_angular_velocity
        init_angle = self.init_angle

        # noise = self.config.env.wheel.noise
        # if noise:
        #     # Position
        #     noise_x = random.gauss(mu=0.0, sigma=noise.position.x)
        #     noise_y = random.gauss(mu=0.0, sigma=noise.position.y)
        #     init_position += (noise_x, noise_y)

        #     # Linear velocity
        #     noise_x = random.gauss(mu=0.0, sigma=noise.linear_velocity.x)
        #     noise_y = random.gauss(mu=0.0, sigma=noise.linear_velocity.y)
        #     init_linear_velocity += (noise_x, noise_y)

        #     # Angular velocity
        #     noise_angular_velocity = random.gauss(mu=0.0, sigma=noise.angular_velocity)
        #     init_angular_velocity += noise_angular_velocity

        #     # Angle
        #     noise_angle = random.gauss(mu=0.0, sigma=noise.angle)
        #     init_angle += (noise_angle * math.pi) / 180.0

        self.body.position = init_position
        self.body.linearVelocity = init_linear_velocity
        self.body.angularVelocity = init_angular_velocity
        self.body.angle = init_angle

    def mutate(self, vertices: list) -> None:
        """Mutates wheel's vertices.

        TODO: Move to optimizer class.
        """
        self.body.DestroyFixture(self.fixture)

        p = self.config.optimizer.mutation_probability
        rho = self.config.optimizer.mutation_rate

        def _mutate(x: float) -> float:
            return x + (random.random() < p) * random.gauss(0, rho)

        def _clip(x: float) -> float:
            return min(x, 0.5 * self.diam)

        # Mutate vertices
        self.vertices = [(_mutate(x), _mutate(y)) for (x, y) in vertices]

        # Keep wheels from getting too big
        self.vertices = [(_clip(x), _clip(y)) for (x, y) in self.vertices]

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=self.vertices),
            density=self.density,
            friction=self.friction,
            filter=b2Filter(groupIndex=-1),
        )
        self.fixture = self.body.CreateFixture(fixture_def)

    def apply_action(self, force_from_nn: tuple = (0.0, 0.0, 0.0, 0.0)) -> None:
            """Applies force to BoosterBox coming from neural network.

            Pretty verbose, however, b2Vec2 does not offer much functionality.

            Each engine is controlled individually.

            Args:
                apply_force: Tuple of force predicted by network and to be applied to BoosterBox.

            Shorter but less readable solution:
                position = [b2Vec2(1, 0), b2Vec2(-1, 0), b2Vec2(0, -1), b2Vec2(0, 1)]

                for force, pos in zip(self.forces, position):
                    f = self.body.GetWorldVector(localVector=force * pos)
                    p = self.body.GetWorldPoint(localPoint=-0.5 * self.diam * pos)    
                    self.body.ApplyForce(f, p, True)
            """
            self.forces = [random.uniform(0, 1) * self.max_force for _ in range(4)]  # some random data

            f_left, f_right, f_up, f_down = self.forces

            # Left
            f = self.body.GetWorldVector(localVector=b2Vec2(f_left, 0.0))
            p = self.body.GetWorldPoint(localPoint=b2Vec2(-0.5 * self.diam, 0.0))    
            self.body.ApplyForce(f, p, True)

            # Right
            f = self.body.GetWorldVector(localVector=b2Vec2(-f_right, 0.0))
            p = self.body.GetWorldPoint(localPoint=b2Vec2(0.5 * self.diam, 0.0))    
            self.body.ApplyForce(f, p, True)

            # Up
            f = self.body.GetWorldVector(localVector=b2Vec2(0.0, -f_up))
            p = self.body.GetWorldPoint(localPoint=b2Vec2(0.0, 0.5 * self.diam))    
            self.body.ApplyForce(f, p, True)

            # Down
            f = self.body.GetWorldVector(localVector=b2Vec2(0.0, f_down))
            p = self.body.GetWorldPoint(localPoint=b2Vec2(0.0, 0.5 * self.diam))    
            self.body.ApplyForce(f, p, True)