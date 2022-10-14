import copy
import math
import random

import torch
import torch.nn as nn
import numpy as np

from Box2D import b2FixtureDef, b2PolygonShape, b2RayCastCallback
from Box2D.Box2D import b2World, b2Vec2, b2Filter, b2Body

from src.config import Config


class RayCastCallback(b2RayCastCallback):
    """Callback detects closest hit.

    See also this example for more information about ray casting in PyBox2D:
    https://github.com/pybox2d/pybox2d/blob/master/library/Box2D/examples/raycast.py
    
    """
    def __init__(self, **kwargs) -> None:
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction) -> float:
        """Reports hit fixture.
        """
        if fixture.filterData.groupIndex == -1: # Ignore engines.
            return 1.0
        self.hit = True
        self.fixture = fixture          # Fixture of the hit body. Interesting for multi-agent environments.
        self.point = b2Vec2(point)      # Point of contact of body.
        self.normal = b2Vec2(normal)    # Normal vector at point of contact. Perpendicular to body surface.
        return fraction


class NeuralNetwork(nn.Module):

    def __init__(self, config: Config) -> None:
        """Initializes NeuralNetwork class."""
        super().__init__()

        self.mutation_prob = config.optimizer.mutation_probability
        self.mutation_rate = config.optimizer.mutation_rate

        cfg = config.env.box.neural_network

        in_features = cfg.n_dim_in
        out_features = cfg.n_dim_out
        hidden_features = cfg.n_dim_hidden

        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.linear2 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.linear3 = nn.Linear(in_features=hidden_features, out_features=out_features)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=5.0/3.0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    @torch.no_grad()
    def _mutate_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            mask = torch.rand_like(module.weight) < self.mutation_prob
            mutation = self.mutation_rate * torch.randn_like(module.weight)
            module.weight.add_(mask * mutation)
            if module.bias is not None:
                mask = torch.rand_like(module.bias) < self.mutation_prob
                mutation = self.mutation_rate * torch.randn_like(module.bias)
                module.bias.add_(mask * mutation)

    def mutate_weights(self) -> None:
        """Mutates the network's weights."""
        self.apply(self._mutate_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=0)
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        return torch.sigmoid(self.linear3(x))


class Flyer:
    """Flyer class.

    A flyer consists of a box with four boosters attached.
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
        friction = 2.0      # TODO: move this to config

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
                    friction=self.friction,
                    filter=b2Filter(groupIndex=-1),  # negative groups never collide
                )

                self.body.CreateFixture(engine_fixture_def)

    def __init__(self, world: b2World, config: Config):
        """Initializes the wheel class."""

        self.world = world
        self.config = config

        self.ray_length = self.config.env.box.ray_length
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
            fixedRotation=True      # TODO: move this to config
        )

        self.vertices = [(self.diam * x, self.diam * y) for (x, y) in self._vertices]
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

        self.callback = RayCastCallback

        # Ray casting points
        # self.points = [
        #     b2Vec2(0, self.ray_length), 
        #     b2Vec2(-self.ray_length, 0), 
        #     b2Vec2(0, -self.ray_length), 
        #     b2Vec2(self.ray_length, 0)
        # ]
        self.points = [
            b2Vec2(self.ray_length, self.ray_length), 
            b2Vec2(-self.ray_length, self.ray_length), 
            b2Vec2(-self.ray_length, -self.ray_length), 
            b2Vec2(self.ray_length, -self.ray_length)
        ]

        # Domain
        self.domain_diam_x = config.env.domain.x_max - config.env.domain.x_min
        self.domain_diam_y = config.env.domain.y_max - config.env.domain.y_min
        
        # Odometer
        self.distance = 0.0
        self.position_old = None

        # Obstacle detection data
        self.data = None

        # Neural Network
        self.model = NeuralNetwork(config)

        # Forces predicted by neural network
        self.forces = [0.0 for _ in range(4)]

    def reset(self, noise: bool = False) -> None:
        """Resets wheel to initial position and velocity."""
        init_position = self.init_position
        init_linear_velocity = self.init_linear_velocity
        init_angular_velocity = self.init_angular_velocity
        init_angle = self.init_angle

        noise = self.config.env.box.noise
        if noise:
            # Position
            noise_x = random.gauss(mu=0.0, sigma=noise.position.x)
            noise_y = random.gauss(mu=0.0, sigma=noise.position.y)
            init_position += (noise_x, noise_y)

            # # Linear velocity
            # noise_x = random.gauss(mu=0.0, sigma=noise.linear_velocity.x)
            # noise_y = random.gauss(mu=0.0, sigma=noise.linear_velocity.y)
            # init_linear_velocity += (noise_x, noise_y)

            # # Angular velocity
            # noise_angular_velocity = random.gauss(mu=0.0, sigma=noise.angular_velocity)
            # init_angular_velocity += noise_angular_velocity

            # # Angle
            # noise_angle = random.gauss(mu=0.0, sigma=noise.angle)
            # init_angle += (noise_angle * math.pi) / 180.0

        self.body.position = init_position
        self.body.linearVelocity = init_linear_velocity
        self.body.angularVelocity = init_angular_velocity
        self.body.angle = init_angle

        # Reset variables for odometer
        self.distance = 0.0
        self.position_old = None

    def mutate(self, model: nn.Module) -> None:
        """Mutates flyers model."""

        # Copy weights
        self.model = copy.deepcopy(model)

        # Mutate weights
        self.model.mutate_weights()

    def ray_casting(self):
        """"""
        cb_ = []
        p1_ = []
        p2_ = []

        self.data = []

        for point in self.points:       # for ray in self.rays
            p1 = self.body.position
            p2 = p1 + self.body.GetWorldVector(localVector=point)
            cb = self.callback()
            self.world.RayCast(cb, p1, p2)
            cb_.append(copy.copy(cb))
            p1_.append(copy.copy(p1))
            p2_.append(copy.copy(p2))

            # Collect data
            if cb.hit:
                # self.data.append((cb.point.x, cb.point.y))
                # Compute diagonal distance from Flyer to wall from raw features.
                self.data.append(((cb.point.x - p1.x)**2 + (cb.point.y - p1.y)**2)**0.5)
            else:
                # self.data.append((-1.0, -1.0))
                self.data.append(-1.0)

        # Normalize data
        self.data = torch.tensor(self.data) / (self.domain_diam_x**2 + self.domain_diam_y**2)**0.5

        # Return data for rendering
        return cb_, p1_, p2_

    def odometer(self) -> float:
        """Measures distance traveled by flyer."""
        if self.position_old is None:
            self.position_old = self.body.position
        d = self.position_old - self.body.position
        self.distance += math.sqrt(d.x**2 + d.y**2)
        self.position_old = copy.copy(self.body.position)

    def comp_action(self) -> None:
        """Computes next section of actions applied to engines."""
        forces = self.model(self.data)
        forces = forces.detach().numpy()
        self.forces = self.config.env.box.engine.max_force * forces.astype(np.float)

    def apply_action(self) -> None:
            """Applies force to Flyer coming from neural network.

            Each engine is controlled individually.

            """
            # self.forces = [random.uniform(0, 1) * self.max_force for _ in range(4)]  # some random data

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