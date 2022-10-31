"""Contains drone definition."""
import copy
import math
import random

import torch
import torch.nn as nn
import numpy as np

from Box2D import b2FixtureDef, b2PolygonShape
from Box2D.Box2D import b2World, b2Vec2, b2Filter

from src.config import Config
from src.body.engine import Engines
from src.body.raycast import RayCastCallback
from src.body.model import NeuralNetwork


class Drone:
    """Drone class.

    A drone consists of a body with four boosters attached.
    """

    _vertices = [
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
    ]

    def __init__(self, world: b2World, config: Config) -> None:
        """Initializes the wheel class."""

        self.world = world
        self.config = config

        self.ray_length = self.config.env.drone.ray_length
        self.max_force = self.config.env.drone.engine.max_force
        self.diam = self.config.env.drone.diam

        fixed_rotation = self.config.env.drone.fixed_rotation
        density = self.config.env.drone.density
        friction = self.config.env.drone.friction

        self.init_position = b2Vec2(
            config.env.drone.init_position.x, config.env.drone.init_position.y
        )
        self.init_linear_velocity = b2Vec2(
            config.env.drone.init_linear_velocity.x,
            config.env.drone.init_linear_velocity.y,
        )
        self.init_angular_velocity = config.env.drone.init_angular_velocity
        self.init_angle = (config.env.drone.init_angle * math.pi) / 180.0

        self.body = world.CreateDynamicBody(
            bullet=False,
            allowSleep=True,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle=self.init_angle,
            fixedRotation=fixed_rotation,
        )

        # Negative groups never collide.
        group_index = -1 if not self.config.env.allow_collision else 0

        vertices = [(self.diam * x, self.diam * y) for (x, y) in self._vertices]
        self.fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices),
            density=density,
            friction=friction,
            filter=b2Filter(groupIndex=group_index),
        )

        self.fixture = self.body.CreateFixture(self.fixture_def)

        self.engines = Engines(body=self.body, config=config)

        self.callback = RayCastCallback

        # Ray casting points
        self.points = [
            b2Vec2(self.ray_length, self.ray_length),
            b2Vec2(-self.ray_length, self.ray_length),
            b2Vec2(-self.ray_length, -self.ray_length),
            b2Vec2(self.ray_length, -self.ray_length),
        ]

        # Domain
        self.domain_diam_x = config.env.domain.limit.x_max - config.env.domain.limit.x_min
        self.domain_diam_y = config.env.domain.limit.y_max - config.env.domain.limit.y_min

        # Odometer
        self.distance = 0.0
        self.position_old = None

        # Obstacle detection data
        self.data = None

        # Neural Network
        self.model = NeuralNetwork(config)
        # Use evaluation model as no gradients for genetic optimization required.
        self.model.eval()   

        # Forces predicted by neural network
        self.forces = [0.0 for _ in range(4)]

        # Ray casting rendering
        self.callbacks = []
        self.p1 = []
        self.p2 = []

    def reset(self, noise: bool = False) -> None:
        """Resets Drone to initial position and velocity."""
        init_position = self.init_position
        init_linear_velocity = self.init_linear_velocity
        init_angular_velocity = self.init_angular_velocity
        init_angle = self.init_angle

        noise_pos_x = self.config.env.drone.noise.position.x
        noise_pos_y = self.config.env.drone.noise.position.y

        x_min = self.config.env.domain.limit.x_min
        x_max = self.config.env.domain.limit.x_max
        y_min = self.config.env.domain.limit.y_min
        y_max = self.config.env.domain.limit.y_max

        pos_x = noise_pos_x * random.uniform(a=x_min, b=x_max)
        pos_y = noise_pos_y * random.uniform(a=y_min, b=y_max)
        init_position = b2Vec2(pos_x, pos_y)

        self.body.position = init_position
        self.body.linearVelocity = init_linear_velocity
        self.body.angularVelocity = init_angular_velocity
        self.body.angle = init_angle

        # Reset variables for odometer
        self.distance = 0.0
        self.position_old = None

    def mutate(self, model: nn.Module) -> None:
        """Mutates Drone model."""

        # Copy weights
        self.model = copy.deepcopy(model)

        # Mutate weights
        self.model.mutate_weights()

    def ray_casting(self):
        """Uses ray casting to measure distane to domain walls."""

        self.callbacks = []
        self.p1 = []
        self.p2 = []
        self.data = []

        for point in self.points:  # for ray in self.rays
            p1 = self.body.position
            p2 = p1 + self.body.GetWorldVector(localVector=point)
            cb = self.callback()
            self.world.RayCast(cb, p1, p2)

            # Store ray casting data for rendering
            self.callbacks.append(copy.copy(cb))
            self.p1.append(copy.copy(p1))
            self.p2.append(copy.copy(p2))

            # Collect data
            if cb.hit:
                # Compute diagonal distance from Drone to wall from raw features.
                self.data.append(
                    ((cb.point.x - p1.x) ** 2 + (cb.point.y - p1.y) ** 2) ** 0.5
                )
            else:
                self.data.append(-1.0)

        # Normalize data
        self.data = (
            torch.tensor(self.data)
            / (self.domain_diam_x**2 + self.domain_diam_y**2) ** 0.5
        )

    def odometer(self) -> float:
        """Measures distance traveled by drone."""
        if self.position_old is None:
            self.position_old = self.body.position
        d = self.position_old - self.body.position
        self.distance += math.sqrt(d.x**2 + d.y**2)
        self.position_old = copy.copy(self.body.position)

    def comp_action(self) -> None:
        """Computes next section of actions applied to engines.
        
        Next steps of action are computed by feeding obstacle data coming
        from ray casting to the drone's neural network which then returns 
        a set of actions (forces) to be applied to the drone's engines.
        """
        forces = self.model(self.data)
        forces = forces.detach().numpy()
        self.forces = self.config.env.drone.engine.max_force * forces.astype(np.float)

    def apply_action(self) -> None:
        """Applies force to Drone coming from neural network.

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
