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

        # Initialize drone body
        rx_init = config.env.drone.init_position.x
        ry_init = config.env.drone.init_position.y
        self.init_position = b2Vec2(rx_init, ry_init)

        vx_init = config.env.drone.init_linear_velocity.x
        vy_init = config.env.drone.init_linear_velocity.y
        self.init_linear_velocity = b2Vec2(vx_init, vy_init)

        self.init_angular_velocity = config.env.drone.init_angular_velocity
        self.init_angle = (config.env.drone.init_angle * math.pi) / 180.0
        fixed_rotation = config.env.drone.fixed_rotation

        self.body = world.CreateDynamicBody(
            bullet=False,
            allowSleep=False,
            position=self.init_position,
            linearVelocity=self.init_linear_velocity,
            angularVelocity=self.init_angular_velocity,
            angle=self.init_angle,
            fixedRotation=fixed_rotation,
        )

        self.diam = config.env.drone.diam
        vertices = [(self.diam * x, self.diam * y) for (x, y) in self._vertices]

        # Negative groups never collide.
        if not config.env.allow_collision:
            group_index = -1
        else:
            group_index = 0

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices),
            density=config.env.drone.density,
            friction=config.env.drone.friction,
            filter=b2Filter(groupIndex=group_index),
        )

        self.fixture = self.body.CreateFixture(fixture_def)

        # Engines   TODO: move this to Engine class?
        self.engines = Engines(body=self.body, config=config)
        self.max_force = config.env.drone.engine.max_force

        # Define direction in which we look for obstacles
        ray_length = config.env.drone.ray_length
        self.points = [
            b2Vec2(ray_length, ray_length),
            b2Vec2(-ray_length, ray_length),
            b2Vec2(-ray_length, -ray_length),
            b2Vec2(ray_length, -ray_length),
        ]

        # Neural Network
        self.model = NeuralNetwork(config)
        self.model.eval()   # No gradients for genetic optimization required.

        # Domain 
        self.x_max = config.env.domain.limit.x_max
        self.x_min = config.env.domain.limit.x_min
        self.y_max = config.env.domain.limit.y_max
        self.y_min = config.env.domain.limit.y_min

        # Compute normalization parameter
        domain_diam_x = self.x_max - self.x_min
        domain_diam_y = self.y_max - self.y_min
        self.normalizer = 1.0 / (domain_diam_x ** 2 + domain_diam_y ** 2) ** 0.5

        # Odometer
        self.distance = 0.0
        self.position_old = None

        # Forces predicted by neural network.
        # Initialized with 0 for each engine.
        self.forces = [0.0 for _ in range(4)]

        # Ray casting   TODO: move this to Raycast class?
        self.callback = RayCastCallback
        # Ray casting rendering
        self.callbacks = []
        self.p1 = []
        self.p2 = []

        # Input data for neural network 
        self.data = None

    def reset(self) -> None:
        """Resets Drone to initial position and velocity."""

        if self.config.env.drone.noise.add_noise:
            # Add noise to position
            noise_pos_x = self.config.env.drone.noise.position.x
            noise_pos_y = self.config.env.drone.noise.position.y
            pos_x = noise_pos_x * random.uniform(a=self.x_min, b=self.x_max)
            pos_y = noise_pos_y * random.uniform(a=self.y_min, b=self.y_max)
            init_position = b2Vec2(pos_x, pos_y)
        else:
            init_position = self.init_position

        self.body.position = init_position
        self.body.linearVelocity = self.init_linear_velocity
        self.body.angularVelocity = self.init_angular_velocity
        self.body.angle = self.init_angle

        # Reset variables for odometer
        self.distance = 0.0
        self.position_old = None

    def mutate(self, model: nn.Module) -> None:
        """Mutates drone's neural network.
        
        Attr:
            model: The current best model.
        """
        self.model = copy.deepcopy(model)
        self.model.mutate_weights()

    def odometer(self) -> float:
        """Computes distance traveled by drone."""
        if self.position_old is None:
            self.position_old = self.body.position
        diff = self.position_old - self.body.position
        self.distance += (diff.x ** 2 + diff.y ** 2) ** 0.5
        self.position_old = copy.copy(self.body.position)

    def ray_casting(self):
        """Uses ray casting to measure distane to domain walls."""

        self.callbacks = []
        self.p1 = []
        self.p2 = []
        self.data = []

        p1 = self.body.position

        for point in self.points: 

            # Perform ray casting from drone position p1 to to point p2.
            p2 = p1 + self.body.GetWorldVector(localVector=point)
            cb = self.callback()
            self.world.RayCast(cb, p1, p2)

            # Save ray casting data for rendering.
            self.p1.append(p1)
            self.p2.append(p2)
            self.callbacks.append(cb)

            # Gather data
            if cb.hit:
                # When the ray has hit something compute distance
                # from drone to obstacle from raw features.
                diff = cb.point - p1
                dist = (diff.x ** 2 + diff.y ** 2) ** 0.5
                self.data.append(dist)
            else:
                self.data.append(-1.0)

        # Normalize data
        self.data = self.normalizer * torch.tensor(self.data)

    def comp_action(self) -> None:
        """Computes next section of actions applied to engines.

        Next steps of action are computed by feeding obstacle data coming
        from ray casting to the drone's neural network which then returns
        a set of actions (forces) to be applied to the drone's engines.
        """
        forces = self.model(self.data)
        forces = forces.detach().numpy().astype(np.float)
        self.forces = self.max_force * forces

    def apply_action(self) -> None:
        """Applies force to Drone coming from neural network.

        Each engine is controlled individually.

        """
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
