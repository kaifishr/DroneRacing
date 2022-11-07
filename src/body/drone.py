"""Contains drone definition."""
import copy
import math

import torch.nn as nn

from Box2D import b2FixtureDef, b2PolygonShape
from Box2D.Box2D import b2World, b2Vec2, b2Filter

from src.utils.config import Config
from src.body.engine import Engines
from src.body.raycast import RayCastCallback
from src.body.model import NetworkLoader


class Drone:
    """Drone class.

    A drone consists of a body with four boosters attached.
    """

    vertices = [
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
    ]

    num_engines = 4

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
        vertices = [(self.diam * x, self.diam * y) for (x, y) in self.vertices]

        # Negative groups never collide.
        if not config.env.allow_collision_drones:
            group_index = -1
        else:
            group_index = 0

        fixture_def = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices),
            density=config.env.drone.density,
            friction=config.env.drone.friction,
            filter=b2Filter(groupIndex=group_index),
        )
        self.body.CreateFixture(fixture_def)

        # Engines
        self.engine = Engines(body=self.body, config=config)
        self.max_force = (
            config.env.drone.engine.max_force
        )  # TODO: move this to Engine class?

        # Raycasting
        ray_length = config.env.drone.raycasting.ray_length

        # Define direction for raycasting in which we look for obstacles.
        self.points = [
            b2Vec2(ray_length, 0.0),
            b2Vec2(0.0, ray_length),
            b2Vec2(-ray_length, 0.0),
            b2Vec2(0.0, -ray_length),
            b2Vec2(ray_length, ray_length),
            b2Vec2(-ray_length, ray_length),
            b2Vec2(-ray_length, -ray_length),
            b2Vec2(ray_length, -ray_length),
        ]

        # Collision threshold
        self.collision_threshold = 1.1 * (2.0**0.5)*(0.5*self.diam+self.engine.height)
        # self.collision_threshold = 1.1 * ((0.5*self.diam+self.engine.height)**2 + (0.5*self.engine.width_max)**2)**0.5 

        # Neural Network
        self.model = NetworkLoader(config=config)()

        # Forces predicted by neural network.
        # Initialized with 0 for each engine.
        self.forces = [0.0 for _ in range(self.num_engines)]

        # Ray casting rendering
        self.callbacks = []
        self.p1 = []
        self.p2 = []

        # Input data for neural network
        self.data = None

        # Fitness score
        self.score = 0.0

        self.path_points = []
        self.path_length = 100
        self.every_point = 10
        self.point_counter = 0

    def mutate(self, model: nn.Module) -> None:
        """Mutates drone's neural network.

        Attr:
            model: The current best model.
        """
        self.model = copy.deepcopy(model)
        self.model.mutate_weights()

    def comp_score(self) -> None:
        """Computes current fitness score.

        Accumulates drone's linear velocity over one generation.
        This effectively computes the distance traveled by the
        drone over time divided by the simulation's step size.
        """
        if self.body.active:
            # Add score just for being alive.
            # Reward drone for living long.
            self.score += 1.0 

            # Maximise distance traveled.
            vel = self.body.linearVelocity
            time_step = 0.0167
            score = time_step * (vel.x**2 + vel.y**2) ** 0.5
            # print("vel score", score)
            self.score += score

            # Penalize drone when too close to an obstacle.
            # eta = 2.0
            # phi = 0.2
            # score = 1.0
            # for cb in self.callbacks:
            #     diff = cb.point - self.body.position
            #     dist = (diff.x**2 + diff.y**2) ** 0.5
            #     if dist < eta * self.collision_threshold:
            #         score = 0.0 
            #         break
            # print("dist score", score)
            # self.score += phi * score

            # Maximise exploration by maximising distance to past path points.
            # rho = 0.1
            # score = 0.0
            # for path_point in self.path_points:
            #     diff = path_point - self.body.position
            #     score += (diff.x**2 + diff.y**2) ** 0.5
            # if score:
            #     score *= rho / len(self.path_points)
            # self.score += score
            # # print("expl score", score)
            # if self.point_counter % self.every_point == 0:
            #     self.path_points.append(copy.copy(self.body.position))
            # if len(self.path_points) > self.path_length:
            #     self.path_points.pop(0)
            # self.point_counter += 1
            # # print()


    def ray_casting(self):
        """Uses ray casting to measure distane to domain walls."""

        if self.body.active:

            self.callbacks = []
            self.p1 = []
            self.p2 = []
            self.data = []

            p1 = self.body.position

            for point in self.points:

                # Perform ray casting from drone position p1 to to point p2.
                p2 = p1 + self.body.GetWorldVector(localVector=point)
                cb = RayCastCallback()
                self.world.RayCast(cb, p1, p2)

                # Save ray casting data for rendering.
                self.p1.append(copy.copy(p1))
                self.p2.append(copy.copy(p2))
                self.callbacks.append(copy.copy(cb))

                # Gather distance data.
                if cb.hit:
                    # When the ray has hit something compute distance
                    # from drone to obstacle from raw features.
                    diff = cb.point - p1
                    dist = (diff.x**2 + diff.y**2) ** 0.5
                    self.data.append(dist)
                else:
                    self.data.append(-1.0)

    def detect_collision(self):
        """Detects collision with objects.

        We use the raycast information here and speak of a collision
        when an imaginary circle with the total diameter of the drone
        touches another object.
        """
        if self.body.active:
            for p1, cb in zip(self.p1, self.callbacks):
                diff = cb.point - p1
                dist = (diff.x**2 + diff.y**2) ** 0.5
                if dist < self.collision_threshold:
                    self.body.active = False
                    self.forces = self.num_engines * [0.0] 
                    self.callbacks = []
                    self.p1 = []
                    self.p2 = []
                    break

    def comp_action(self) -> None:
        """Computes next section of actions applied to engines.

        Next steps of action are computed by feeding obstacle data coming
        from ray casting to the drone's neural network which then returns
        a set of actions (forces) to be applied to the drone's engines.
        """
        if self.body.active:
            force_pred = self.model(self.data)
            self.forces = self.max_force * force_pred

    def apply_action(self) -> None:
        """Applies force to Drone coming from neural network.

        Each engine is controlled individually.
        """
        if self.body.active:
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
