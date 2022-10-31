"""Raycasting for drones. The drone's eyes."""
from Box2D import b2RayCastCallback
from Box2D.Box2D import b2Vec2


class RayCastCallback(b2RayCastCallback):
    """Callback detects closest hit.

    See also this example for more information about ray casting in PyBox2D:
    https://github.com/pybox2d/pybox2d/blob/master/library/Box2D/examples/raycast.py

    Attributes:
        fixture:
        hit:
    """

    def __init__(self, **kwargs) -> None:
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction) -> float:
        """Reports hit fixture.

        Args:
            fixture:
            point:
            normal:
            fraction:
        """
        # Ignore engines.
        if fixture.filterData.groupIndex == -1:
            return 1.0

        # Fixture of the hit body. Interesting for multi-agent environments.
        self.hit = True
        self.fixture = fixture

        # Point of contact of body.
        self.point = b2Vec2(point)

        # Normal vector at point of contact. Perpendicular to body surface.
        self.normal = b2Vec2(normal)

        return fraction
