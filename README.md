# SpaceDrones

*SpaceDrones* provides a simple learning environment for genetic optimization with many possible extensions. The drones' goal is to navigate through the environment, covering as much distance as possible in a certain period of time. *SpaceDrones* comes with different worlds that vary in difficulty.

|||
|:--:|:--:|
|![](docs/map_cross.png)|![](docs/map_track.png)|
|![](docs/map_empty.png)|![](docs/map_block.png)|

# Method

*SpaceDrones* uses *PyBox2D* for the rigid physics simulation and *Pygame* for visualization. The visualization can be turned off to accelerate the optimization process.

The environment consists of a square box with rigid walls, containing one or more drones. Each drone consists of a square body with four boosters attached to each side. Additionally, each drone is equipped with a distance meter that looks in four directions.

# Experiments

...

# Results

...

# References

- [PyBox2D](https://github.com/pybox2d/pybox2d) on GitHub.
- [backends](https://github.com/pybox2d/pybox2d/tree/master/library/Box2D/examples/backends) for PyBox2D.
- PyBox2D [tutorial](https://github.com/pybox2d/cython-box2d/blob/master/docs/source/getting_started.md).
- PyBox2D C++ [documentation](https://box2d.org/documentation/).

# TODO:

- Save best networks
- Use horizontal and vertial rays.
- Allow thrust to have two components
- Take contact to walls into account
    - Reduce score
    - Deactivate drone if it hits wall
- Use data of last $N$ time steps as new input

# License

MIT
