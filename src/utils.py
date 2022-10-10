"""Script with helper functions."""
import random
import numpy


def set_random_seed(seed: int = 0) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
