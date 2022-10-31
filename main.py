"""Runs the genetic optimization."""
from src.config import load_config
from src.optimizer import Optimizer
from src.utils import set_random_seed


if __name__ == "__main__":
    config = load_config(path="config.yml")
    set_random_seed(seed=config.random_seed)
    optimizer = Optimizer(config=config)
    optimizer.run()
