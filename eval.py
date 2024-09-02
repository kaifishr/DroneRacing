"""Runs an evaluation for a pre-trained agent."""

from src.environment import Environment
from src.evaluator import Eval
from src.utils.config import init_config
from src.utils.utils import set_random_seed


if __name__ == "__main__":
    config = init_config(path="config.yml")
    set_random_seed(seed=config.random_seed)

    config.optimizer.num_agents = 8
    config.env.drone.respawn = "all_random"

    env = Environment(config=config)
    evaluator = Eval(env=env, config=config)
    evaluator.run()
