"""Runs an evaluation for a pre-trained agent."""
from src.environment import Environment
from src.optimizer import EvolutionStrategy
from src.optimizer import GeneticOptimizer
from src.trainer import Eval
from src.utils.config import init_config
from src.utils.utils import set_random_seed


if __name__ == "__main__":
    config = init_config(path="config.yml")
    set_random_seed(seed=config.random_seed)

    config.env.name = "catch"  # collect, snitch
    config.optimizer.num_agents = 16

    env = Environment(config=config)
    eval = Eval(env=env, config=config)
    eval.run()
