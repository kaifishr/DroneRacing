"""Runs an evaluation for a pre-trained agent."""
from src.environment import Environment
from src.evaluator import Eval
from src.utils.config import init_config
from src.utils.utils import set_random_seed


if __name__ == "__main__":
    config = init_config(path="config.yml")
    set_random_seed(seed=config.random_seed)

    config.optimizer.num_agents = 16
    config.env.drone.respawn.is_random = False
    config.env.drone.respawn.is_all_random = True

    env = Environment(config=config)
    evaluator = Eval(env=env, config=config)
    evaluator.run()
