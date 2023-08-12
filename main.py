"""Runs the genetic optimization."""
from src.environment import Environment
from src.optimizer_ import EvolutionStrategy
from src.optimizer_ import GeneticOptimizer
from src.optimizer import Trainer
from src.utils.config import init_config
from src.utils.utils import set_random_seed


if __name__ == "__main__":
    config = init_config(path="config.yml")
    set_random_seed(seed=config.random_seed)

    config.id = 1

    env = Environment(config=config)

    optimizer = EvolutionStrategy(
        agents=env.drones,
        learning_rate=0.1,
        sigma=0.1,
    )

    # optimizer = GeneticOptimizer(
    #     agents=env.drones,
    #     mutation_probability=0.1,
    #     mutation_rate=0.1,
    # )

    trainer = Trainer(env=env, optimizer=optimizer, config=config)
    trainer.run()
