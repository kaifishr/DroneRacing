"""Runs the genetic optimization."""
from src.environment import Environment
from src.optimizer import EvolutionStrategy
from src.optimizer import GeneticOptimizer
from src.trainer import Trainer
from src.utils.config import init_config
from src.utils.utils import set_random_seed


if __name__ == "__main__":
    config = init_config(path="config.yml")
    set_random_seed(seed=config.random_seed)

    config.env.name = "catch"  # collect

    env = Environment(config=config)

    if config.optimizer.name == "nes":
        optimizer = EvolutionStrategy(
            agents=env.drones,
            learning_rate=0.01,
            sigma=0.2,
        )

    elif config.optimizer.name == "evo":
        optimizer = GeneticOptimizer(
            agents=env.drones,
            mutation_probability=0.05,
            mutation_rate=0.05,
        )
    else:
        raise NotImplementedError(
            f"Optimizer '{config.optimizer.name}' not implemented."
        )

    trainer = Trainer(env=env, optimizer=optimizer, config=config)
    trainer.run()
