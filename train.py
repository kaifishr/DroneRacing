"""Runs selected optimization."""
from src.environment import Environment
from src.optimizer import EvolutionStrategy
from src.optimizer import GeneticOptimizer
from src.trainer import Trainer
from src.utils.config import init_config
from src.utils.utils import set_random_seed


if __name__ == "__main__":
    config = init_config(path="config.yml")
    set_random_seed(seed=config.random_seed)

    env = Environment(config=config)

    if config.optimizer.name == "nes":
        cfg = config.optimizer.nes
        optimizer = EvolutionStrategy(
            agents=env.drones,
            learning_rate=cfg.learning_rate,
            sigma=cfg.sigma,
        )

    elif config.optimizer.name == "evo":
        cfg = config.optimizer.evo
        optimizer = GeneticOptimizer(
            agents=env.drones,
            mutation_probability=cfg.mutation_probability,
            mutation_rate=cfg.mutation_rate,
        )
    else:
        raise NotImplementedError(f"Optimizer '{config.optimizer.name}' not implemented.")

    trainer = Trainer(env=env, optimizer=optimizer, config=config)
    trainer.run()
