import random
import numpy
import pygame
import torch

from pathlib import Path

from src.utils import Config


def set_random_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def capture_screen(screen, iteration: int) -> None:
    filename = Path("frames") / f"{iteration:04d}.png"
    pygame.image.save(screen, filename)


def save_checkpoint(model: object, config: Config) -> None:
    model_name = config.checkpoints.model_name
    checkpoint_name = f"{f'{model_name}' if model_name else ''}"
    model_path = Path(config.checkpoints.model_path) / f"{checkpoint_name}.pth"
    torch.save(obj=model.state_dict(), f=model_path)


def load_checkpoint(model: object, config: Config) -> None:
    model_name = config.checkpoints.model_name
    checkpoint_name = f"{f'{model_name}' if model_name else ''}"
    model_path = Path(config.checkpoints.model_path) / f"{checkpoint_name}.pth"

    if model_path.is_file():
        state_dict = torch.load(f=model_path)
        model.load_state_dict(state_dict=state_dict)
    else:
        print(
            f"Model checkpoint '{checkpoint_name}' not found. \
            Continuing with random weights."
        )
