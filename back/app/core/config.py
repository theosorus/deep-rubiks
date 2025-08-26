# Standard
import os
from enum import Enum
from typing import List
from datetime import timedelta

# Third-party imports
from starlette.config import Config
from pydantic_settings import BaseSettings
from functools import lru_cache
from solver.utils import get_device
import torch


current_file_dir = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(current_file_dir, "..", ".env")
config = Config(env_path)


class AppDetails(BaseSettings):
    APP_NAME: str = "deep-rubiks"
    DEVICE : torch.device = get_device()
    MODEL_PATH : str = "./output/rubiks_cube_solver_small.pth"
    







class Settings(
    AppDetails
):
    pass


@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
