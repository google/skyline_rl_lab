"""Entry point to get RL testing environment."""
import enum

from skyline.lab import errors
from skyline.lab import gridworld_env


class Env(enum.Enum):
  GridWorld = 'GridWorld'


def make(env: Env) -> env.Environment:
  if env == Env.GridWorld:
    return gridworld_env.GridWorldEnvironment()

  raise UnknownLabEnvError(f'Unknown env={env}')
