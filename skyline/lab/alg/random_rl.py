"""This module implement random algorithm to interact with environment."""
import numpy as np
import random

from skyline.lab import alg
from skyline.lab import rl_protos
from tqdm import tqdm


class RandomRL(alg.RLAlgorithm):
  """Random Method."""

  def __init__(self, seed_num: int = 123):
    random.seed(seed_num)

  def fit(self, environment: rl_protos.Environment):
    """Random don't need training."""
    pass

  def play(self, environment: rl_protos.Environment):
    """Take action randomly in the environment."""
    random_action = random.choice(
        environment.available_actions_from_current_state())
    return environment.step(random_action)
