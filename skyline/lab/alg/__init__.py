import numpy as np

from skyline.lab import env
from typing import Any, Optional, Mapping, Protocol, Union


class RLAlgorithmProto(Protocol):
  """Reinforcement learning algorithm protocol."""

  def fit(self, environment: env.Environment):
    """Conducts training by given environment."""
    ...

  def play(self, environment: env.Environment):
    """Plays in the given environment."""
    ...


class RLAlgorithm(RLAlgorithmProto):

  def max_dict(self, d: Mapping[Any, Union[int, float]]):
    """ returns the argmax (key) and max (value) from a dictionary

    put this into a function since we are using it so often
    find max val.
    """
    max_val = max(d.values())

    # find keys corresponding to max val
    max_keys = [key for key, val in d.items() if val == max_val]

    return np.random.choice(max_keys), max_val
