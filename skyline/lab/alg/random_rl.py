# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
