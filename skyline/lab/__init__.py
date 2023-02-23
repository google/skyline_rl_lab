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

"""Entry point to get RL testing environment."""
import enum

from prettytable import PrettyTable

from skyline.lab import errors
from skyline.lab import gridworld_env
from skyline.lab import rl_protos

from typing import Dict, List


class Scoreboard:
  """Scoreboard to rank the RL methods."""

  def rank(self, examiner: rl_protos.RLExaminer,
           env: rl_protos.Environment,
           rl_methods: List[rl_protos.RLAlgorithmProto]) -> Dict[rl_protos.RLAlgorithmProto, rl_protos.Comparable]:
    """Ranks the given RL methods."""
    rl_2_score_dict = {}
    for rl_method in rl_methods:
      rl_2_score_dict[rl_method.name] = examiner.score(rl_method, env)

    sorted_scores = sorted(
        rl_2_score_dict.items(), key=lambda t: t[1], reverse=True)

    score_board = PrettyTable()
    score_board.field_names = ['Rank.', 'RL Name', 'Score']
    for rank, score_info in enumerate(sorted_scores, start=1):
      score_board.add_row([rank, score_info[0], score_info[1]])

    print(score_board)
    return sorted_scores


class Env(enum.Enum):
  GridWorld = (
      'GridWorld',
      (
          'This is a environment to show case of Skyline lab. '
          'The environment is a grid world where you can move up, down, right and left'
          'if you don\'t encounter obstacle. When you obtain the reward (-1, 1, 2), '
          'the game is over. You can use env.info() to learn more.'))


def list_env():
  """List supported environment(s)."""
  for env_enum in Env:
    env_name, env_desc = env_enum.value
    print(f'===== {env_name} =====')
    print(env_desc)
    print("")

  print("")


def make(env: Env) -> rl_protos.Environment:
  """Make the environment.

  Args:
    env: Enum of environment to make.

  Returns:
    Target environment object.

  Raises:
    errors.UnknownLabEnvError: The given environment enum is unknown.
  """
  if env == Env.GridWorld:
    return gridworld_env.GridWorldEnvironment()

  raise errors.UnknownLabEnvError(f'Unknown env={env}')
