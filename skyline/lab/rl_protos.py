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

"""Module to get RL testing environments."""
import dataclasses

from typing import Any, List, Optional, Protocol


@dataclasses.dataclass
class ActionResult:
  action: Any
  state: Any
  reward: Any
  is_done: bool
  is_truncated: bool = False
  info: Optional[Any] = None


class Comparable(Protocol):
  """Something that is comparable used in sorting."""
  def __lt__(self, other: Any) -> bool: ...


class Environment(Protocol):
  def reset(self):
    """Resets the environment."""
    ...

  def info(self) -> Any:
    """Get environment information."""
    ...

  def step(self, action: Any) -> ActionResult:
    """Takes action in the environment.

    Args:
      action: Action to take.
    """
    ...

  def available_actions(self, s: Optional[Any]=None) -> List[Any]:
    """Gets available action list (from given state)."""
    ...

  def available_actions_from_current_state(self) -> List[Any]:
    """Gets available action list from current state."""
    ...

  def random_action(self, s: Optional[Any]=None) -> Optional[Any]:
    """Gets random action from given state."""
    ...

  def available_states(self) -> List[Any]:
    """Gets available state list."""
    ...

  def set_state(self, state: Any):
    """Sets the current state to given one."""
    ...

  @property
  def name(self) -> str:
    """Gets name of RL method."""
    ...

  @property
  def current_state(self) -> Any:
    """Gets current state."""
    ...

  @property
  def is_done(self) -> bool:
    """Checks if environment is completed."""
    ...


class RLAlgorithmProto(Protocol):
  """Reinforcement learning algorithm protocol."""

  @property
  def name(self) -> str:
    """Gets name of RL method."""
    return self.__class__.__name__

  def fit(self, environment: Environment):
    """Conducts training by given environment."""
    ...

  def play(self, environment: Environment):
    """Plays in the given environment."""
    ...


class RLExaminer(Protocol):
  """Used to calculate the score of RL method."""

  def score(self, rl_method: RLAlgorithmProto, env: Environment,
            play_round: int=10, show_boxplot: bool=False) -> Comparable:
    """Evaluates the given RL method with given environment.i

    Args:
      rl_method: RL method to evaluation.
      env: Environment to evaluate given RL method.
      play_round: How many times to evaluate the given RL method.
      show_boxplot: True to show the Boxplot of evaluation process.

    Returns:
      Score of evaluation result.
    """
    ...
