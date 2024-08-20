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
import abc
import dataclasses

from typing import Any, Sequence, Optional, Protocol, TypeAlias


@dataclasses.dataclass
class ActionResult:
  action: Any
  state: Any
  reward: Any
  is_done: bool
  is_truncated: bool = False
  info: Optional[Any] = None


class _ComparableOp(Protocol):
  """Protocol for annotating comparable types."""

  @abc.abstractmethod
  def __lt__(self: Any, other: Any) -> bool:
    pass


Comparable: TypeAlias = float | int | _ComparableOp
ExamineScore: TypeAlias = tuple[Comparable, list[Comparable]]


class Environment(Protocol):
  """Environment of RL problem."""

  def reset(self):
    """Resets the environment."""
    ...

  def info(self) -> Any:
    """Get environment information."""
    ...

  def step(self, action: Any, tentative: bool = False) -> ActionResult:
    """Takes action in the environment.

    Args:
      action: Action to take.
      tentative: True to return the result without changing the state of
          environment.
    """
    ...

  def available_actions(self, s: Optional[Any] = None) -> Sequence[Any]:
    """Gets available action list (from given state)."""
    ...

  def available_actions_from_current_state(self) -> Sequence[Any]:
    """Gets available action list from current state."""
    ...

  def random_action(self, s: Optional[Any] = None) -> Optional[Any]:
    """Gets random action from given state."""
    ...

  def render(self):
    """Renders environment."""
    ...

  def available_states(self) -> list[Any]:
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

  @property
  def round_num(self) -> int:
    """Maximum number of playing round."""
    ...

  @round_num.setter
  def round_num(self, val: int):
    """Sets maximum number of playing round."""
    ...


class RLAlgorithmProto(Protocol):
  """Reinforcement learning algorithm protocol.

  Attributes:
    name: Name of RL algorithm.
  """

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

  def passive_play(self, environment: Environment):
    """Passive play by returning action only."""
    ...


class RLExaminer(Protocol):
  """Used to calculate the score of RL method."""

  def score(self, rl_method: RLAlgorithmProto, env: Environment,
            play_round: int = 10, show_boxplot: bool = False,
            extra_data: Any | None = None) -> ExamineScore:
    """Evaluates the given RL method with given environment.i

    Args:
      rl_method: RL method to evaluation.
      env: Environment to evaluate given RL method.
      play_round: How many times to evaluate the given RL method.
      show_boxplot: True to show the Boxplot of evaluation process.
      extra_data: Extra data used in calculating score.

    Returns:
      Score information of evaluation result.
    """
    ...
