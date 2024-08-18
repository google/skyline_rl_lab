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

"""GridWorld environment to play with RL algorithms."""
from __future__ import annotations

import random
from skyline.lab import errors
from skyline.lab import rl_protos
from typing import Any, Optional
import enum
import dataclasses


ActionResult = rl_protos.ActionResult
Comparable = rl_protos.Comparable
RLAlgorithmProto = rl_protos.RLAlgorithmProto
Environment = rl_protos.Environment


@dataclasses.dataclass
class GridState:
  """State in GridWorld."""
  i: int
  j: int

  def copy(self) -> GridState:
    return GridState(i=self.i, j=self.j)

  def __eq__(self, state):
    if isinstance(state, GridState):
      return state.i == self.i and state.j == self.j

    return False

  def __hash__(self):
    return hash((self.i, self.j))


class GridAction(enum.Enum):
  """Actions in GridWorld."""
  UP = ('U', -1, 0)
  DOWN = ('D', 1, 0)
  LEFT = ('L', 0, -1)
  RIGHT = ('R', 0, 1)

  @classmethod
  def get_move(cls, action_str: str):
    for action in cls:
      if action.value[0] == action_str:
        return action.value[1:]


class GridWorldEnvironment(rl_protos.Environment):
  """GridWorld environment for testing RL algorithm.

  This class defines a grid that describes the reward for arriving at each
  state and possible actions at each state.

  The grid looks like this:
  - x means you can't go there
  - s means start position
  - number means reward at that state
  ```
  .  .  .  1
  .  x  . -1
  .  .  .  x
  s  x  .  2
  ```
  """

  def __init__(self, init_state: Optional[GridState] = None):
    self.rows = 4
    self.cols = 4
    init_state = init_state or GridState(i=3, j=0)
    self._begin_state = init_state
    self._state = self._begin_state.copy()
    self.rewards = {
        GridState(i=0, j=3): 1,
        GridState(i=1, j=3): -1,
        GridState(i=3, j=3): 2}
    self.actions = {
      GridState(0, 0): ('D', 'R'),
      GridState(0, 1): ('L', 'R'),
      GridState(0, 2): ('L', 'D', 'R'),
      GridState(1, 0): ('U', 'D'),
      GridState(1, 2): ('U', 'D', 'R'),
      GridState(2, 0): ('U', 'D', 'R'),
      GridState(2, 1): ('L', 'R'),
      GridState(2, 2): ('L', 'R', 'U', 'D'),
      GridState(3, 0): ('U'),
      GridState(3, 2): ('U', 'R'),
    }

  def info(self) -> Any:
    """Get environment information."""
    print('- environment is a grid world')
    print("- x means you can't go there")
    print('- s means start position')
    print('- number means reward at that state')
    print('===========')
    print('.  .  .  1')
    print('.  x  . -1')
    print('.  .  .  x')
    print('s  x  .  2')
    print('===========\n')

  def reset(self):
    """Reset the environment."""
    self._state = self._begin_state.copy()

  def set_state(self, s: GridState):
    """Sets the current state."""
    self._state = s.copy()

  def random_action(self, s: Optional[GridState] = None) -> Optional[str]:
    """Get random action."""
    s = s or self.current_state
    return random.choice(self.actions.get(s, [None]))

  def available_actions(self, s: Optional[GridState] = None) -> list[Any]:
    """Get available actions."""
    if not s:
      return [action.value[0] for action in GridAction]

    return self.actions[s]

  def available_actions_from_current_state(self) -> list[Any]:
    """Gets available action list from current state."""
    return self.actions.get(self.current_state, [])

  def available_states(self) -> list[Any]:
    """Gets available state list."""
    return [state.copy() for state in self.actions.keys()]

  def step(self, action: str) -> ActionResult:
    # check if legal move first
    new_state = self._state.copy()
    if action not in self.actions[new_state]:
      raise errors.IllegalActionError(
          f'action={action} can not in state={new_state}')

    move_i, move_j = GridAction.get_move(action)
    new_state.i += move_i
    new_state.j += move_j
    self._state = new_state.copy()
    reward = self.rewards.get(self._state, 0)

    return rl_protos.ActionResult(
        action=action,
        state=new_state,
        reward=reward,
        is_done=self.is_done,
        is_truncated=False)

  @property
  def current_state(self) -> GridState:
    """Gets current state."""
    return self._state.copy()

  @property
  def is_done(self) -> bool:
    """Checks if environment is completed."""
    return self._state not in self.actions


class GridWorldExaminer(rl_protos.RLExaminer):
  """Examiner of GridWorld."""

  def score(self, rl_method: RLAlgorithmProto, env: Environment,
            play_round: int = 1, show_boxplot: bool = False) -> Comparable:
    """Calculates the score of given RL method."""
    collected_reward_list = []
    for _ in range(play_round):
      env.reset()
      step_count = 0
      accumulated_reward = 0
      while not env.is_done:
        result = rl_method.play(env)
        step_count += 1
        accumulated_reward += result.reward

      collected_reward_list.append(accumulated_reward / step_count)

    return sum(collected_reward_list) / len(collected_reward_list)
