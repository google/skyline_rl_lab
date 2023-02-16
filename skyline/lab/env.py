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

from typing import Any, Optional, Protocol


@dataclasses.dataclass
class ActionResult:
  action: Any
  state: Any
  reward: Any
  is_done: bool
  is_truncated: bool = False
  info: Optional[Any] = None



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

  def available_actions(self) -> list[Any]:
    """Gets available action list."""
    ...

  def random_action(self, s: Optional[Any]=None) -> Optional[Any]:
    """Gets random action from given state."""
    ...

  def available_states(self) -> list[Any]:
    """Gets available state list."""
    ...

  def set_state(self, state: Any):
    """Sets the current state to given one."""
    ...

  @property
  def current_state(self) -> Any:
    """Gets current state."""
    ...

  @property
  def is_done(self) -> bool:
    """Checks if environment is completed."""
    ...
