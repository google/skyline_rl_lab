"""Module to get RL testing environments."""
import dataclasses

from typing import Any, Optional, Protocol


@dataclasses.dataclass
class ActionResult:
  state: Any
  reward: Any
  is_done: bool
  is_truncated: bool = False
  info: Optional[Any] = None



class Environment(Protocol):
  def reset(self):
    """Resets the environment."""
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
