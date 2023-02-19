"""Environment to simulate the test case selection in BCST testing."""

from __future__ import annotations
from copy import deepcopy
import dataclasses
from itertools import permutations, product
import random
from immutabledict import immutabledict

from skyline.lab import errors
from skyline.lab import rl_protos


@dataclasses.dataclass
class TestCaseModel:
  """Model to simulate failure occured during the selection of test case."""
  test_case_name_set: set[str]
  fail_state_info: dict[tuple[str], float]


DEFAULT_TEST_CASE_MODEL = TestCaseModel(
    test_case_name_set = frozenset([
        f'rl_test_case{i}' for i in range(1, 6)]),
    fail_state_info = immutabledict({
        ('rl_test_case1',): 0.1,
        ('rl_test_case2',): 0.05,
        ('rl_test_case3',): 0.2,
        ('rl_test_case4',): 0.01,
        ('rl_test_case5',): 0.1,
        ('rl_test_case2', 'rl_test_case4'): 0.3,
        ('rl_test_case5', 'rl_test_case2'): 0.1,
        ('rl_test_case4', 'rl_test_case1', 'rl_test_case3'): 0.4,
        ('rl_test_case2', 'rl_test_case4', 'rl_test_case1'): 0.6,
        ('rl_test_case3', 'rl_test_case4', 'rl_test_case5'): 1,
    }))


class BCSTEnvironment(rl_protos.Environment):
  """Mock BCST environment to simulate test case selection."""

  def __init__(self, test_case_model: TestCaseModel = DEFAULT_TEST_CASE_MODEL,
               round_num: int = 500):
    self._round_num = round_num
    self._exe_round = 0
    self._test_case_list = list(test_case_model.test_case_name_set)
    self._fail_state_info = test_case_model.fail_state_info.copy()
    self._selected_test_case_history = []
    self._available_state_list = [tuple()] + [tuple([test_case]) for test_case in self._test_case_list]
    self._max_test_case_sequence_length = max([
        len(test_case_sequence) for test_case_sequence in self._fail_state_info.keys()])

    if self._max_test_case_sequence_length > 1:
      for sequence_length in range(2, self._max_test_case_sequence_length+1):
        for test_case_sequence in product(self._test_case_list, repeat=sequence_length):
          self._available_state_list.append(test_case_sequence)

  def info(self) -> Any:
    """Get environment information."""
    print('- Environment as BCST testing environment.')

  def reset(self):
    """Reset the environment."""
    self._selected_test_case_history.clear()
    self._exe_round = 0

  def set_state(self, s: list[str]):
    """Sets the current state."""
    self._selected_test_case_history = list(deepcopy(s))

  def random_action(self, s: Optional[str]=None) -> Optional[str]:
    """Get random action."""
    return random.choice(self._test_case_list)

  def available_actions(self, s: Optional[str]=None) -> list[str]:
    """Get available actions."""
    return self._test_case_list.copy()

  def available_actions_from_current_state(self) -> list[str]:
    """Gets available action list from current state."""
    return self.available_actions()

  def available_states(self) -> list[tuple[str]]:
    """Gets available state list."""
    return self._available_state_list

  @property
  def current_state(self) -> tuple[str, ...]:
    """Gets current state."""
    return tuple(self._selected_test_case_history[
        -self._max_test_case_sequence_length:])

  @property
  def is_done(self) -> bool:
    """Checks if environment is completed."""
    return self._exe_round >= self._round_num

  def step(self, action: str) -> ActionResult:
    self._exe_round += 1
    self._selected_test_case_history.append(action)
    reward = 0
    for seq_length in range(1, len(self.current_state)+1):
      state = self.current_state[-seq_length:]
      fail_prob = self._fail_state_info.get(state, 0)
      if random.uniform(0,1) <= fail_prob:
        reward = 1
        break

    return rl_protos.ActionResult(
        action=action,
        state=deepcopy(self.current_state),
        reward=reward,
        is_done=self.is_done,
        is_truncated=False)


class BCSTRewardCountExaminer(rl_protos.RLExaminer):
  """Examiner to count reward collected by RL method."""

  def score(self, rl_method: RLAlgorithmProto, env: Environment,
            play_round: int=1) -> Comparable:
    """Calculates the score of given RL method."""
    collected_reward_list = []
    for _ in range(play_round):
      env.reset()
      accumulated_reward = 0
      while not env.is_done:
        result = rl_method.play(env)
        accumulated_reward += result.reward

      collected_reward_list.append(accumulated_reward)

    return sum(collected_reward_list) / len(collected_reward_list)
