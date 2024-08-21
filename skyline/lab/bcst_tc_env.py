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

"""Environment to simulate the test case selection in BCST testing."""

from __future__ import annotations
import abc
import collections
from copy import deepcopy
from typing import Any, FrozenSet, Mapping, Optional, Protocol, TypeAlias
import dataclasses
from itertools import product
import matplotlib.pyplot as plt
import random
from immutabledict import immutabledict

from skyline.lab import rl_protos


ActionResult = rl_protos.ActionResult
Comparable = rl_protos.Comparable
Environment = rl_protos.Environment
RLAlgorithmProto = rl_protos.RLAlgorithmProto


class _ComparableOp(Protocol):
  """Protocol for annotating comparable types."""

  @abc.abstractmethod
  def __lt__(self: Any, other: Any) -> bool:
    pass

Comparable: TypeAlias = float | int | _ComparableOp


@dataclasses.dataclass
class TestCaseModel:
  """Model to simulate failure occured during the selection of test case."""
  test_case_name_set: set[str]
  fail_state_info: dict[tuple[str], float]


DEFAULT_TEST_CASE_MODEL = TestCaseModel(
    test_case_name_set=frozenset([
        f'rl_test_case{i}' for i in range(1, 6)]),
    fail_state_info=immutabledict({
        ('rl_test_case1',): 0.01,
        ('rl_test_case2',): 0.05,
        ('rl_test_case3',): 0.05,
        ('rl_test_case4',): 0.01,
        ('rl_test_case5',): 0.01,
        ('rl_test_case6',): 0.01,
        ('rl_test_case1', 'rl_test_case2'): 0.01,
        ('rl_test_case1', 'rl_test_case3'): 0.01,
        ('rl_test_case1', 'rl_test_case4'): 0.04,
        ('rl_test_case2', 'rl_test_case4'): 0.03,
        ('rl_test_case2', 'rl_test_case6'): 0.07,
        ('rl_test_case3', 'rl_test_case4'): 0.01,
        ('rl_test_case3', 'rl_test_case5'): 0.01,
        ('rl_test_case3', 'rl_test_case6'): 0.05,
        ('rl_test_case5', 'rl_test_case2'): 0.01,

        ('rl_test_case1', 'rl_test_case2', 'rl_test_case3'): 0.5,
        ('rl_test_case1', 'rl_test_case2', 'rl_test_case2'): 0.1,
        ('rl_test_case1', 'rl_test_case2', 'rl_test_case4'): 0.1,
        ('rl_test_case1', 'rl_test_case2', 'rl_test_case5'): 0.2,
        ('rl_test_case1', 'rl_test_case2', 'rl_test_case6'): 0.3,

        ('rl_test_case2', 'rl_test_case4', 'rl_test_case1'): 0.2,
        ('rl_test_case2', 'rl_test_case3', 'rl_test_case4'): 0.3,
        ('rl_test_case2', 'rl_test_case3', 'rl_test_case6'): 0.5,
        ('rl_test_case2', 'rl_test_case3', 'rl_test_case1'): 0.3,
        ('rl_test_case2', 'rl_test_case3', 'rl_test_case5'): 0.3,

        ('rl_test_case3', 'rl_test_case4', 'rl_test_case5'): 1,
        ('rl_test_case3', 'rl_test_case4', 'rl_test_case4'): 0.2,
        ('rl_test_case3', 'rl_test_case4', 'rl_test_case3'): 0.2,
        ('rl_test_case3', 'rl_test_case5', 'rl_test_case1'): 0.5,
        ('rl_test_case3', 'rl_test_case5', 'rl_test_case2'): 0.1,
        ('rl_test_case3', 'rl_test_case5', 'rl_test_case3'): 0.1,
        ('rl_test_case3', 'rl_test_case5', 'rl_test_case4'): 0.2,

        ('rl_test_case4', 'rl_test_case1', 'rl_test_case2'): 0.1,
        ('rl_test_case4', 'rl_test_case1', 'rl_test_case3'): 0.2,
        ('rl_test_case4', 'rl_test_case1', 'rl_test_case4'): 0.6,
        ('rl_test_case4', 'rl_test_case1', 'rl_test_case5'): 0.1,

        ('rl_test_case5', 'rl_test_case2', 'rl_test_case4'): 0.6,
        ('rl_test_case5', 'rl_test_case2', 'rl_test_case2'): 0.1,
        ('rl_test_case5', 'rl_test_case2', 'rl_test_case3'): 0.1,
        ('rl_test_case5', 'rl_test_case2', 'rl_test_case1'): 0.2,

        ('rl_test_case6', 'rl_test_case3', 'rl_test_case1'): 0.1,
        ('rl_test_case6', 'rl_test_case2', 'rl_test_case4'): 0.5,
        ('rl_test_case6', 'rl_test_case2', 'rl_test_case1'): 0.1,
        ('rl_test_case6', 'rl_test_case2', 'rl_test_case3'): 0.1,
        ('rl_test_case6', 'rl_test_case2', 'rl_test_case5'): 0.1,
        ('rl_test_case6', 'rl_test_case4', 'rl_test_case2'): 0.5,
    }))


class BCSTEnvironment(rl_protos.Environment):
  """Mock BCST environment to simulate test case selection.

  Attributes:
    round_num: Number of round to selecte test case. It implies that the
        environment will be done after `round_num` of test cases being
        selected.
  """

  def __init__(self, test_case_model: TestCaseModel = DEFAULT_TEST_CASE_MODEL,
               round_num: int = 500):
    self._round_num = round_num
    self._exe_round = 0
    self._test_case_list = list(test_case_model.test_case_name_set)
    self._fail_state_info = test_case_model.fail_state_info.copy()
    self._selected_test_case_history = []
    self._available_state_list = (
        [tuple()] +
        [tuple([test_case])for test_case in self._test_case_list])
    self._max_test_case_sequence_length = max([
        len(test_case_sequence)
        for test_case_sequence in self._fail_state_info.keys()])

    if self._max_test_case_sequence_length > 1:
      for sequence_length in range(2, self._max_test_case_sequence_length+1):
        for test_case_sequence in product(
            self._test_case_list, repeat=sequence_length):
          self._available_state_list.append(test_case_sequence)

  @property
  def current_state(self) -> tuple[str, ...]:
    """Gets current state."""
    return tuple(self._selected_test_case_history[
        -self._max_test_case_sequence_length:])

  @property
  def is_done(self) -> bool:
    """Checks if environment is completed."""
    return self._exe_round >= self.round_num

  @property
  def name(self) -> str:
    """Gets name of RL method."""
    return 'BCST test case execution simulator'

  @property
  def round_num(self) -> int:
    return self._round_num

  def info(self) -> Any:
    """Get environment information."""
    print('- Environment as BCST testing environment.')
    print('- You can set attribute `round_num` of environment to decide the max round'  # noqa: E501
          '  of execution.')
    print('- Each action is a test case to select.')
    print('- State is the sequence of last executed test case sequence.')
    print('- A reward equal to 1 means the execution resulted in a crash/ramdump '  # noqa: E501
          'or 0 means nothing was caught.')

  def reset(self):
    """Reset the environment."""
    self._selected_test_case_history.clear()
    self._exe_round = 0

  def set_state(self, s: list[str]):
    """Sets the current state.

    Args:
      state: State to set in environment. Here is a list of executed test cases
          in sequence.
    """
    self._selected_test_case_history = list(deepcopy(s))

  def random_action(self, s: Optional[str] = None) -> Optional[str]:
    """Get random action."""
    return random.choice(self._test_case_list)

  def available_actions(self, s: Optional[str] = None) -> list[str]:
    """Get available actions."""
    return self._test_case_list.copy()

  def available_actions_from_current_state(self) -> list[str]:
    """Gets available action list from current state."""
    return self.available_actions()

  def available_states(self) -> list[tuple[str]]:
    """Gets available state list."""
    return self._available_state_list

  def step(self, action: str, tentative: bool = False) -> ActionResult:
    """Executes the given action and return result.

    Args:
      action: Action to execute. In current environment, it is the
          test case to execute.
      tentative: True to return the result without changing the state of
          environment.

    Returns:
      Executed result including current state, reward etc.
    """
    self._exe_round += 1
    if not tentative:
      self._selected_test_case_history.append(action)

    reward = 0
    for seq_length in range(1, len(self.current_state)+1):
      state = self.current_state[-seq_length:]
      fail_prob = self._fail_state_info.get(state, 0)
      if random.uniform(0, 1) <= fail_prob:
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

  def score(self, rl_method: RLAlgorithmProto,
            env: Environment,
            play_round: int = 10,
            show_boxplot: bool = False,
            extra_data: Any | None = None) -> rl_protos.ExamineScore:
    """Calculates the score of given RL method."""
    collected_reward_list = []
    for _ in range(play_round):
      env.reset()
      accumulated_reward = 0
      while not env.is_done:
        result = rl_method.play(env)
        accumulated_reward += result.reward

      collected_reward_list.append(accumulated_reward)

    if show_boxplot:
      fig, ax = plt.subplots()
      ax.set_title(f'RL method({rl_method.name}) on environment {env}')
      ax.boxplot(collected_reward_list, vert=False)

    return (
        sum(collected_reward_list) / len(collected_reward_list),
        collected_reward_list)


class BCSTRewardDiversityExaminer(rl_protos.RLExaminer):
  """Examiner to examine diversity of reward collected by RL method.

  The score of this examiner is composed of two parts:
  - Part1: Diversity of state.
  - Part2: Diversity of reward obtained from state.

  From input arguments, you could use:
  - `weight_of_action_diversity` to adjust portion of contribution of part1.
  - `weight_of_reward_diversity` to adjust portion of contribution of part2.

  `weight_of_action_diversity` + `weight_of_reward_diversity` SHOULD equal 1.
  """

  def __init__(self, weight_of_action_diversity: float = 0.1,
               weight_of_reward_diversity: float = 0.9):
    super().__init__()
    self._weight_of_action_diversity = weight_of_action_diversity
    self._weight_of_reward_diversity = weight_of_reward_diversity
    assert(weight_of_action_diversity + weight_of_reward_diversity == 1)

  def score_with_action_2_reward_info(
      self, rl_method: rl_protos.RLAlgorithmProto,
      env: rl_protos.Environment,
      play_round: int = 10) -> tuple[Comparable, list[Comparable], list[Any]]:
    """Calculates the score of given RL method."""
    collected_reward_list = []
    collected_action_2_reward_info = []

    class _FixedActionSeq:
      def __init__(self, size: int = 3):
        self.size = size
        self.data = []

      def add(self, action: Any):
        self.data.append(action)
        if len(self.data) > self.size:
          del self.data[0]

      def __str__(self):
        return ','.join(self.data)

    for _ in range(play_round):
      env.reset()
      taken_action_set = set()
      action_seq = _FixedActionSeq()
      action_2_reward_map = collections.defaultdict(list)
      while not env.is_done:
        result = rl_method.play(env)
        action_seq.add(result.action)
        action_seq_key = str(action_seq)
        taken_action_set.add(action_seq_key)
        if result.reward:
          action_2_reward_map[action_seq_key].append(result.reward)

      reward = 0
      reward += self._weight_of_action_diversity * len(taken_action_set)
      accumulated_action_reward = 0
      for _, reward_list in action_2_reward_map.items():
        accumulated_action_reward += sum(
            [r / (i * 10 + 1) for i, r in enumerate(reward_list)])

      reward += (
          self._weight_of_reward_diversity * accumulated_action_reward)
      collected_reward_list.append(reward)
      collected_action_2_reward_info.append(action_2_reward_map)

    return (
        sum(collected_reward_list) / len(collected_reward_list),
        collected_reward_list, collected_action_2_reward_info)

  def score(self, rl_method: rl_protos.RLAlgorithmProto,
            env: rl_protos.Environment,
            play_round: int = 10,
            show_boxplot: bool = False,
            extra_data: Any | None = None) -> rl_protos.ExamineScore:
    """Calculates the score of given RL method."""
    avg_score, collected_reward_list, _ = (
        self.score_with_action_2_reward_info(
            rl_method=rl_method,
            env=env,
            play_round=play_round))

    if show_boxplot:
      _, ax = plt.subplots()
      ax.set_title(f'RL method({rl_method.name}) on environment {env}')
      ax.boxplot(collected_reward_list, vert=False)

    return avg_score, collected_reward_list
