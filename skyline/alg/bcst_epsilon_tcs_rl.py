"""This module is the algorithm of BCST RL test case selector."""
from __future__ import annotations

import collections
import dataclasses
import enum
import logging
import random
import sys
from typing import Any

import tqdm

from skyline import alg
from skyline import rl_protos


INIT_EPSILON_VALUE = sys.float_info.min
"""Initial value of Q table."""

DefaultSymtomDictField = dataclasses.field(
    default_factory=collections.defaultdict)
"""Default symptom dict field in dataclass."""


@dataclasses.dataclass
class ExecutionHistory:
  """BCST test case execution history."""
  count: int = 0
  accumulated_reward: float = INIT_EPSILON_VALUE

  def to_dict(self) -> dict[str, Any]:
    """Turns dataclass into json data."""
    return {
        'count': self.count,
        'accumulated_reward': self.accumulated_reward,
    }

  def average_hit(self) -> float:
    """Returns the average crash/ramdump count per test case."""
    return self.accumulated_reward / self.count

  @classmethod
  def from_json(cls, json_data: dict[str, Any]) -> ExecutionHistory:
    """Turns json data into dataclass."""
    return ExecutionHistory(
        count=json_data['count'],
        accumulated_reward=json_data['accumulated_reward'])

Expath2RewardType = dict[tuple[str, ...], ExecutionHistory]
"""Execution path to reward type."""

QtableType = dict[str, Expath2RewardType]
"""Q table type."""


class QtableMetaKey(str, enum.Enum):
  STRATEGY = 'strategy'
  EXECUTED_TC_NUM = 'executed_tc_num'
  DETECTED_SYMPTOM_NUM = 'detected_symptom_num'
  AVERAGE_REWARD = 'average_reward'
  SEARCH_PATH_COVERAGE = 'search_path_coverage'
  EPISODE = 'episode'


class TCSStrategy(alg.RLAlgorithm):
  """Abstract class for test case selection strategy.

  Attributes:
    depth: Search depth
    function_count: Function selection count.
    last_test_case_execution_queue: A queue to keep the last 5 executed test
      case names. The latest executed one will be at the end.
    search_path_coverage: dict with key as search pth and value as coverage.
    execution_path: A tuple as execution path from past executed test cases.
    qtable: Q table learned from execution history.
    round: Round(s) of execution.
  """
  _TEST_CASE_NAME_PLACEHOLDER = 'NA'

  def __init__(
      self, name: str | None = None,
      search_depth: int = 3,
      favor_long_sequence: bool = False):
    super().__init__(name or self.__class__.__name__)
    self._log = logging.getLogger(self.__class__.__name__)
    self._round = 0
    self._search_depth = search_depth
    self._depth2qtable: QtableType = {}
    self._init_qtable()
    self._function_selection_count = {}
    self._last_test_case_execution_queue = (
        collections.deque(
            [self._TEST_CASE_NAME_PLACEHOLDER] * search_depth, search_depth))
    self._favor_long_sequence = favor_long_sequence

  def _reset(self):
    """Reset learned knowledge."""
    self._init_qtable()
    self._function_selection_count = {}
    self._last_test_case_execution_queue = (
        collections.deque(
            [self._TEST_CASE_NAME_PLACEHOLDER] * self._search_depth,
            self._search_depth))

  def _init_qtable(self):
    """Initializes Q tables.

    For each Q-table, the key will be tuple of function name; value is initial
    qvalue. Take `function_name_data`=['a', 'b'] as example and `e` as initial
    qvalue. self._depth2qtable will be:
    {
      1: {('a',): e, ('b',): e},
      2: {('a', 'b'): e, ('b', 'a'): e},
    }
    """
    for d in range(1, self._search_depth + 1):
      qtable: Expath2RewardType = {}
      self._depth2qtable[str(d)] = qtable

  def _update_test_case_selection_stat(self, test_case_name: str) -> None:
    """Updates statistic data by the given selected test case name.

    Args:
      test_case_name: Name of test case.
    """
    self._round += 1
    self._function_selection_count[test_case_name] = (
        self._function_selection_count.get(test_case_name, 0) + 1)
    self._last_test_case_execution_queue.append(test_case_name)

    # Update Qtable
    cur_execution_path = self.execution_path
    for depth in range(1, len(cur_execution_path) + 1):
      qtable = self._depth2qtable[str(depth)]
      execution_path_in_depth = cur_execution_path[-depth:]
      if execution_path_in_depth not in qtable:
        qtable[execution_path_in_depth] = ExecutionHistory()

      qtable[execution_path_in_depth].count += 1

  @property
  def round(self) -> int:
    """Round(s) of execution."""
    return self._round

  @property
  def execution_path(self) -> tuple[str, ...]:
    """Accumulated execution path under given search depth."""
    return tuple(
        filter(lambda n: n != self._TEST_CASE_NAME_PLACEHOLDER,
               tuple(
                   self.last_test_case_execution_queue)))[-self._search_depth:]

  @property
  def qtable(self) -> QtableType:
    """Q table learned from execution history."""
    return self._depth2qtable

  @property
  def depth(self) -> int:
    """Depth of searching space."""
    return self._search_depth

  @property
  def function_count(self) -> dict[str, int]:
    """Function selection count."""
    return self._function_selection_count

  @property
  def last_test_case_execution_queue(self) -> collections.deque[str]:
    """Queue to store last 5 executed test cases."""
    return self._last_test_case_execution_queue

  @property
  def log(self) -> logging.Logger:
    """Logger object literally."""
    return self._log

  def feedback_reward(self, reward: float) -> None:
    """Feedback reward for RC learning.

    Args:
      reward: Reward of detected symptom from the environment.
    """
    # Update qtable
    cur_execution_path = self.execution_path
    cur_execution_path_length = len(cur_execution_path)
    for depth in range(1, cur_execution_path_length + 1):
      qtable = self._depth2qtable[str(depth)]
      execution_path_in_depth = cur_execution_path[-depth:]
      exec_history = qtable.get(execution_path_in_depth, None)
      if not exec_history:
        exec_history = ExecutionHistory()
        qtable[execution_path_in_depth] = exec_history

      reward_factor = (
          depth / cur_execution_path_length if self._favor_long_sequence else 1)
      exec_history.accumulated_reward += reward * reward_factor


class EGreedyStrategy(TCSStrategy):
  """RL ε-Greedy Policy to select BCST test case.

  Attributes:
    eps: Epsilon value of ε-Greedy Policy. (default=0.3)
    beta: Beta value to decide the shrinking rate of Epsilon value when the
        round of test case selection grows. (default=0.1)
    gamma: Gamma value to decide the proportion of impact from future execution
        result. Larger this value, more influence granted from the future.
    delta: Delta value to decide the proportion of adopting the sequences from
        `pre_train_actions` if given.
    round_num: Maximum rounds required in training. (default=0.7)
    foremost_exploring_num: The fixed foremost number of test case selection set
        for exploration. (default=50)
    pre_train_actions: List of actions to select before the training.
    debug_mode: True to turn on debugging mode to print more information.
  """

  def __init__(self,
               name: str | None = None,
               round_num: int = 1000,
               min_search_depth: int = 1,
               search_depth: int = 3,
               eps: float = 0.3,
               beta: float = 0.1,
               gama: float = 0.7,
               delta: float = 0.1,
               foremost_exploring_num: int = 50,
               enable_random_after_training: bool = False,
               enable_explore_after_training: bool = False,
               favor_long_sequence: bool = False,
               debug_mode: bool = False,
               pre_train_actions: list[list[Any]] | None = None):
    super().__init__(name, search_depth, favor_long_sequence)
    self._min_search_depth = min_search_depth
    self._is_under_training = False
    self._round_num = round_num
    self._eps = eps
    self._beta = beta
    self._gama = gama
    self._delta = delta
    self._foremost_exploring_num = foremost_exploring_num
    self._pre_train_actions = pre_train_actions

    # Key as execution path (state); value as previous calculated weighting.
    # https://developers.google.com/machine-learning/glossary#q-function
    self._qfunc = {}
    self._requested_actions: list[Any] = []
    self.debug_mode = debug_mode
    self._enable_random_after_training = enable_random_after_training
    self._enable_explore_after_training = enable_explore_after_training

  @property
  def delta(self) -> float:
    """Delta value for entering mode to follow raw data sequence strictly."""
    return self._delta

  @property
  def eps(self) -> float:
    """Epsilon value of ε-Greedy Policy."""
    if self._is_under_training:
      return self._eps / (1 + self._round * self._beta)

    return 0

  @property
  def foremost_exploring_num(self) -> int:
    """The fixed foremost number of test case selection set for exploration."""
    return self._foremost_exploring_num

  @property
  def round_num(self) -> int:
    """The maximum number of round to interact with environment in training."""
    return self._round_num

  def _next_execution_path(self, search_space: int,
                           new_test_case_name: str,
                           execution_path: tuple[str, ...] | None = None
                           ) -> tuple[str, ...]:
    """Gets next execution path according to input conditions.

    The execution path will have length as input `search_space` and the last
    test case name will be `new_test_case_name`.

    Args:
      search_space: Search space depth.
      new_test_case_name: Next test case name to execute.
      execution_path: Target execution path to formulate next execution path.

    Returns:
      Next execution path.

    Raises:
      ValueError: Invalid search space is given.
    """
    if search_space <= 0 or search_space > self._search_depth:
      raise ValueError(f'Invalid search space={search_space}')

    execution_path = execution_path or self.execution_path
    new_full_execution_path = (execution_path +
                               (new_test_case_name,))[-self._search_depth:]
    return new_full_execution_path[-search_space:]

  def _explore(self, environment: rl_protos.Environment,
               test_case_names: list[str] | None = None) -> str:
    """Conducts exploration in selecting test case.

    Args:
      environment: Environment to explore.
      test_case_names: Test case name list for exploring process.

    Returns:
      The selected test case.
    """
    test_case_names = (
        test_case_names or
        environment.available_actions_from_current_state())
    for test_case_name in random.sample(
        test_case_names, k=len(test_case_names)):
      for search_space in range(1, self._search_depth + 1):
        new_execute_path = self._next_execution_path(search_space,
                                                     test_case_name)
        if new_execute_path not in self.qtable.get(str(search_space), {}):
          self._log.debug(
              'Exploring untouched sequence with test case=%s (state=%s)',
              test_case_name, environment.current_state)
          return test_case_name

    test_case_name = random.choice(test_case_names)
    self._log.debug('Exploring by random with test case=%s (state=%s)',
                    test_case_name, environment.current_state)
    return test_case_name

  def _exploit(self, environment: rl_protos.Environment,
               test_case_names: list[str] | None = None) -> str:
    """Conducts exploitation in selecting test case.

    Args:
      environment: Environment to exploit
      test_case_names: Test case name list for exploiting process.

    Returns:
      The selected test case.
    """
    test_case_name_list = (
        test_case_names or environment.available_actions_from_current_state())
    weighting_data = []
    for test_case_name in test_case_name_list:
      weight = 0
      count = 0
      for search_space in range(self._min_search_depth, self._search_depth + 1):
        new_execution_path = self._next_execution_path(search_space,
                                                       test_case_name)
        sub_qtable = self.qtable.get(str(search_space), {})
        test_case_exec_history = sub_qtable.get(new_execution_path, None)

        if not test_case_exec_history or test_case_exec_history.count == 0:
          weight += INIT_EPSILON_VALUE
          continue

        if self.debug_mode:
          print(
              f'search_space={search_space}; test_case_name={test_case_name};'
              f' new_execution_path={new_execution_path }')
          print(f'test_case_exec_history: {test_case_exec_history}')
        weight += max(
            test_case_exec_history.average_hit(),
            INIT_EPSILON_VALUE) * search_space
        count = test_case_exec_history.count

        # Borrow concept in Bellman equation to calculate impact from
        # future impact of execution result.
        # When we have `self._search_depth <= 1`, it means execution path won't
        # make any sense and we could ignore the future impact as well.
        next_max_weight = 0
        if self._search_depth > 1 and search_space == self._search_depth:
          for test_case_name in test_case_name_list:
            next_execution_path = self._next_execution_path(
                search_space,
                test_case_name,
                execution_path=new_execution_path)
            next_max_weight = max(
                next_max_weight, self._qfunc.get(next_execution_path, 0))

          self._qfunc[new_execution_path] = weight
          weight += next_max_weight * self._gama * self._search_depth

      self._log.debug(
          '\t%s with weighting=%.02f (count=%s)',
          new_execution_path, weight, f'{count:,d}')
      weighting_data.append(weight)

    if self._is_under_training or self._enable_random_after_training:
      return random.choices(test_case_name_list, weights=weighting_data)[0]

    if self.debug_mode:
      for c, w in sorted(
          zip(test_case_name_list, weighting_data),
          key=lambda t: t[1], reverse=True)[:10]:
        print(f'{self.execution_path}->{c}: {w}')

    return sorted(
        zip(test_case_name_list, weighting_data), key=lambda t: t[1])[-1][0]

  def _run_policy(self, environment: rl_protos.Environment) -> str:
    """Runs ε-Greedy Policy to select BCST test case."""
    if not self._requested_actions and self._pre_train_actions:
      p = random.random()
      if p < self.delta:
        self._requested_actions = list(random.choice(self._pre_train_actions))

    if self._requested_actions:
      selected_bcst_test_case = self._requested_actions.pop(0)
      self._update_test_case_selection_stat(selected_bcst_test_case)
      return selected_bcst_test_case

    select_test_case_approach = self._exploit
    if self._is_under_training or self._enable_explore_after_training:
      p = random.random()
      if self.round < self.foremost_exploring_num:
        select_test_case_approach = self._explore
      elif p < self.eps:
        select_test_case_approach = self._explore

    selected_bcst_test_case = select_test_case_approach(environment)

    if selected_bcst_test_case:
      self._update_test_case_selection_stat(selected_bcst_test_case)

    return selected_bcst_test_case

  def fit(self, environment: rl_protos.Environment, do_reset: bool = False):
    """Conduct training from the given environment."""
    if do_reset:
      self._reset()

    self._is_under_training = True

    if self._pre_train_actions:
      self._log.info('Executing pre-training actions...')
      shuffled_actions = random.sample(
          self._pre_train_actions, len(self._pre_train_actions))
      for i in tqdm.tqdm(range(len(shuffled_actions))):
        action_list = shuffled_actions[i]
        environment.reset()
        for action in action_list:
          result = environment.step(action)
          self._update_test_case_selection_stat(action)
          if result.reward != 0:
            self.feedback_reward(result.reward)
          if environment.is_done:
            break

    self._log.info(
        'Start training by interacting with environment=%s...',
        environment.name)
    for _ in tqdm.tqdm(range(self.round_num)):
      environment.reset()
      while not environment.is_done:
        result = self.play(environment)
        if result.reward > 0:
          self.feedback_reward(result.reward)

    self._is_under_training = False

  def play(self, environment: rl_protos.Environment):
    """Select test case from the given environment."""
    return environment.step(self._run_policy(environment))

  def passive_play(self, environment: rl_protos.Environment) -> Any:
    return self._run_policy(environment)
