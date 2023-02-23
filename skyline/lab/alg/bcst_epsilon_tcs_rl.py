"""This module is a migrated version of BCST RL test case selector in G3."""
from __future__ import annotations

import collections
import dataclasses
import enum
import logging
import numpy as np
import random
import sys

from skyline.lab import alg
from skyline.lab import rl_protos
from tqdm import tqdm
from typing import Any, List, Optional, Dict, Tuple


INIT_EPSILON_VALUE = sys.float_info.min
"""Initial value for Q table."""

DefaultSymtomDictField = dataclasses.field(
    default_factory=collections.defaultdict)
"""Default symptom dict field in dataclass."""

@dataclasses.dataclass
class ExecutionHistory:
  """BCST test case execution history."""
  count: int = 0
  accumulated_reward: float = INIT_EPSILON_VALUE

  def to_dict(self) -> Dict[str, Any]:
    """Turns dataclass into json data."""
    tmp_detected_symptom_counter = {}

    return {
        'count': self.count,
        'accumulated_reward': self.accumulated_reward,
    }

  def average_hit(self) -> float:
    """Returns the average crash/ramdump count per test case."""
    return self.accumulated_reward / self.count

  @classmethod
  def from_json(cls, json_data: Dict[str, Any]) -> ExecutionHistory:
    """Turns json data into dataclass."""
    eh = ExecutionHistory(
        count=json_data['count'],
        accumulated_reward=json_data['accumulated_reward'])

    return eh


QtableType = Dict[str, Dict[Tuple[str, ...], ExecutionHistory]]
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

  def __init__(self, name: Optional[str]=None, search_depth: int = 3):
    super().__init__()
    self._name = name or self.__class__.__name__
    self._log = logging.getLogger(self.__class__.__name__)
    self._round = 0
    self._search_depth = search_depth
    self._depth2qtable: QtableType = {}
    self._init_qtable()
    self._function_selection_count = {}
    self._last_test_case_execution_queue = (
        collections.deque([self._TEST_CASE_NAME_PLACEHOLDER] * 5, 5))

  @property
  def name(self) -> str:
    """Gets name of RL method."""
    return self._name

  def _reset(self):
    """Reset learned knowledge."""
    self._init_qtable()
    self._function_selection_count = {}
    self._last_test_case_execution_queue = (
        collections.deque([self._TEST_CASE_NAME_PLACEHOLDER] * 5, 5))

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
      qtable = {}
      self._depth2qtable[str(d)] = qtable

  def _update_test_case_selection_stat(self, test_case_name: str) -> None:
    """Updates statistic data by the given selected test case name."""
    self._round += 1
    self._function_selection_count[test_case_name] = (
        self._function_selection_count.get(test_case_name, 0) + 1)
    self._last_test_case_execution_queue.append(test_case_name)

    # Update qtable
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
  def execution_path(self) -> Tuple[str, ...]:
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
  def function_count(self) -> Dict[str, int]:
    """Function selection count."""
    return self._function_selection_count

  @property
  def last_test_case_execution_queue(self) -> collections.deque[str]:
    """Queue to store last 5 executed test cases."""
    return self._last_test_case_execution_queue

  def feedback_reward(self, reward: float) -> None:
    """Feedback reward for RC learning.

    Args:
      detected_symptom: Detected symptom from the environment as reward.
    """
    # Update qtable
    cur_execution_path = self.execution_path
    for depth in range(1, len(cur_execution_path) + 1):
      qtable = self._depth2qtable[str(depth)]
      execution_path_in_depth = cur_execution_path[-depth:]
      exec_history = qtable.get(execution_path_in_depth, None)
      if not exec_history:
        exec_history = ExecutionHistory()
        qtable[execution_path_in_depth] = exec_history

      exec_history.accumulated_reward += reward


class EGreedyStrategy(TCSStrategy):
  """RL ε-Greedy Policy to select BCST test case.

  Attributes:
    eps: Epsilon value of ε-Greedy Policy.
    beta: Beta value to decide the shrinking rate of Epsilon value when the
        round of test case selection grows.
    gamma: Gamma value to decide the proportion of impact from future execution
        result. Larger this value, more influence granted from the future.
    foremost_exploring_num: The fixed foremost number of test case selection set
        for exploration.
  """
  def __init__(self,
               name: Optional[str] = None,
               round_num: int=1000,
               search_depth: int = 3,
               eps: float = 0.3,
               beta: float = 0.1,
               gama: float = 0.7,
               foremost_exploring_num: int = 50):
    super().__init__(name, search_depth)
    self._is_under_trainin = False
    self._round_num = round_num
    self._eps = eps
    self._beta = beta
    self._gama = gama
    self._foremost_exploring_num = foremost_exploring_num
    self._log = logging.getLogger(self.__class__.__name__)

    # Key as execution path (state); value as previous calculated weighting.
    # https://developers.google.com/machine-learning/glossary#q-function
    self._qfunc = {}

  @property
  def round_num(self) -> int:
    """The number of round to interact with environment for training."""
    return self._round_num

  @property
  def foremost_exploring_num(self) -> int:
    """The fixed foremost number of test case selection set for exploration."""
    return self._foremost_exploring_num

  @property
  def eps(self) -> float:
    """Epsilon value of ε-Greedy Policy."""
    if self._is_under_trainin:
      return self._eps / (1 + self._round * self._beta)

    return 0

  def _next_execution_path(self, search_space: int,
                           new_test_case_name: str,
                           execution_path: Optional[Tuple[str, ...]] = None
                           ) -> Tuple[str, ...]:
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
               test_case_names: Optional[List[str]] = None) -> str:
    """Conducts exploration in selecting test case.

    Args:
      environment: Environment to explore.
      test_case_names: Test case name list for exploring process.

    Returns:
      The selected test case.
    """
    test_case_names = test_case_names or environment.available_actions_from_current_state()
    for test_case_name in random.sample(
        test_case_names, k=len(test_case_names)):
      for search_space in range(1, self._search_depth + 1):
        new_execute_path = self._next_execution_path(search_space,
                                                     test_case_name)
        if new_execute_path not in self.qtable.get(str(search_space), {}):
          self._log.info('Exploring untouched sequence with test case=%s',
                        test_case_name)
          return test_case_name

    test_case_name = random.choice(test_case_names)
    self._log.info('Exploring by random with test case=%s', test_case_name)
    return test_case_name

  def _exploit(self, environment: rl_protos.Environment,
               test_case_names: Optional[List[str]] = None) -> str:
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
      for search_space in range(1, self._search_depth + 1):
        new_execution_path = self._next_execution_path(search_space,
                                                       test_case_name)
        sub_qtable = self.qtable.get(str(search_space), {})
        test_case_exec_history = sub_qtable.get(new_execution_path, None)

        if not test_case_exec_history or test_case_exec_history.count == 0:
          weight += INIT_EPSILON_VALUE
          continue

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

      self._log.debug('\t%s with weighting=%.02f (count=%s)',
                     new_execution_path, weight, f'{count:,d}')
      weighting_data.append(weight)

    if self._is_under_trainin:
      return random.choices(test_case_name_list, weights=weighting_data)[0]
    else:
      return sorted(
          zip(test_case_name_list, weighting_data), key=lambda t: t[1])[-1][0]

  def _run_policy(self, environment: rl_protos.Environment) -> str:
    """Runs ε-Greedy Policy to select BCST test case."""
    p = random.random()
    test_case_names = environment.available_actions_from_current_state()
    if self._is_under_trainin:
      if self.round < self.foremost_exploring_num:
        selected_bcst_test_case = self._explore(environment)
      elif p < self.eps:
        selected_bcst_test_case = self._explore(environment)
      else:
        selected_bcst_test_case = self._exploit(environment)
    else:
      selected_bcst_test_case = self._exploit(environment)

    if selected_bcst_test_case:
      self._update_test_case_selection_stat(selected_bcst_test_case)

    return selected_bcst_test_case

  def fit(self, environment: rl_protos.Environment):
    """Conduct training from the given environment."""
    self._reset()
    self._is_under_trainin = True
    for _ in tqdm(range(self.round_num)):
      environment.reset()
      while not environment.is_done:
        result = self.play(environment)
        if result.reward > 0:
          self.feedback_reward(result.reward)

    self._is_under_trainin = False

  def play(self, environment: rl_protos.Environment):
    """Select test case from the given environment."""
    return environment.step(self._run_policy(environment))
