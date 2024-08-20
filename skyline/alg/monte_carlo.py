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

"""This module implement monte carlo algorithm to conduct RL."""
import collections
import dataclasses
import numpy as np
from skyline import alg
from skyline import rl_protos
from tqdm import tqdm
from typing import Any, Optional, Sequence


_DEFAULTDICT = collections.defaultdict


@dataclasses.dataclass(frozen=True)
class ExploringCollection:
  """Dataclass to hold collected information why exploring environment."""
  states: Sequence[Any]
  actions: Sequence[Any]
  rewards: Sequence[float]


class MonteCarlo(alg.RLAlgorithm):
  """Monte Carlo Method."""

  def __init__(
      self, name: Optional[str] = None,
      round_num: int = 10000, gamma: float = 0.9,
      default_q_value: float = 0,
      max_tcs_length: int = 4):
    super().__init__(name or self.__class__.__name__)
    self._round = round_num
    self._gamma = gamma
    self._default_q_value = default_q_value
    self._max_tcs_length = max_tcs_length
    self._state_2_value = {}
    self._q = {}
    self._policy = {}  # Key as state; value as action
    self._deltas = []
    self._sample_counts = {}
    self._deltas = []
    # Key as state; value as hit count
    self._short_team_memory = _DEFAULTDICT(int)

  def _reset(self):
    self._state_2_value = {}
    self._q = {}
    self._policy = {}  # Key as state; value as action
    self._deltas = []
    self._sample_counts = {}
    self._deltas = []

  @property
  def name(self) -> str:
    """Gets name of RL method."""
    return self._name

  def fit(self, environment: rl_protos.Environment):
    """Conducts training by given environment."""
    # initialize V(s) and returns
    self._reset()
    states = environment.available_states()
    actions = environment.available_actions()
    for s in states:
      environment.reset()
      environment.set_state(s)
      self._policy[s] = environment.random_action()
      self._state_2_value[s] = 0

    # Initialize Q(s,a) and returns
    self._sample_counts = {}
    for s in states:
      environment.reset()
      environment.set_state(s)
      if not environment.is_done:
        self._q[s] = {}
        self._sample_counts[s] = {}
        for a in actions:
          self._q[s][a] = 0
          self._sample_counts[s][a] = 0

    # repeat until convergence
    for _ in tqdm(range(self._round)):
      # generate an episode using pi
      biggest_change = 0
      states, actions, rewards = self._play_game(environment)

      # create a list of only state-action pairs for lookup
      states_actions = list(zip(states, actions))

      T = len(states)
      G = 0
      for t in range(T - 2, -1, -1):
        # retrieve current, s, a, r tuple
        s = states[t]
        a = actions[t]

        # update G
        G = rewards[t+1] + self._gamma * G

        # Check if we have already seen (s, a) ("first-visit")
        if (s, a) not in states_actions[:t]:
          old_q = self._q[s][a]
          self._sample_counts[s][a] += 1
          lr = 1 / self._sample_counts[s][a]
          self._q[s][a] = old_q + lr * (G - old_q)

          # update policy
          best_action = self.max_dict(self._q[s])[0]
          if best_action in environment.available_actions(s):
            self._policy[s] = best_action

          # update delta
          biggest_change = max(biggest_change, np.abs(old_q - self._q[s][a]))

        self._deltas.append(biggest_change)

      for s, _ in self._q.items():
        self._state_2_value[s] = self.max_dict(self._q[s])[1]

  def play(self, environment: rl_protos.Environment):
    """Plays in the given environment."""
    result = environment.step(
        self._policy[environment.current_state])
    return result

  def _play_game(self, environment: rl_protos.Environment, max_steps=20):
    # reset game to start at a random position
    # we need to do this if we have a deterministic policy
    # we would never end up at certain states,
    # but we still want to measure their value
    # this is called the "exploring starts" method
    start_states = environment.available_states()
    start_idx = np.random.choice(len(start_states))
    environment.set_state(start_states[start_idx])
    s = environment.current_state
    a = environment.random_action(s)

    states = [s]
    actions = [a]
    rewards = [0]

    for _ in range(max_steps):
      result = environment.step(a)
      s = environment.current_state

      # update states and rewards lists
      states.append(s)
      rewards.append(result.reward)

      if environment.is_done:
        break

      a = self._policy.get(s, None)
      if not a:
        random_action = environment.random_action()
        self._policy[s] = random_action
        a = random_action

      actions.append(a)

    # we want to return:
    # states  = [s(0), s(1), ..., s(T-1), s(T)]
    # actions = [a(0), a(1), ..., a(T-1),     ]
    # rewards = [   0, R(1), ..., R(T-1), R(T)]

    return states, actions, rewards
