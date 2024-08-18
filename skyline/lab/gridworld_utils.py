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

"""Utility to keep functions for convenience of demonstration of this lab."""

from skyline.lab import gridworld_env


def print_values(V, env: gridworld_env.GridWorldEnvironment):
  """Prints the value of all states in GridWorld.

  Args:
    V: Learned value function with key as state; value as value.
    env: GridWorldEnvironment object.
  """
  for i in range(env.rows):
    print("---------------------------")
    for j in range(env.cols):
      state = gridworld_env.GridState(i, j)
      if state in env.rewards:
        reward = env.rewards[state]
        if reward >= 0:
          print(" %.2f|" % env.rewards[state], end="")
        else:
          print("%.2f|" % env.rewards[state], end="")
      else:
        v = V.get(state, 0)
        if v >= 0:
          print(" %.2f|" % v, end="")
        else:
          print("%.2f|" % v, end="")  # -ve sign takes up an extra space
    print("")


def print_policy(P, env: gridworld_env.GridWorldEnvironment):
  """Prints the policy of all states in GridWorld.

  Args:
    P: Learned policy with key as state; value as action.
    env: GridWorldEnvironment object.
  """
  for i in range(env.rows):
    print("---------------------------")
    for j in range(env.cols):
      state = gridworld_env.GridState(i, j)
      if state in env.rewards:
        print("  ?  |", end="")
        continue

      a = P.get(state, 'x')
      print("  %s  |" % a, end="")
    print("")
