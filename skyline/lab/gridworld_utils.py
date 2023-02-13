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
      v = V.get(gridworld_env.GridState(i, j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
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
      a = P.get(gridworld_env.GridState(i, j), ' ')
      print("  %s  |" % a, end="")
    print("")
