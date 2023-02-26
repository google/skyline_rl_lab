## skyline_rl_lab
We are going to implement and make experiment on RL algorithms in this repo to facilitate our research and tutoring purposes. Below we are going to explain how use this repo with a simple example.

## Environment
For RL to work, we need an environment to interact with. From Skyline lab, We can list supported environment as below:
```python
>>> from skyline import lab
>>> lab.list_env()
===== GridWorld =====
This is a environment to show case of Skyline lab. The environment is a grid world where you can move up, down, right and leftif you don't encounter obstacle. When you obtain the reward (-1, 1, 2), the game is over. You can use env.info() to learn more.
```
Then We use function `make` to obtain the desired environment. e.g.
```python
>>> grid_env = lab.make(lab.Env.GridWorld)
>>> grid_env.info()
- environment is a grid world
- x means you can't go there
- s means start position
- number means reward at that state
===========
.  .  .  1
.  x  . -1
.  .  .  x
s  x  .  2
===========
```
For avaiable actions in the environment, you can know by:
```python
>>> grid_env.available_actions()
['U', 'D', 'L', 'R']
```
To get the current state of an environment:
```python
>>> grid_env.current_state
GridState(i=3, j=0)
```
The starting position (`s`) is of axis as (3, 1) in this case.

Let's take a action and check how the state changes in the environment:
```python
>>> grid_env.step('U')  # Take action 'Up'
ActionResult(action='U', state=GridState(i=2, j=0), reward=0, is_done=False, is_truncated=False, info=None)

>>> grid_env.current_state  # Get current state
GridState(i=2, j=0)
```

After taking action `U`, we expect the i-axis to move up from 3->2 and we can confirm it from the return action result. Let's reset the environment by calling method <font color='blue'>reset</font> which will bring the state of environment back to intial state GridState(i=3, j=0):
```python
>>> grid_env.reset()
>>> grid_env.current_state
GridState(i=3, j=0)
```

## Experiments of RL algorithms
Here we are going to test some well-known RL algorithms and demonstrate the
usage of this lab. All RL methods we are going to implement must implement proto
<font color='blue'>**RLAlgorithmProto**</font> in
[`rl_protos.py`](skyline/lab/rl_protos.p). We will take a look at some
implementation of RL methods to know the usage of them.

<a id='monte_carlo_method'></a>
### Monte Carlo Method
<b>In this method, we simply simulate many trajectories</b> (<font color='brown'>decision processes</font>)<b>, and calculate the average returns.</b> ([wiki page](https://en.wikiversity.org/wiki/Reinforcement_Learning#Monte_Carlo_policy_evaluation))

We implement this algorithm in [`monte_carlo.py`](skyline/lab/alg/monte_carlo.py). Below code snippet will initialize this RL method:
```python
>>> from skyline.lab.alg import monte_carlo
>>> mc_alg = monte_carlo.MonteCarlo()
```

Each RL method object will support method `fit` to learn from the given
environment object. For example:
```python
>>> mc_alg.fit(grid_env)
```

Then we can leverage utility [`gridworld_utils.py`](skyline/lab/gridworld_utils.py) to print out the learned RL knowledge. Below will show the learned [value function](https://en.wikipedia.org/wiki/Reinforcement_learning#Value_function) from the Monte Carlo method:
```python
>>> gridworld_utils.print_values(mc_alg._state_2_value, grid_env)
---------------------------
 1.18| 1.30| 1.46| 1.00|
---------------------------
 1.31| 0.00| 1.62|-1.00|
---------------------------
 1.46| 1.62| 1.80| 0.00|
---------------------------
 1.31| 0.00| 2.00| 2.00|
```

Then let's check the learned policy:
```python
>>> gridworld_utils.print_policy(mc_alg._policy, grid_env)
---------------------------
  D  |  R  |  D  |  ?  |
---------------------------
  D  |  x  |  D  |  ?  |
---------------------------
  R  |  R  |  D  |  x  |
---------------------------
  U  |  x  |  R  |  ?  |
```

Finally, we could use trained Monte Carlo method object to interact with
environment. Below is the sample code for reference:
```python
# Play game util done
grid_env.reset()

print(f'Begin state={grid_env.current_state}')
step_count = 0
while not grid_env.is_done:
    result = mc_alg.play(grid_env)
    step_count += 1
    print(result)

print(f'Final reward={result.reward} with {step_count} step(s)')
```

The execution would look like:
```shell
Begin state=GridState(i=3, j=0)
ActionResult(action='U', state=GridState(i=2, j=0), reward=0, is_done=False, is_truncated=False, info=None)
ActionResult(action='R', state=GridState(i=2, j=1), reward=0, is_done=False, is_truncated=False, info=None)
ActionResult(action='R', state=GridState(i=2, j=2), reward=0, is_done=False, is_truncated=False, info=None)
ActionResult(action='D', state=GridState(i=3, j=2), reward=0, is_done=False, is_truncated=False, info=None)
ActionResult(action='R', state=GridState(i=3, j=3), reward=2, is_done=True, is_truncated=False, info=None)
Final reward=2 with 5 step(s)
```

### Random Method
This method takes random action(s) in the given environment. It is often used as a baseline to evaluate other RL methods. The code below will instantiate a random RL method:

```python
from skyline.lab.alg import random_rl

random_alg = random_rl.RandomRL()
```

Random RL method won't require training at all. So if you call method `fit` of
`random_alg`, it will return immediately:

```
# Training
random_alg.fit(grid_env)
```

Since this is a random process, each time you play the game will have difference result:
```python
# Play game util done
grid_env.reset()

print(f'Begin state={grid_env.current_state}')
step_count = 0
while not grid_env.is_done:
    result = random_alg.play(grid_env)
    step_count += 1
    print(result)
print(f'Final reward={result.reward} with {step_count} step(s)')
```

Below is one execution example:
```
Begin state=GridState(i=3, j=0)
ActionResult(action='U', state=GridState(i=2, j=0), reward=0, is_done=False, is_truncated=False, info=None)
...
ActionResult(action='R', state=GridState(i=0, j=3), reward=1, is_done=True, is_truncated=False, info=None)
Final reward=1 with 16 step(s)
```

From the result above, the random RL method took more steps and not guarantee to obtain the best reward, Therefore, it is obvious that the [**Monte Carlo method**](#monte_carlo_method) performs much better than the Random RL method!

## How to rank RL methods
Before we start introducing how score board work, we need to understand
[**RLExaminer**](#rlexaminer) first. Basically, scoreboard is a design to help you rank the
different RL methods.

<a id='rlexaminer'></a>
### RLExaminer
Every environment can have more than one examiner to calculate the score of RL method. Each examiner may have its own aspect to evaluate the RL method (time, reward etc.). Let's check one used to calculate the average reward of grid environment:

```python
# This examiner considers both reward and number of steps.
examiner = gridworld_env.GridWorldExaminer()
```
Then, what's score of Monte Carlo Method:
```python
# Monte Carlo will get reward 2 by taking 5 steps.
# So the score will be reward / steps: 2 / 5 = 0.4
examiner.score(mc_alg, grid_env)
```

[Monte Carlo method](#monte_carlo_method) got score 0.4. Let's check another RL method Random Method:

```python
# The number of steps required by random RL method is unknown.
# Also the best reward is not guaranteed. So the score here will be random.
examiner.score(random_alg, grid_env)
```
Random RL method often got scores to be less than Monte Carlo method.

### Scoreboard
Scoreboard literally calculate the scores of given RL methods according to the specific examiner and the rank those RL methods accordingly:

```python
score_board = lab.Scoreboard()
sorted_scores  = score_board.rank(
    examiner=examiner, env=grid_env, rl_methods=[random_alg, mc_alg])
```
Below output will be produced:
```
+-------+------------+---------------------+
| Rank. |  RL Name   |        Score        |
+-------+------------+---------------------+
|   1   | MonteCarlo |         0.4         |
|   2   |  RandomRL  | 0.13333333333333333 |
+-------+------------+---------------------+
```

## Resources
* [Tensorflow - Introduction to RL and Deep Q Networks](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
* [Udemy - Artificial Intelligence: Reinforcement Learning in Python](https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/)
