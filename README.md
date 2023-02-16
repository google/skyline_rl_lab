## skyline_rl_lab
We are going to implement and make experiment on RL algorithms in this repo to facilitate our research and tutoring purposes. Below we are going to explain how use this repo with a simple example.

## Environment
For RL to work, we need environment to interact with. From Skyline lab, We can list supported environment as below:
```python
>>> from skyline import lab
>>> lab.list_env()
===== GridWorld =====
This is a environment to show case of Skyline lab. The environment is a grid world where you can move up, down, right and leftif you don't encounter obstacle. When you obtain the reward (-1, 1, 2), the game is over. You can use env.info() to learn more.
```
Then We use function make to obtain the desired environment. e.g.
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
To get the current state of environment:
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

## Resources
* [Tensorflow - Introduction to RL and Deep Q Networks](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
* [Udemy - Artificial Intelligence: Reinforcement Learning in Python](https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/)
