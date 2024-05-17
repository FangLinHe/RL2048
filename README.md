[![ci](https://github.com/FangLinHe/RL2048/actions/workflows/ci.yaml/badge.svg)](https://github.com/FangLinHe/RL2048/actions/workflows/ci.yaml)

# RL2048

This project aims to implement a Python version of [2048 game](https://play2048.co/) and train an RL model to play the game.

## Install and play

```
pip install git+https://github.com/FangLinHe/RL2048
rl2048-play-gui  # to play 2048 manually
rl2048-dqn-eval-gui
```
or a specific commit / tag like:
```
pip install git+https://github.com/FangLinHe/RL2048@<COMMIT_OR_TAG>
```

## Implement 2048 Game with Python

To implement the game, we first need to use some module to implement the game interface. Not doing much of research, I found the [Python module `pygame`](https://www.pygame.org/news) easy to use, so I chose it as the starting point.

### Algorithms - Version 1

This version is done on May, 07, 2024. There is no animation in this version.

First, 4*4 rectangles are pre-generated using `pygame.rect` in initializer of `TilePlotter`.
They are at the fixed locations, and their colors are then decided based on the values on the grids.
Note that this actually caused the problem if we want to make animations, so I decided to refactor it in the next version.

The class `Tile` stores the size of the board and the values in each grid. It provides the method to randomly restart the board by generating two values on random grids, each one is either 2 or 4.

The class `GameEngine` provides the methods to move grids in four directions: up, down, left, and right, as well as the method to generate a new value on an empty grid randomly, its value is also either 2 or 4. When the grids are moved, it first "merges" values if they have the same values as their neighbors, then all the grids are moved towards the specified direction to fill the gaps between grids. Then increased score and if the movement succeeded are returned by the function. Internally, the accumulated score is also updated.

`TilePlotter` provides the `plot` function to render rectangles and texts of all grids based on the plot properties specified in the initializer, including grid size, space, and radius of rectangles.

### Algorithms - Version 2

As I found that without animation of moving grids, it's hard to see what happens, so I want to add animations. I need to record the information of starting location and destination, and update the rectangle location based on current step. For a simplified example, if we want to move a point in `s=10` steps from location `a=-4` to `b=5`, the trajectory is [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5], which can be obtained by `a + (b - a) * (t / (s - 1))` for `t = 0:9`. We might want to add fancier animations in the future, like merging grids, slightly bigger rectangles when they reach the destination, etc.

In order to do that, we need to refactor how the rectangles are pre-generated. First, we need to render all rectangles with the same color as the background. Second, we need to dynamically generate / remove rectangles when they are generated / merged, and record the moving information to make animations.

## Implement RL algorithms

* State: grid values, i.e. for a 4x4 game board, we can get a vector of their values by `g[0][0], g[0][1], g[0][2], g[0][3], g[1][0], ..., g[3][3]`.
* Reward: we can get the current score and score obtained after movement from the game engine.
* Game over state: we can know if the game is over from the game engine.
* Actions: `[MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]`

### Deep Q-Learning

Suppose there are 16 grids, each grid has values limited to [0, 2, 4, ..., 32768], so there are in total `16^16 = 18446744073709551616` possible states. That's impossible to use tabular Q-Learning, as it would require to build a `(16^16) x 4` Q-table, which definitely doesn't fit in the memory.

On the other hand, **Deep Q-Learning** is a good alternative. It utilizes a (deep) neural network, which takes the current state as the input and predicts Q-values for each action. Specifically in our case, it takes input features from status of 2048 game board and predicts 4 Q-values for each direction as the output. Simply using a few linear layers with ReLU as activation layers is already enough to reach 2048 occasionally in our experiments.

There are at least two ways to represent the input: use the raw values from 16 grids or encode it with one-hot. For example, if we suppose the maximum reachable grid value is 32768, we can encode it as a vector of size 16, one and only one value is 1. If value is 0 (means no grid in my setup), the 0-th element is 1, and otherwise we take log2 as the location of 1. I experimented with both approaches, but with raw values (with or without normalization), the loss doesn't converge.

Regarding the reward, of course the scores we obtained in each step are the most intuitive solution. However, considering some actions are not allowed in some case, we should also give this information to the reward function. I give it a big penalty of -256. For game over state, I still count the score that it gets, but a penalty -4096 is also added to reward. Somehow the model doesn't converge fast enough if I use the raw value of the reward. Thus, I divide the reward by 256 for the network to compute the Q-value.

We used epsilon-greedy algorithm during training to trade-off exploration and exploitation. Epsilon values exponentially decrease from initial values to end values, inspired by [PyTorch's tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). After model is optimized one step, we update the model by weighted average (i.e., 0.005 taken from optimized state, 0.995 taken from original state). From our experiment, it does help a lot to train the network and reach better performance.

## Build package

```
# Build package
make build

# Run tests and show test coverage
make test

# Check formats
make lint

# Check if imports are sorted
make isort
```

## TODO

- [ ] Write tests and run coverage
  * Just got started!
- [ ] Write GitHub workflows
- [x] <del>Make sure scripts can be executed with only library installed</del> (Tested on Windows!)
- [ ] Make a report
