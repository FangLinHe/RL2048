# RL2048

This project aims to implement a Python version of [2048 game](https://play2048.co/) and train an RL model to play the game.

## Implement 2048 Game with Python

To implement the game, we first need to use some module to implement the game interface. Not doing much of research, I found the [Python module `pygame`](https://www.pygame.org/news) easy to use, so I chose it as the starting point.

### Algorithms

**Move up**
```
2 0 0 0               4 0 0 0
2 0 0 0  move up      4 0 0 0
0 0 0 0 ------------> 0 0 0 0
4 0 0 0               0 0 0 0
```

* Step 1: Merge values with above one if values are the same
  ```
  2 0 0 0               4 0 0 0
  2 0 0 0  move up      0 0 0 0
  0 0 0 0 ------------> 0 0 0 0
  4 0 0 0               4 0 0 0
  ```

* Step 2: Move all the values up

**Move down**
```
2 0 0 0               0 0 0 0
0 0 0 0  move down    0 0 0 0
2 0 0 0 ------------> 4 0 0 0
4 0 0 0               4 0 0 0
```

**Move left**
```
2 2 0 4               4 4 0 0
0 0 0 0  move down    0 0 0 0
0 0 0 0 ------------> 0 0 0 0
0 0 0 0               0 0 0 0
```

**Move right**
```
2 2 4 0               0 0 4 4
0 0 0 0  move down    0 0 0 0
0 0 0 0 ------------> 0 0 0 0
0 0 0 0               0 0 0 0
```

## Implement RL algorithms

