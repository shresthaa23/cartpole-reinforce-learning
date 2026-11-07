# cartpole-reinforce-learning
Training an artificial agent to control Gymnasium's `CartPole-v1` using Deep Q-Learning. CartPole is a pole attached by a joint to a cart, and the cart moves along a frictionless track. The pendulum is placed upward and the goal is to move the cart left/right to balance the pole upward for as long as possible. 

The code trains the agent on 1000 episodes, using a deep neural network to estimate rewards rather than a Q-table. Afterwards, the neural network is evaluated on a new set of 100 fresh episodes and the results are printed into the `results/` folder with WHATEVER. THE VISUALZATIONS AND BLAH BLAH.


## To Run

```
uv venv
uv pip install -r requirements.txt
uv run <path-to-file>/project2.py
```

## Setup

| | |
|---|---|
| Python | 3.12.3 |
| gymnasium | 1.4.0 |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |
| keras | 3.15.1 | 
| tensorflow | 2.21.0 |
| Hardware | Intel Core i7-1360P laptop CPU (16 threads), 7.6 GB RAM, Ubuntu on WSL2, no GPU |
| Runtime | about 8 min end-to-end |

Other downloaded items in the requirements are automatically setup when the above packages are installed properly.

## Given Hyperparameters

| replay_capacity | gamma | epsilon | epsilon_min | epsilon_decay | num_episodes | evaluation_episodes | learning_rate | batch_size | training_frequency | target_update_frequency |
|---|---|---|---|---|---|---|---|---|---|---|
| 20000 | 0.95 | 1.0 | 0.01 | 0.995 | 1000 | 100 | 0.001 | 32 | 4 | 200 |

Epsilon is decayed once after each completed episode.

## Random Seeds

- Training environment seed - 123
- Evaluation environment seed - 420


## Evaluation Results

Online model on 100 fresh episodes:

| Mean return | Win rate | Draw rate | Loss rate |
|---|---|---|---|
| -0.0677 | 42.22% | 8.79% | 48.99% |

## Results / Graph Explanation