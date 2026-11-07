# cartpole-reinforce-learning
Training an artificial agent to control Gymnasium's `CartPole-v1` using Deep Q-Learning. CartPole is a pole attached by a joint to a cart, and the cart moves along a frictionless track. The pendulum is placed upward and the goal is to move the cart left/right to balance the pole upward for as long as possible. 

The code trains the agent on 1000 episodes, using a deep neural network to estimate rewards rather than a Q-table. Afterwards, the neural network is evaluated on a new set of 100 fresh episodes and the results are printed into the `results/` folder along with a graph depicting the learning and the scores on the final evaluation.

## To Run

```
uv venv
uv pip install -r requirements.txt
uv run src/cart_pole/project2.py
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
| Runtime | about 6-7 min end-to-end |

Other downloaded items in the requirements are automatically setup when the above packages are installed properly.

## Given Hyperparameters

| replay_capacity | gamma | epsilon | epsilon_min | epsilon_decay | num_episodes | evaluation_episodes | learning_rate | batch_size | training_frequency | target_update_frequency |
|---|---|---|---|---|---|---|---|---|---|---|
| 20000 | 0.95 | 1.0 | 0.01 | 0.995 | 1000 | 100 | 0.001 | 32 | 4 | 200 |

Epsilon is decayed once after each completed episode.

## Random Seeds

- Training environment seed - 123
- Evaluation environment seed - 420

## High Level Architecture

Built with TensorFlow + Keras:

- **Input** - 4 values - cart position, cart velocity, pole angle, pole angular velocity
- **Hidden** - 2 dense layers, 24 neurons each, ReLU activation
- **Output** - 2 values (q-value per action: left, right), linear activation 
- **Loss + Optimizer** - MSE, Adam optimizer with learning rate 0.001

## Reward Handling

In the training process, the reward on true terminal ending was replaced with -10 instead
of the original result. This was used in training, but native results stored still kept original
values. Truncation rewards were not changed.

## Evaluation Results

Frozen online model on 100 fresh episodes:

| Mean return | Min | Max | Reached 475? |
|---|---|---|---|
| 233.64 | 208 | 299 | No |

## Results / Graph Explanation
**Training (top panel):** The returns are relatively low for the first ~200 episodes of training. 
The epsilon remains high and the agent is exploring the cartpole at first. Between episodes 200-300, 
the reward climbs quickly to about 200-330 and then levels off around 120-160. It does spike again
at around 900 with some episodes hitting 500, before the average drops back down. 

**Evaluation (bottom panel):** The frozen policy is mostly consistent with all episodes scoring
in the 200s, with an average of 233.64. This is much lower than the 475 threshold with the given
hyperparameters.

**Why low score**
- **γ = .95.** This has a short planning horizon reward of only about 20 steps away. In  
  CartPole, while knowing very far ahead may be useless, knowing a bit farther is crucial as
  a slow drift or slowly growing angle can tilt the pole and cause failure later than the action 
  expected it to. This rate balances the short-term well but may not correct long-term drift.
- **DQN.** Deep Q learning is unstable and this online network changes very often at 4 steps, 
  and the TD is coming from itself still learning. There are a lot of dips and rises which is 
  seen in the graph. Also the policy evaluated is the final policy at 1000, not necessarily the 
  best one seen at all. 
- **Limited training.** 1000 episodes isn't a ton in the grand scheme of things. Also the 24-24
  network and hyperparameters could potentially be tuned to be more optimal. 

**Ways to improve**
- Larger discount like (γ = 0.99) could lengthen the horizon
- Lower learning rate (.0005) could minimize instability
- Implementing double-DQN to reduce overestimation of Q-values
- Running more episodes and different seeds to improve on different environments
