import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
import collections
import random
import os
import time

# hyperparams 
REPLAY_CAP = 20_000
GAMMA = .95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
NUM_EPISODES = 1_000
EVAL_EPISODES = 100
BATCH_SIZE = 32
TRAIN_FREQ = 4
TARGET_UPDATE_FREQ = 200

TRAIN_SEED = 123
EVAL_SEED = 420
INPUT_SHAPE = (4,)

SOLVED_THRESHOLD = 475

# setup environment & variables
env = gym.make("CartPole-v1")
state, info = env.reset(seed=TRAIN_SEED)  
rng = np.random.default_rng(TRAIN_SEED)
keras.utils.set_random_seed(TRAIN_SEED)
buffer = collections.deque(maxlen=REPLAY_CAP)
step_count = 0
gradient_count = 0
epsilon = EPSILON
episode_return = 0
episode_returns = []
batch_prints = []

# setup models
def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=INPUT_SHAPE),
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dense(2, activation="linear")
    ])
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model

# fixing batches of desired size
def fix_batches(buffer):
    batch = random.sample(buffer, BATCH_SIZE)
    states, actions, rewards, next_states, terminated, truncated = zip(*batch)
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminated, dtype=np.float32), np.array(truncated)

# choosing random action or best action
def choose_action(online, state, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(2))

    q_values = online(state[None, :]).numpy()
    return int(np.argmax(q_values))

# creating models
online = build_model()
target = build_model()
target.set_weights(online.get_weights())


############################### TRAINING ###############################
train_start = time.time()
for episode in range(NUM_EPISODES):
    done = False
    episode_return = 0

    while not done:
        step_count += 1
        action = choose_action(online, state, epsilon)
        obs, reward, term, trunc, info = env.step(action)
        episode_return += reward                                            # store actual rewards, not training ones

        training_reward = -10.0 if term else reward                         # changing reward scheme    
        buffer.append((state, action, training_reward, obs, term, trunc))

        if step_count % TRAIN_FREQ == 0 and len(buffer) >= BATCH_SIZE:      # mini batch update 
            states, actions, rewards, next_states, terminated, truncated = fix_batches(buffer)

            target_q = target(next_states).numpy()
            best_target = np.argmax(target_q, axis=1)
            y = rewards + GAMMA * (1 - terminated) * target_q[np.arange(BATCH_SIZE), best_target]

            online_q = online(states).numpy()
            online_q[np.arange(BATCH_SIZE), actions] = y
            loss = online.train_on_batch(states, online_q)

            gradient_count += 1
            if gradient_count % TARGET_UPDATE_FREQ == 0:
                target.set_weights(online.get_weights())

        state = obs
        if term or trunc:
            done = True

    state, _ = env.reset()
    episode_returns.append(episode_return)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # printing progress every 50
    if (episode + 1) % 50 == 0:
        statement = f"Episode {episode + 1:4d} | epsilon: {epsilon:.3f} | mean return (last 50): {np.mean(episode_returns[-50:]):6.2f}"
        batch_prints.append(statement)
        print(statement)

env.close()
train_time = time.time() - train_start


############################### EVALUATION ############################### 
eval_env = gym.make("CartPole-v1")
state, info = eval_env.reset(seed=EVAL_SEED)
all_rewards = []

for episode in range(EVAL_EPISODES):
    done = False
    total_reward = 0

    while not done:
        q_values = online(state[None, :]).numpy()
        action = int(np.argmax(q_values))
        obs, reward, term, trunc, _ = eval_env.step(action)
        total_reward += reward
        state = obs
        done = term or trunc

    all_rewards.append(total_reward)
    state, _ = eval_env.reset()

eval_env.close()


############################### RESULTS ###############################
eval_mean = np.mean(all_rewards)

results = (
    "Evaluation: frozen greedy policy (epsilon = 0)\n"
    f"Training seed: {TRAIN_SEED} | Eval seed: {EVAL_SEED}\n"
    f"Training episodes: {len(episode_returns)} | Training time: {train_time:.0f} s\n"
    f"Eval episodes: {EVAL_EPISODES}\n"
    f"Mean return: {eval_mean:.2f}\n"
    f"Min / max return: {min(all_rewards):.0f} / {max(all_rewards):.0f}\n"
    f"Reached {SOLVED_THRESHOLD}: {'Yes' if eval_mean >= SOLVED_THRESHOLD else 'No'}\n"
)
results += "\nTraining progress (every 50 episodes):\n" + "\n".join(batch_prints) + "\n"

print("\n" + results)

os.makedirs("results/", exist_ok=True)
with open("results/results.txt", "w") as f:
    f.write(results)


############################### GRAPH ###############################
window = 50
moving_average = np.convolve(episode_returns, np.ones(window) / window, mode="valid")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# training progress
ax1.plot(episode_returns, alpha=0.3, label="Episode return")
ax1.plot(np.arange(window - 1, len(episode_returns)), moving_average, label=f"Average (last {window} episodes)")
ax1.axhline(SOLVED_THRESHOLD, color="green", linestyle="--", label="Solved (475)")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Return")
ax1.set_title("CartPole DQN Training Progress")
ax1.legend()
ax1.grid(True)

# greedy evaluation
ax2.bar(range(1, EVAL_EPISODES + 1), all_rewards)
ax2.axhline(eval_mean, color="black", label=f"Mean ({eval_mean:.1f})")
ax2.axhline(SOLVED_THRESHOLD, color="green", linestyle="--", label="Solved (475)")
ax2.set_xlabel("Eval episode")
ax2.set_ylabel("Return")
ax2.set_title("Greedy Evaluation (100 episodes)")
ax2.legend()
ax2.grid(True)

fig.tight_layout()
fig.savefig("results/training_graph.png")
plt.close(fig)
