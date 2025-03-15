import numpy as np
import random
import gymnasium as gym
from collections import deque
import pickle

# Initialize environment
env = gym.make("Taxi-v3")

# Hyperparameters
alpha = 0.5  # Learning rate
gamma = 0.99  # Discount factor for long-term rewards
epsilon = 1.0  # Initial exploration probability
epsilon_min = 0.01  # Minimum exploration
epsilon_decay = 0.9995  # Exploration decay rate
num_episodes = 100000  # Number of training episodes
batch_size = 64  # Mini-batch size for experience replay

# Initialize Q-tables (Double Q-learning)
q_table1 = np.random.uniform(low=-1, high=1, size=(env.observation_space.n, env.action_space.n))
q_table2 = np.random.uniform(low=-1, high=1, size=(env.observation_space.n, env.action_space.n))

# Experience replay buffer
replay_buffer = deque(maxlen=10000)

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    visited_states = set()  # Track visited states

    while not done:
        # Epsilon-greedy strategy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            # Use Double Q-learning for action selection
            action = np.argmax(q_table1[state] + q_table2[state])  # Combined action selection

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count += 1
        total_reward += reward

        # Store the experience in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Sample a mini-batch from the replay buffer
        if len(replay_buffer) >= batch_size:
            mini_batch = random.sample(replay_buffer, batch_size)

            for state_batch, action_batch, reward_batch, next_state_batch, done_batch in mini_batch:
                # Update Q-values using Double Q-learning
                if random.uniform(0, 1) < 0.5:
                    # Use q_table1 for action selection, q_table2 for Q-value update
                    best_next_action = np.argmax(q_table1[next_state_batch])
                    q_table1[state_batch, action_batch] += alpha * (
                        reward_batch + gamma * q_table2[next_state_batch, best_next_action] - q_table1[state_batch, action_batch]
                    )
                else:
                    # Use q_table2 for action selection, q_table1 for Q-value update
                    best_next_action = np.argmax(q_table2[next_state_batch])
                    q_table2[state_batch, action_batch] += alpha * (
                        reward_batch + gamma * q_table1[next_state_batch, best_next_action] - q_table2[state_batch, action_batch]
                    )

        state = next_state  # Move to the next state

        # Early stopping if excessive wandering
        if step_count > 200:  
            break

    # Adjust epsilon (faster decay at start)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    alpha = max(0.1, alpha * 0.995)  # Decay learning rate

    # Print progress every 5000 episodes
    if episode % 5000 == 0:
        print(f"Episode {episode}/{num_episodes}, Steps: {step_count}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}, Alpha: {alpha:.4f}")

# Save trained Q-tables
with open("q_table1.pkl", "wb") as f1, open("q_table2.pkl", "wb") as f2:
    pickle.dump(q_table1, f1)
    pickle.dump(q_table2, f2)

print("Training complete. Q-tables saved as 'q_table1.pkl' and 'q_table2.pkl'.")
