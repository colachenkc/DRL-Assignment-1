import numpy as np
import pickle
import random
import gymnasium as gym

# Initialize environment
env = gym.make("Taxi-v3")

# Hyperparameters
alpha = 0.5  # Faster learning initially
gamma = 0.99  # Prioritize long-term rewards
epsilon = 1.0  # Initial exploration
epsilon_min = 0.01  # Minimum exploration
epsilon_decay = 0.9995  # Faster decay initially
num_episodes = 100000  # More training

# Initialize Q-table
q_table = np.random.uniform(low=-1, high=1, size=(env.observation_space.n, env.action_space.n))

# Memory of bad states (to avoid repeating mistakes)
bad_state_memory = set()

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
            action = np.argmax(q_table[state])  # Exploit best-known action

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count += 1
        total_reward += reward

        # Store visited states
        visited_states.add(state)

        # Smarter reward shaping
        if reward == -10:  # Incorrect pickup/dropoff
            reward -= 30  
            bad_state_memory.add(state)  # Store mistake state
        elif reward == 50:  # Successful dropoff
            reward += 50  
        elif step_count > 75:  # Prevent excessive wandering
            reward -= 5  
        elif reward == -5:  # Hit obstacle
            reward -= 15  # Increase penalty
            bad_state_memory.add(state)  # Store bad states
        elif reward == -0.1:  # Movement penalty
            reward -= 0.5  # Reduce steps

        # Update Q-value
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state  # Move to the next state

        # Early stopping if excessive wandering
        if step_count > 200:  
            break

    # Adjust epsilon (faster decay at start)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    alpha = max(0.1, alpha * 0.995)

    # Print progress every 5000 episodes
    if episode % 5000 == 0:
        print(f"Episode {episode}/{num_episodes}, Steps: {step_count}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}, Alpha: {alpha:.4f}")

# Save trained Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training complete. Q-table saved as 'q_table.pkl'.")
