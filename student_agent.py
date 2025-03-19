# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.


#     return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
#     # You can submit this random agent to evaluate the performance of a purely random strategy.

# Load the trained Q-table
try:
    with open("q_table1.pkl", "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    q_table = None  # Handle missing file case

prev = 0

def get_action(obs):
    """Selects an action using the learned Q-table or falls back to random actions."""
    # if q_table is not None and obs in range(q_table.shape[0]):
    #     return np.argmax(q_table[obs])  # Choose best action

    next = random.choice([0, 1, 2, 3])  # Random fallback
    while prev == next:
        next = random.choice([0, 1, 2, 3])
    return next
