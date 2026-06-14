from collections import defaultdict
import numpy as np
import gymnasium as gym

class DecoderAgent:
    def __init__(
            self, 
            env, 
            learning_rate: float, # how quickly q values update
            initial_epsilon: float, # starting exploration rate
            epsilon_decay: float, # how quickly to stop exploring
            final_epsilon: float, # min exploration rate
            discount_factor: float): # weight for future rewards
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_epsilon

    def get_state(self, observation):
        # Convert the observation to a tuple to use as a key in the Q-table
        decoder_time_left = tuple(observation["decoder_time_left"].astype(int))
        next_cost = int(observation["next_cost"][0])
        packets_remaining = int(observation["packets_remaining"][0])
        return (decoder_time_left, next_cost, packets_remaining)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best known action

    def update_q_table(self, state, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state][action])
