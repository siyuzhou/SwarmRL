import random
from collections import deque
import numpy as np


class NStepRolloutBuffer:
    def __init__(self, n, capacity=1e5, gamma=0.95, num_states=1):
        self.n = n # Max number of rollout per trajectory
        self.capacity = int(capacity)
        self.gamma = gamma

        self._num_states = num_states

        self.state_buffer = []
        self.action_buffer = []
        self.log_prob_buffer = []
        self.reward_buffer = []

        self.memory = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.memory)

    def path_end(self):
        return len(self.reward_buffer) >= self.n

    def add_transition(self, state, action, reward, log_prob):
        if self.path_end():
            raise PathEndError(f'Failed to add transition beyond {self.n} steps')

        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.log_prob_buffer.append(log_prob)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state):
        v = 0.

        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v = r + self.gamma * v
            discounted_r.append(v)

        discounted_r = discounted_r[::-1]

        memory_batch = zip(self.state_buffer, self.action_buffer, discounted_r, self.log_prob_buffer, [next_state]*self.n)
        self.memory.extend(memory_batch)

        self.clear_cache()
        
    def get_buffer(self, batch_size):
        if batch_size > len(self.memory):
            batch = self.memory
        else:
            batch = random.sample(self.memory, batch_size)

        return self._to_numpy(batch)

    def _to_numpy(self, experiences):
        states, actions, rewards_to_go, log_probs, next_states = zip(*experiences)

        if self._num_states > 1:
            states = [np.array(state, dtype=np.float32) for state in zip(*states)]
            next_states = [np.array(next_state, dtype=np.float32) for next_state in zip(*next_states)]
        else:
            states = np.array(states, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)

        actions = np.array(actions, dtype=np.float32)
        rewards_to_go = np.expand_dims(rewards_to_go, -1).astype(np.float32)
        log_probs = np.array(log_probs, dtype=np.float32)

        return states, actions, rewards_to_go, log_probs, next_states

    def clear_cache(self):
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.log_prob_buffer.clear()
        self.reward_buffer.clear()


class PathEndError(Exception):
    def __init__(self, message):
        super().__init__(message)
