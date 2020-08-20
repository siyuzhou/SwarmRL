import numpy as np


class RolloutBuffer:
    def __init__(self, n=None, gamma=0.95, num_state_inputs=1):
        self.n = n
        self.gamma = gamma
        self._num_state_inputs = num_state_inputs

        if num_state_inputs > 1:
            self.state_buffer = [[] for _ in range(num_state_inputs)]
        else:
            self.state_buffer = []

        self.action_buffer = []
        self.reward_buffer = []
        self.reward_to_go_buffer = []

    def __len__(self):
        return len(self.action_buffer)

    def is_full(self):
        return (self.n is not None) and len(self) >= self.n

    def add_transition(self, state, action, reward):
        if self.is_full():
            raise BufferFullError('Buffer is full')

        if self._num_state_inputs > 1:
            for buffer, s in zip(self.state_buffer, state):
                buffer.append(s)
        else:
            self.state_buffer.append(state)

        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state_value):
        v = next_state_value

        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v = r + self.gamma * v
            discounted_r.append(v)

        self.reward_to_go_buffer.extend(discounted_r[::-1])
        self.reward_buffer.clear()

    def get_buffer(self):
        valid_len = len(self.reward_to_go_buffer)

        if self._num_state_inputs > 1:
            states = [np.array(s[:valid_len], dtype=np.float32) for s in self.state_buffer]
        else:
            states = np.array(self.state_buffer[:valid_len], dtype=np.float32)

        actions = np.array(self.action_buffer[:valid_len], dtype=np.float32)
        rewards_to_go = np.expand_dims(self.reward_to_go_buffer[:valid_len], -1).astype(np.float32)

        return states, actions, rewards_to_go

    def clear(self):
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.reward_to_go_buffer.clear()


class BufferFullError(Exception):
    def __init__(self, message):
        super().__init__(message)
