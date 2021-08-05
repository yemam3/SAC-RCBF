import random
import numpy as np

class ReplayMemory:

    def __init__(self, capacity, seed, store_t=False):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.store_t = store_t

    def push(self, state, action, reward, next_state, done, t=None, next_t=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if self.store_t:
            self.buffer[self.position] = (state, action, reward, next_state, done, t, next_t)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def batch_push(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, t_batch=None, next_t_batch=None):

        for i in range(state_batch.shape[0]):  # TODO: Optimize This
            self.push(state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i], done_batch[i], t_batch, next_t_batch)  # Append transition to memory

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        if self.store_t:
            state, action, reward, next_state, done, t, next_t = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done, t, next_t

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
