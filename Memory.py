import random
from collections import namedtuple, deque

# Creating Transitions between steps
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    # Initializing replay buffer with fixed capacity to store recent data
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # deque = double-ended queue: add/take elements both end

    # Saving new transitions
    def push(self, *args):
        self.memory.append(Transition(*args))

    # Taking random samples
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)  # for reaching the length of memory easier

