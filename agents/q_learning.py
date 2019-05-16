from tqdm import tqdm
import numpy as np
from collections import defaultdict


class QLearning:
    def __init__(self, env, discount=0.99, lr=0.5, epsilon=0.1):
        self.env = env
        self.discount = discount
        self.lr = lr
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(env.action_space)))

    def train(self, episodes=500, render=False):
        for i in tqdm(range(episodes)):
            state = self.env.reset()
            if render: self.env.render()
            done = False
            while not done:
                # epsilon greedy
                if np.random.rand() <= self.epsilon:
                    action = np.random.choice(self.env.action_space)
                else:
                    action = np.argmax(self.Q[tuple(state)])
                n_state, reward, done = self.env.step(action)
                q_vals = self.Q[tuple(state)]
                q_vals[action] = q_vals[action] + \
                    self.lr * (reward + self.discount * np.max(self.Q[tuple(n_state)]) - q_vals[action])
                state = n_state
                if render: self.env.render()

    def test(self, render=True):
        while True:
            state = self.env.reset()
            if render: self.env.render()
            done = False
            while not done:
                action = np.argmax(self.Q[tuple(state)])
                state, _, done = self.env.step(action)
                if render: self.env.render()


class DoubleQLearning:
    def __init__(self):
        pass

    def train(self, render=False):
        """
        Implement here.
        """
        pass

    def test(self, render=True):
        pass
