from tqdm import tqdm
import numpy as np
from collections import defaultdict


class MonteCarloOnPolicy:
    def __init__(self, env, discount=0.99, epsilon=0.1):
        self.env = env
        self.discount = discount
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(len(env.action_space)))
        self.ret = defaultdict(lambda: [[]] * len(env.action_space))

    def train(self, episodes=500, render=False):
        for i in tqdm(range(episodes)):
            sar = []  # (state, action, reward)
            state = self.env.reset()
            if render: self.env.render(wait=0)
            done = False
            while not done:
                action = self._get_action(state)
                n_state, reward, done = self.env.step(action)
                sar.append((tuple(state), action, reward))
                state = n_state
                if render: self.env.render(wait=0)

            G = 0
            sar.reverse()
            visited = set()
            for s, a, r in sar:
                G = self.discount * G + r
                if (s + (a,)) not in visited:
                    visited.add((s + (a,)))
                    self.ret[s][a].append(G)
                    self.Q[s][a] = np.mean(self.ret[s][a])

    def _get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.env.action_space)
        else:
            action = np.argmax(self.Q[tuple(state)])
        return action

    def test(self, render=True):
        while True:
            state = self.env.reset()
            if render: self.env.render()
            done = False
            while not done:
                action = np.argmax(self.Q[tuple(state)])
                state, _, done = self.env.step(action)
                if render: self.env.render()


class MonteCarloOffPolicy:
    def __init__(self):
        pass

    def train(self, render=False):
        """
        Implement here.
        """
        pass

    def test(self, render=True):
        pass
