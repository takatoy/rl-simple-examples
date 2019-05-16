from env import GridWorld
from agents.monte_carlo import MonteCarloOnPolicy
from agents.q_learning import QLearning
from agents.sarsa import Sarsa

env = GridWorld()
# agent = QLearning(env)
agent = MonteCarloOnPolicy(env)

agent.train(episodes=100000, render=False)
agent.test()
