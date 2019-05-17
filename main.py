from env import GridWorld
from agents.human import Human
from agents.monte_carlo import MonteCarloOnPolicy
from agents.q_learning import QLearning
from agents.sarsa import Sarsa

env = GridWorld()
agent = Human(env)
# agent = MonteCarloOnPolicy(env)
# agent = QLearning(env)
# agent = Sarsa(env)

agent.train(episodes=100000, render=True)
agent.test()
