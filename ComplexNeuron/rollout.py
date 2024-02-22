import gymnasium as gym
import numpy as np
from neural_diversity_net import NeuralDiverseNet


def fitness(net: NeuralDiverseNet, env_name: str) -> float:
    env = gym.make(env_name)
    obs = env.reset()[0]
    done = False
    r_tot = 0
    while not done:
        action = net.forward(obs)
        obs, r, term, trun, _ = env.step(action)
        if term == True or trun == True:
            done = True
        r_tot += r
    env.close()
    return r_tot
