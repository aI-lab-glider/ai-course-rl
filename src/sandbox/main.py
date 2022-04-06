import sys 
from pathlib import Path

path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))

from sandbox.enviroments.multi_armed_bandit import BanditEnv, BanditTrainer, EpsilonGreedy
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution
import numpy as np 


def main():
    np.random.seed(123)
    n_bandits = 10
    n_episodes = 1000
    bandits = [NormalDistribution(mean, 1) for mean in np.random.normal(0, 1, n_bandits)]
    
    env = BanditEnv(bandits)
    policy = EpsilonGreedy(n_bandits, eps=0.1, init_value=0)

    trainer = BanditTrainer(env, policy)
    _ = trainer.train(n_episodes)

    trainer.display_history()


if __name__ == '__main__':
    main()
    