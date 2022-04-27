import sys 
from pathlib import Path

path = Path(__file__)
sys.path.append(str(path.parents[1].absolute()))

from sandbox.enviroments.multi_armed_bandit import BanditEnv, BanditTrainer, EpsilonGreedy, UCB, ThompsonSampling
from sandbox.enviroments.multi_armed_bandit.env import NormalDistribution
import numpy as np 


def main():
    # np.random.seed(113)
    n_bandits = 10
    n_episodes = 10000
    bandits = [NormalDistribution(mean, 1) for mean in np.random.normal(0, 1, n_bandits)]
    
    env = BanditEnv(bandits)
    policy1 = EpsilonGreedy(n_bandits, eps=0.1, init_value=0)
    policy2 = UCB(n_bandits, init_value=0)
    policy3 = EpsilonGreedy(n_bandits, eps=0.1, init_value=0)
    policy4 = ThompsonSampling(n_bandits, init_value=0)


    trainer = BanditTrainer(env, [policy1, policy2, policy3, policy4])
    _ = trainer.train(n_episodes)

    trainer.display_history()


if __name__ == '__main__':
    main()
    