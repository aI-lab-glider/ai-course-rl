import gym 


class GridPathfindingEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        pass 

    def step(self, action: int):
        raise NotImplementedError()
    
    def reset(self, *args, **kwargs):
        raise NotImplementedError()

    def render(self, *args, **kwargs):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        raise NotImplementedError()