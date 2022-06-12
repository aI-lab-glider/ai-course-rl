from gym.envs.registration import register
from importlib_metadata import entry_points
from sandbox.enviroments.grid_pathfinding.env import GridPathfindingEnv
from sandbox.enviroments.multi_armed_bandit.env import BanditEnv
from sandbox.enviroments.twenty_forty_eight.env import TwentyFortyEightEnv
import sandbox.enviroments as envs
from inspect import getmodule

def as_entry_point(cls: type):
    return f'{getmodule(cls).__name__}:{cls.__name__}'

register("custom/2048-v0", entry_point=as_entry_point(TwentyFortyEightEnv))
register("custom/gridpathfinding-v0", entry_point=as_entry_point(GridPathfindingEnv))
register("custom/multiarmed-bandits-v0", entry_point=as_entry_point(BanditEnv))