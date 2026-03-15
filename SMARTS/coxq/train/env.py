import sys
from pathlib import Path
from stable_baselines3.common.utils import set_random_seed
import random

# To import train folder
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import gymnasium as gym

from smarts.zoo.agent_spec import AgentSpec


def make_env(env_id, scenario, agent_spec: AgentSpec, config, seed, weights,
             train_, forbidden, cov_dim=2):
    from preprocess import Preprocess
    from reward import Reward
    from stable_baselines3.common.monitor import Monitor

    from smarts.env.gymnasium.wrappers.single_agent import SingleAgent

    env = gym.make(
        env_id,
        scenario=scenario,
        agent_interface=agent_spec.interface,
        seed=seed,
        sumo_headless=not config.sumo_gui,  # If False, enables sumo-gui display.
        headless=not config.head,  # If False, enables Envision display.
    )
    env = Reward(env=env, forbidden=forbidden)
    env = SingleAgent(env=env)
    env = Preprocess(env=env, config=config, weight_value = weights, cov_dim=cov_dim, train=train_)
    env = Monitor(env, info_keywords=("cost", "return"))

    return env

def make_parallel_env(env_id, scenario, agent_spec, config, seed, weights, forbidden, worker_id, train_=True, cov_dim=2):

    def _init():
        my_seed = seed + worker_id
        import random, numpy as _np, torch as _th
        random.seed(my_seed)
        _np.random.seed(my_seed)
        _th.manual_seed(my_seed)
        # rank = random.randint(0, 100)
        env = make_env(env_id, scenario, agent_spec, config, my_seed, weights, train_, forbidden, cov_dim=cov_dim)
        env.reset(seed=my_seed)
        return env
    # set_random_seed(seed)
    return _init
