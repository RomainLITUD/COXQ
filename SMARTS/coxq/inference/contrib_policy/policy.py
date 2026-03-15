import sys
from pathlib import Path
import torch
import numpy as np
import random
from contrib_policy import network
from contrib_policy.filter_obs import FilterObs
from contrib_policy.make_dict import MakeObsDict
from contrib_policy.format_action import FormatAction

# To import contrib_policy folder
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from smarts.core.agent import Agent

class Policy(Agent):

    def __init__(self, action_space_type,
                 config
                 ):
        
        self.model = self.get_model(config)

        self.filter_obs = FilterObs(config)
        self.filter_obs.reset()
        self.format_obs = MakeObsDict()
        self.observation_space = self.format_obs.make_observation_space(config)

        self._format_action = FormatAction(action_space_type=action_space_type)
        self.action_space = self._format_action.action_space

    def act(self, obs):
        processed_obs = self._process(obs)

        action, _ = self.model.predict(observation=processed_obs, deterministic=True)
        formatted_action = self._format_action.format(action)

        return formatted_action

    def _process(self, obs):
        if obs["steps_completed"] == 1:
            # Reset memory because episode was reset.
            self.filter_obs.reset()

        obs = self.filter_obs.filter(obs)
        return obs

    def _get_model(self):
        import stable_baselines3 as sb3lib

        return sb3lib.PPO.load(path=Path(__file__).resolve().parents[0] / "saved_model")

    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)