import gymnasium as gym
from contrib_policy.filter_obs import FilterObs
from contrib_policy.format_action import FormatAction, get_actions
from contrib_policy.make_dict import *
from collections import deque
import random

from smarts.zoo.agent_spec import AgentSpec

from smarts.core.controllers import ActionSpaceType


class Preprocess(gym.Wrapper):
    def __init__(self, env: gym.Env, 
                 config, 
                 weight_value,
                 cov_dim=2,
                 train=True
                 ):
        super().__init__(env)

        self._filter_obs = FilterObs(env, config)
        self.cov_dim = cov_dim
        self._make_dict = MakeObsDict()
        self._format_action = FormatAction(ActionSpaceType.RelativeTargetPose)
        self.train = train
        self.observation_space = self._make_dict.make_observation_space(config)
        lb = np.array([-1,-1], dtype=np.float64)
        ub = np.array([1, 1], dtype=np.float64)
        self.action_space = gym.spaces.box.Box(low=lb, high=ub, dtype=np.float64)
        #self.action_space = gym.spaces.MultiDiscrete([3, 5])
        self.index = weight_value
        #self.weight = np.eye(3)[weight_value]
        if self.train:
            initial_w = 1#random.uniform(0, 2)
            self.weight = np.array([initial_w, 1])
        else:
            self.weight = np.array([1, 1])


    def _process(self, obs):
        obs = self._filter_obs.filter(obs, self.weight)

        return obs

    def step(self, action):
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        
        #action_exe = self.action_deque.pop()
        formatted_action = self._format_action.format(
            action=action, 
            prev_heading=self._prev_heading, 
            prev_speed = self._prev_speed,
            index = self.index,
        )

        #self.action_deque.appendleft(action)

        obs, reward, terminated, truncated, info = self.env.step(formatted_action)
        self._prev_heading = obs.ego_vehicle_state.heading #obs["ego_vehicle_state"]["heading"]
        self._prev_speed = obs.ego_vehicle_state.speed
        obs = self._process(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if self.train:
            new_w = random.uniform(0, 2)
            self.weight = np.array([new_w, 1])
        self._filter_obs.reset()
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_heading = obs.ego_vehicle_state.heading
        self._prev_speed = obs.ego_vehicle_state.speed


        obs = self._process(obs)
        return obs, info
