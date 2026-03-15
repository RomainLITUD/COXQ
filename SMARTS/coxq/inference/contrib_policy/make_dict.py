from typing import Dict

import gymnasium as gym
import numpy as np


class MakeObsDict(object):
    """Converts gym.spaces.Box to gym.spaces.Dict."""

    def __init__(self):
        self.obs = 1
        self.weight_dim = 2

    def make_observation_space(self, config):

        num_neighbors = config.num_neighbors
        num_lanes = config.num_lanes
        num_waypoints = config.num_waypoints
        hist_steps = config.hist_steps

        space_bev = gym.spaces.Box(
            low=0,
            high=255,
            shape=(5,
                130,
                200
            ),
            dtype=np.uint8,
        )

        space_ego = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,2), 
                                   dtype=np.float32)
        
        space_neighbor = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                        shape=(num_neighbors, hist_steps, 5),
                                        dtype=np.float32)
        space_map = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                        shape=(num_lanes, num_waypoints, 3),
                                        dtype=np.float32)
        space_goal = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                        shape=(2,), dtype=np.float32)
        space_weight = gym.spaces.Box(low=0, high=1, 
                                        shape=(self.weight_dim,), dtype=np.float32)

        observation_space = gym.spaces.Dict({'ego_state': space_ego, 
                                            'ego_map': space_map, 
                                            #'bev': space_bev,
                                            'neighbors_state': space_neighbor, 
                                            'goal': space_goal,
                                            'weight': space_weight,
                                            })

        return observation_space
