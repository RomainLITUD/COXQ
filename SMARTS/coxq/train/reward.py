from typing import Tuple
import torch as th
import gymnasium as gym
import numpy as np
import math
from itertools import chain
from shapely.geometry import LineString, Point, MultiLineString
import matplotlib.pyplot as plt
import time
from collections import OrderedDict
import random
from scipy.stats import gennorm

def heading_vec(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=float)


def ttc_capsule(p1: np.ndarray, yaw1: float, v1_scalar: float,
                p2: np.ndarray, yaw2: float, v2_scalar: float) -> float:

    h1, h2 = heading_vec(yaw1), heading_vec(yaw2)
    # vehicle size: 3.68 x 1.47

    r = 0.735 + 0.3
    front1 = 1.84 + 0.6 * v1_scalar
    front2 = 1.84 + 0.6 * v2_scalar

    back1 = 1.84 + 0.2 * v1_scalar
    back2 = 1.84 + 0.2 * v2_scalar

    a1, b1 = p1 + front1*h1, p1 - back1*h1
    a2, b2 = p2 + front2*h2, p2 - back2*h2

    seg1 = LineString([a1, b1])
    seg2 = LineString([a2, b2])

    dist = seg1.distance(seg2)

    if dist < 2*r + 0.1:
        return -1.

    return 0.

def ttc_cost(ttc: float,
             T_warn=3.5, T_crit=1.3, beta=3.0) -> float:
    if math.isinf(ttc):
        return 0.
    if ttc >= T_warn:
        return 0.
    elif ttc > T_crit:
        z = (T_warn - ttc)/(T_warn - T_crit)
        return -z
    else:
        return -1. # -min(1.0, 1.0 + (T_crit - ttc)/beta)


class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env, forbidden):
        """Constructor for the Reward wrapper."""
        super().__init__(env)
        self._total_dist = {}
        self.roadpoints = []
        self.env = env

        self.counts = np.zeros(4)

        self.forbidden = forbidden

    def reset_count(self,):
        self.counts = np.zeros(4)

    def reset(self, *, seed=None, options=None):
        self._total_dist = {}

        return self.env.reset(seed=seed, options=options)


    def calculate_offset_lane(self, obs):

        if obs.ego_vehicle_state.lane_id in self.forbidden:
            return -0.5, -2.

        osl = abs(obs.ego_vehicle_state.lane_position.t/1.6)

        u_t = 1 if osl > 1 else osl

        return -0.5*u_t, 0.

    def calculate_capsule_2dttc(self, ego, neighbor):
        ego_pos = np.array(ego.position[:2])
        ego_heading = ego.heading - np.pi/2
        ego_speed = ego.speed

        nghb_pos = np.array(neighbor.position[:2])
        nghb_heading = neighbor.heading - np.pi/2
        nghb_speed = neighbor.speed

        ttc = ttc_capsule(ego_pos, ego_heading, ego_speed, 
                          nghb_pos, nghb_heading, nghb_speed)

        return ttc_cost(ttc)

    def reward_risk(self, obs):
        ego = obs.ego_vehicle_state
        ego_pos = ego.position

        nghbs = obs.neighborhood_vehicle_states

        nghbs_state = sorted(
            [nghb for nghb in nghbs],
            key=lambda nghb: np.linalg.norm(np.subtract(nghb.position, ego_pos))
        )[:5]


        if len(nghbs_state) == 0:
            return 0

        prs = []
        for nghb_state in nghbs_state:
            # pr = self.calculate_driver_space(ego, nghb_state)
            pr = self.calculate_capsule_2dttc(ego, nghb_state)
            prs.append(pr)

        reward_pr = np.amin(np.array(prs))
        return reward_pr

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        for agent_id, agent_obs in obs.items():
            info[agent_id]['cost'] = -wrapped_reward[agent_id][0]

        for agent_id, agent_obs in obs.items():
            # Accumulate total distance travelled
            self._total_dist[agent_id] = (
                self._total_dist.get(agent_id, 0) + agent_obs.distance_travelled
            )

            # If agent is done
            if terminated[agent_id] == True:
                if agent_obs.events.reached_goal:
                    pass
	            #print(f"{agent_id}: Hooray! Reached goal.")
                elif agent_obs.events.reached_max_episode_steps:
                    print(f"{agent_id}: Reached max episode steps.")
                elif (
                    agent_obs.events.collisions
                    or agent_obs.events.off_road
                    or agent_obs.events.off_route
                    or agent_obs.events.wrong_way
                ):
                    pass
                else:
                    print("Events: ", agent_obs["events"])
                    raise Exception("Episode ended for unknown reason.")

        return obs, wrapped_reward, terminated, truncated, info

    def _reward(self, obs, env_reward):
        reward = {agent_id: np.zeros(2).astype(np.float64) for agent_id in obs.keys()}

        for agent_id, agent_obs in obs.items():

            #if agent_obs.ego_vehicle_state.speed > 13.:
             #   reward[agent_id][0] -= np.float64(10)

            # Penalty for colliding
            if agent_obs.events.collisions:
                reward[agent_id][0] -= np.float64(10)
                self.counts[0] += 1
                print(self.counts)
                #print(f"{agent_id}: collided.", flush=True)
                continue

            # Penalty for driving off road
            if agent_obs.events.off_road:
                reward[agent_id][0] -= np.float64(10)
                #reward[agent_id][0] -= np.float64(1)
                self.counts[1] += 1
                print(self.counts)
                #print(f"{agent_id}: off road.", flush=True)
                continue

            if agent_obs.events.off_route:
                reward[agent_id][0] -= np.float64(10)
                #reward[agent_id][0] -= np.float64(1)
                self.counts[2] += 1
                print(self.counts)
                #print(f"{agent_id}: off route.", flush=True)
                continue

            # Penalty for driving on wrong way
            if agent_obs.events.wrong_way:
                reward[agent_id][0] -= np.float64(10)
                #reward[agent_id][0] -= np.float64(1)
                self.counts[3] += 1
                print(self.counts)
                #print(f"{agent_id}: wrong way.", flush=True)
                continue

            # Reward for reaching goal
            if agent_obs.events.reached_goal:
                reward[agent_id][1] += np.float64(30)
                print("GOAL !!!!!!")
                continue

            # center, outlier = self.calculate_offset_lane(agent_obs)
            # comfort = self.calculate_comfort(agent_obs)
            distance = np.float64(env_reward[agent_id])
            # risk = self.reward_risk(agent_obs)

            #reward[agent_id][0] += risk
            reward[agent_id][1] += distance


        return reward
