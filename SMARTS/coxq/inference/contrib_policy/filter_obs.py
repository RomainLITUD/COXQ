import math
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import time

from smarts.core.utils.core_math import position_to_ego_frame, wrap_value
from smarts.core.agent_interface import RGB
from smarts.core.colors import Colors, SceneColors
from smarts.core.utils.observations import points_to_pixels, replace_rgb_image_color
from smarts.core.agent_interface import AgentInterface
from smarts.core.agent_interface import Waypoints, RoadWaypoints


class FilterObs(object):
    """Prepare observation"""
    def __init__(self, env, config):
        self.num_neighbors = config.num_neighbors
        self.num_lanes = config.num_lanes
        self.num_waypoints = config.num_waypoints
        self.hist_steps = config.hist_steps

        self.ego_position = (0,0,0)
        self.ego_heading = 0

        self.buffer = defaultdict(dict)
        self.timestep = 0
        self._crop = [0, 0, 0, 70]
        
        self._no_color = np.zeros((3,))
        self._wps_color = np.array(Colors.GreenTransparent.value[0:3]) * 255
        self._goal_color = np.array(Colors.Purple.value[0:3]) * 255
        self._road_color = np.array(SceneColors.Road.value[0:3]) * 255
        self._blur_radius = 1
        
    def reset(self):
        self.buffer = defaultdict(dict)
        self.timestep = 0
    
    def ego_view_transform(self, pos):
        pos_new = position_to_ego_frame(pos, self.ego_position, self.ego_heading)
        return pos_new
    
    def adjust_heading(self, head):
        return wrap_value(head - self.ego_heading, -math.pi, math.pi)
    
    def cache(self, obs):
        self.buffer[self.timestep]['AV'] = 0
        neighbors = obs.neighborhood_vehicle_states
        for neighbor in neighbors:
            state = np.concatenate([neighbor.position[:2], 
                                    [neighbor.heading+np.pi/2, neighbor.speed]])
            self.buffer[self.timestep][neighbor.id] = state

        if self.timestep > self.hist_steps-1:
            del self.buffer[self.timestep-self.hist_steps]

    def ego_map_process(self, obs):
        paths = obs.waypoint_paths

        ego_map = np.zeros(shape=(self.num_lanes, self.num_waypoints, 3))

        waypoints = []

        for i, path in enumerate(paths):
            if i >= self.num_lanes:
                break

            for j, point in enumerate(path):
                if np.sum(point.pos) != 0:
                    ego_map[i, j, :2] = self.ego_view_transform(np.append(point.pos, [0]))[:2]
                    ego_map[i, j, -1] = self.adjust_heading(point.heading + np.pi/2) 
                    waypoints.append(np.append(point.pos, [0]))

        return ego_map, np.array(waypoints)
    
    def neighbor_history_process(self, ids):
        neighbor_history = np.zeros(shape=(self.num_neighbors, self.hist_steps, 5))

        for i, id in enumerate(ids):
            timesteps = list(self.buffer.keys())
            idx = -1

            for t in reversed(timesteps):
                if id not in self.buffer[t] or idx < -self.hist_steps:
                    break 
                
                pos = self.buffer[t][id][:2]
                head = self.buffer[t][id][2]
                speed = self.buffer[t][id][3]
                neighbor_history[i, idx, :2] = self.ego_view_transform(np.append(pos, [0]))[:2]
                head_new = self.adjust_heading(head) 
                neighbor_history[i, idx, 2:-1] = np.array([math.cos(head_new), math.sin(head_new)])*speed
                neighbor_history[i, idx, -1] = -idx
                idx -= 1

        return neighbor_history

    def neighbor_process(self, obs):
        neighbors = {}
        for neighbor in obs.neighborhood_vehicle_states:
            neighbors[neighbor.id] = neighbor.position[:2]

        sorted_neighbors = sorted(neighbors.items(), 
                                  key=lambda item: np.linalg.norm(item[1]-self.ego_position[:2]))
        sorted_neighbors = sorted_neighbors[:self.num_neighbors]
        neighbor_ids = [neighbor[0] for neighbor in sorted_neighbors]

        neighbors_state = self.neighbor_history_process(neighbor_ids)

        return neighbors_state
    
    def neighbor_process_current(self, obs):
        neighbors = {}
        for neighbor in obs.neighborhood_vehicle_states:
            neighbors[neighbor.id] = neighbor.position[:2]

        sorted_neighbors = sorted(neighbors.items(), 
                                  key=lambda item: np.linalg.norm(item[1]-self.ego_position[:2]))
        sorted_neighbors = sorted_neighbors[:self.num_neighbors]
        ids = [neighbor[0] for neighbor in sorted_neighbors]

        neighbor_history = np.zeros(shape=(self.num_neighbors, 4))
        idx = 0
        for nghb in obs.neighborhood_vehicle_states:
            if nghb.id in ids:
                neighbor_history[idx, :2] = self.ego_view_transform(np.append(nghb.position[:2], [0]))[:2]
                neighbor_history[idx, 2] = self.adjust_heading(nghb.heading) 
                neighbor_history[idx, 3] = nghb.speed
                idx += 1

        return neighbor_history
    
    def map_rgb(self, obs, wps):
        rgb_heading = (obs.ego_vehicle_state.heading + np.pi) % (2 * np.pi) - np.pi

        rgb = obs.top_down_rgb.data
        h, w, _ = rgb.shape

        context = np.zeros((h,w,2)).astype(np.uint8)

        # Superimpose waypoints onto rgb image
        wps_valid = points_to_pixels(
            points=wps,
            center_position=self.ego_position,
            heading=rgb_heading,
            width=w,
            height=h,
            resolution=0.5,
        )
        for point in wps_valid:
            img_x, img_y = point[0], point[1]
            context[
                    max(img_y-self._blur_radius,0):min(img_y+self._blur_radius,h), 
                    max(img_x-self._blur_radius,0):min(img_x+self._blur_radius,w), 
                    0,
                ] = 255

        # Superimpose goal position onto rgb image       
        if not all((goal:=obs.ego_vehicle_state.mission.goal.position) == np.zeros((3,))):       
            goal_pixel = points_to_pixels(
                points=np.expand_dims(goal,axis=0),
                center_position=self.ego_position,
                heading=rgb_heading,
                width=w,
                height=h,
                resolution=0.5,
            )
            if len(goal_pixel) != 0:
                img_x, img_y = goal_pixel[0][0], goal_pixel[0][1]
                context[
                    max(img_y-self._blur_radius,0):min(img_y+self._blur_radius,h), 
                    max(img_x-self._blur_radius,0):min(img_x+self._blur_radius,w), 
                    1,
                ] = 255
        bev = np.uint8(np.concatenate([rgb, context], -1))
        bev = bev[self._crop[2]:h-self._crop[3],self._crop[0]:w-self._crop[1],:]
        return bev.transpose(2, 0, 1)
    
    def filter(self, obs, weight) -> Dict[str, Any]:
        self.ego_heading = obs.ego_vehicle_state.heading + np.pi/2
        self.ego_position = np.array(obs.ego_vehicle_state.position)

        ego_acc = np.array(obs.ego_vehicle_state.linear_acceleration[:2])
        linear_acc = np.dot(ego_acc, [math.cos(self.ego_heading), math.sin(self.ego_heading)])

        self.cache(obs)

        ego_state = np.array([obs.ego_vehicle_state.speed, linear_acc])
        #print(ego_state)
        ego_waypoints, wps = self.ego_map_process(obs)
        neighbors_state = self.neighbor_process(obs)

        # # Goal position     
        goal = self.ego_view_transform(obs.ego_vehicle_state.mission.goal.position)
        goal_xy = goal[:2]

        # bev = self.map_rgb(obs, wps)

        filtered_obs = {'ego_state': np.expand_dims(ego_state, 0).astype(np.float32), 
                        'ego_map': ego_waypoints.astype(np.float32), 
                        'neighbors_state': neighbors_state.astype(np.float32), 
                        # 'bev': bev,
                        'goal': np.array(goal_xy).astype(np.float32),
                        'weight': weight.astype(np.float32),
                        }
        
        self.timestep += 1
        return filtered_obs
