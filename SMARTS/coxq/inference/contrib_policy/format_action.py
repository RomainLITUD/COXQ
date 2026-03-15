from typing import Callable, Tuple

import gymnasium as gym
import numpy as np

from smarts.core.controllers import ActionSpaceType


class FormatAction:
    def __init__(self, action_space_type: ActionSpaceType):
        if action_space_type == ActionSpaceType.RelativeTargetPose:
            self._wrapper, self.action_space = get_actions()
            #self._wrapper, self.action_space = get_discrete_actions()
        else:
            raise Exception(f"Unknown action space type {action_space_type}.")

    def format(self, action, prev_heading, prev_speed, index):
        wrapped_act = self._wrapper(action, prev_heading, prev_speed, index)
        return wrapped_act

def get_actions():
    lb = np.array([-1,-1], dtype=np.float64)
    ub = np.array([1, 1], dtype=np.float64)
    space = gym.spaces.box.Box(low=lb, high=ub, dtype=np.float64)

    dt = 0.25
    #acc_max = 10

    def wrapper(action, prev_heading, prev_speed, index=0) -> np.ndarray:

        # throttle, brake, steering_rate = action[0], action[1], action[2]
        #print(action[1])
        at, steering_rate = action[0], action[1]*0.2

        #at = ((at + 1) * (10 + 20) / 2) - 20
        at = at*13 #at = at * 6 if at > 0 else at * 16

        t_stop = prev_speed / abs(at) if at < 0 else float('inf')

        delta_heading = steering_rate*(min(t_stop, dt)/dt)
        
        new_heading = prev_heading + delta_heading
        new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi

        #magnitude = acc_dis*1.25 #
        magnitude = prev_speed * min(dt, t_stop) + 0.5 * at * min(dt, t_stop) ** 2
        magnitude = min(max(magnitude, 0), 13*dt)
        
        # Note: On the map, angle is zero at positive y axis, and increases anti-clockwise.
        #       In np.exp(), angle is zero at positive x axis, and increases anti-clockwise.
        #       Hence, numpy_angle = map_angle + π/2
        new_pos = magnitude * np.exp(1j * (new_heading + np.pi / 2))
        delta_x = np.real(new_pos)
        delta_y = np.imag(new_pos)
        #print(prev_speed, at*0.4)
        wrapped_action = np.array([delta_x, delta_y, delta_heading], dtype=np.float32)
        return wrapped_action

    return wrapper, space
