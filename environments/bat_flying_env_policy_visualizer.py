import numpy as np
import math

from .bat_flying_env import *



class BatFlyingEnvPolicyVisualizer(BatFlyingEnv):
    def __init__(self, x=100, y=100, *args):
        super().__init__(*args)
        self.map_shape = (x, y)

        self.left_limit_angle = (
            # np.pi/2 +
            self.max_pulse_angle +
            self.bat.lidar_left_angle)
        self.right_limit_angle = (
            # np.pi/2 +
            -self.max_pulse_angle +
            self.bat.lidar_right_angle)
    
        self.policy_map = np.zeros(
            (*self.map_shape, self.action_space.shape[0]))


    def reset(self):
        xs = np.linspace(-1, 1, self.map_shape[0])
        ys = np.linspace(-1, 1, self.map_shape[1])
        xv, yv = np.meshgrid(xs, ys)

        x_ind = np.arange(0, self.map_shape[0])
        y_ind = np.arange(0, self.map_shape[1])
        x_ind, y_ind = np.meshgrid(x_ind, y_ind)
        self.state_ind = np.stack([x_ind, y_ind], axis=2).reshape(
            self.map_shape[0]*self.map_shape[1], 2)
        self.state_map = np.stack([xv, yv], axis=2)
        self.state_list = self.state_map.reshape(
            self.map_shape[0]*self.map_shape[1], 2)
        self.state = np.c_[
            np.zeros(self.bat.n_memory), 
            np.ones(self.bat.n_memory)
        ].ravel()
        self._update_state()
        self.done = False
        return self.state
    
    def _update_state(self):
        if len(self.state_list) > 0:
            self.state[0:2] = self.state_list[0]
            self._state_ind = self.state_ind[0]
            self.state_list = self.state_list[1:]
            self.state_ind = self.state_ind[1:]
        else:
            base_dir = '/home/lee/python_codes/bat_agent/data/policy_map/'
            from datetime import datetime
            dt = datetime.today()
            self.filename = f'{dt.year}{dt.month:02}{dt.day:02}T{dt.hour:02}{dt.minute:02}{dt.second:02}.npy'
            np.save(base_dir+self.filename, self.policy_map)
            self.done = True

    def step(self, action):
        state = self.state[:2]
        state_angle = math.atan2(*state[::-1])
        if ((self.right_limit_angle < state_angle) and 
            (state_angle < self.left_limit_angle)):
        # if True:
            for i in range(len(action)):
                self.policy_map[
                    self._state_ind[0], self._state_ind[1], i] = action[i]
        self._update_state()
        return self.state, 0, self.done, {}