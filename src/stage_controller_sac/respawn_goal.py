#!/usr/bin/env python3
import numpy as np
from geometry_msgs.msg import Pose

class Respawn():

    def __init__(self):
        self.target_position = Pose()
        self.target_x_list = None
        self.target_y_list = None
        self.len_target_list = None

        self.index = None
        self.last_index = None
        self.init_target_x = None
        self.init_target_y = None
        self.target_position.position.x = None
        self.target_position.position.y = None

    def init_index(self):
        self.index = 0
        self.target_position.position.x = self.init_target_x
        self.target_position.position.y = self.init_target_y
        self.last_index = self.index

    def set_target_list(self, target_list):
        self.target_x_list = [point[0] for point in target_list]
        self.target_y_list = [point[1] for point in target_list]
        self.len_target_list = len(self.target_x_list)
        self.init_target_x = self.target_x_list[0]
        self.init_target_y = self.target_y_list[1]
        self.init_index()

    def get_position(self, position_check=False):
        if position_check:
            random_index = np.random.randint(0, high=self.len_target_list)
            self.target_position.position.x = self.target_x_list[random_index]
            self.target_position.position.y = self.target_y_list[random_index]

        return self.target_position.position.x, self.target_position.position.y
