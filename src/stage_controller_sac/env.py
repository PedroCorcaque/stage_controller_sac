#!/usr/bin/env python3
import rospy
import numpy as np
import time

import gym
from gym import spaces

from stage_controller_sac.respawn_goal import Respawn

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

EPS = 1e-8

class StageEnv(gym.Env):
    """The environment."""
    @property
    def angular_velocity(self):
        return self._angular_velocity
    
    @angular_velocity.setter
    def angular_velocity(self, velocity):
        self._angular_velocity = velocity

    @property
    def linear_velocity(self):
        return self._linear_velocity
    
    @linear_velocity.setter
    def linear_velocity(self, velocity):
        self._linear_velocity = velocity

    def __init__(self, goal_list = None):
        assert goal_list != None
        self.goal_list = np.asarray(goal_list)

        self._read_parameters()

        self.publisher_velocity = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self._odometry_callback)
        # rospy.Subscriber("/base_scan", LaserScan, self._scan_callback)

        self.robot_position = Pose()
        self.target_position_x = None
        self.target_position_y = None

        self.start_position_x = -2
        self.start_position_y = -1

        self.scan = None
        self.min_read_scan = 0.05
        self.max_read_scan = 8

        self._angular_velocity = None
        self._linear_velocity = None
        self.min_linear = 0.1
        self.max_linear = 0.5
        self.min_angular = -2.35
        self.max_angular = 2.35

        self.init_goal = True
        self.target_reached = False
        self.target_distance = None

        self.respawn_goal = Respawn()
        self.respawn_goal.set_target_list(goal_list)

        self.reset_proxy = rospy.ServiceProxy('reset_positions', Empty)

        # This works to a continuous space
        self.action_space = spaces.Box(low=np.array([self.min_angular, self.min_linear]),
                                       high=np.array([self.max_angular, self.max_linear]),
                                       shape=(self.action_dimension,),
                                       dtype=np.float64)
        
        # 0.05 and 8? None?
        # np.full returns a complete array of args[0] length with args[1] value
        self.observation_space = spaces.Box(low=np.full(self.state_dimension, self.min_read_scan, dtype=np.float64),
                                            high=np.full(self.state_dimension, self.max_read_scan, dtype=np.float64))

    def _read_parameters(self):
        """Read the ros parameters."""
        self.state_dimension = rospy.get_param("~state_dimension")
        self.action_dimension = rospy.get_param("~action_dimension")
        self.targetbox_distance = rospy.get_param("~targetbox_distance")
        self.collision_distance = rospy.get_param("~collision_distance")
        self.reward_target = rospy.get_param("~reward_target")
        self.reward_collision = rospy.get_param("~reward_collision")

    def _odometry_callback(self, msg):
        """Callback for robot position."""
        self.robot_position = msg.pose.pose.position

    def _scan_callback(self, msg):
        """Callback for laser scan."""
        self.scan = np.asarray(msg.ranges)

    def _get_target_distance(self):
        """Get the distance between the robot and the target."""
        return np.sqrt((self.target_position_x - self.robot_position.x)**2 + \
                       (self.target_position_y - self.robot_position.y)**2)
    
    def _get_state(self, data):
        """Get current state of the environment."""
        scan_range = []
        done = False

        for i in range(len(data)):
            if data[i] == float("Inf"):
                scan_range.append(self.max_read_scan)
            elif np.isnan(data[i]):
                scan_range.append(self.min_read_scan)
            else:
                scan_range.append(data[i])

        if min(scan_range) < self.collision_distance:
            done = True
        elif self._get_target_distance() < self.targetbox_distance:
            if not done:
                self.target_reached = True
                done = True

        return np.asarray(scan_range), done

    def _set_reward(self, done):
        """Set the reward based on state.
        
        Possible states:
        ---
        Target reached: The robot has reached the target.
        Done: The robot has crashed.
        None: The steps has been finished without success and no crash.
        """
        if self.target_reached:
            reward = self.reward_target
            self._stop_robot()
            self.target_position_x, self.target_position_y = self._get_new_target(True)
            self.target_distance = self._get_target_distance()
            self.target_reached = False
        elif done:
            reward = self.reward_collision
            self._stop_robot()
            if self.respawn_goal.last_index != 0:
                self.respawn_goal.init_index()
                self.target_position_x, self.target_position_y = self._get_new_target(True)
                self.target_distance = self._get_target_distance()
        else:
            reward = -np.log(self._get_target_distance() + EPS)

        return reward
    
    def _get_walked_distance(self):
        """Get the distance of the initial to final position."""
        return np.sqrt((self.robot_position.x - self.start_position_x)**2 + 
                       (self.robot_position.y - self.start_position_y)**2)

    def _publish_velocity(self, msg):
        """A helper to publish the velocity to robot."""
        self.publisher_velocity.publish(msg)

    def _stop_robot(self):
        """To stop the robot."""
        velocity = Twist()
        velocity.linear.x = 0.0
        velocity.angular.z = 0.0
        self._publish_velocity(velocity)

    def _get_new_target(self, check_position):
        """Get the new target to robot."""
        return self.respawn_goal.get_position(check_position)

    def step(self, action):
        self.angular_velocity = np.clip(action[0], 
                                     self.min_angular,
                                     self.max_angular)
        self.linear_velocity = np.clip(action[1], 
                                     self.min_linear,
                                     self.max_linear)

        velocity = Twist()
        velocity.linear.x = self._linear_velocity
        velocity.angular.z = self._angular_velocity
        self._publish_velocity(velocity)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/base_scan', LaserScan, timeout=15)
                data = data.ranges
            except:
                pass

        state, done = self._get_state(data)
        reward = self._set_reward(done)

        return np.asarray(state), reward, done, {}

    def reset(self):
        print(f"Target: {[self.target_position_x, self.target_position_y]}", end=" \t| ")
        self.respawn_goal.set_target_list(self.goal_list)

        if self.init_goal:
            self.target_position_x, self.target_position_y = self._get_new_target(False)
            self.init_goal = False
        else:
            self.target_position_x, self.target_position_y = self._get_new_target(True)

        rospy.wait_for_service('/reset_positions')
        try:
            self.reset_proxy()
        except rospy.ServiceException:
            print("reset_positions service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/base_scan', LaserScan, timeout=15)
                data = data.ranges
            except:
                pass

        self.target_distance = self._get_target_distance()
        state, _ = self._get_state(data)
        print(f"Next target: {[self.target_position_x, self.target_position_y]}")
        return state, {}
