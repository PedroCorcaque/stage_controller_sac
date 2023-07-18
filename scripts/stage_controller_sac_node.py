#!/usr/bin/env python3
import rospy

import gym
import gym_stage
from stage_controller_sac.sac import SAC

class StageControllerSAC():
    """The StageControllerSAC use a SAC to move the robot to target position."""
    def __init__(self, env, goals):
        self.state_dimension = None
        self.action_dimension = None
        self.learning_rate = None
        self.gamma = None
        self.alpha = None
        self.batch_size = None
        self.num_episodes = None
        self.max_steps = None

        self.sac = None
        self.env = env
        self.goals = goals

        self._read_parameters()
        self._start_sac()

    def _read_parameters(self):
        """A internal method to get all the parameters from config file."""
        self.state_dimension = rospy.get_param("~state_dimension")
        self.action_dimension = rospy.get_param("~action_dimension")
        self.learning_rate = rospy.get_param("~learning_rate")
        self.gamma = rospy.get_param("~gamma")
        self.alpha = rospy.get_param("~alpha")
        self.batch_size = rospy.get_param("~batch_size")
        self.num_episodes = rospy.get_param("~num_episodes")
        self.max_steps = rospy.get_param("~max_steps")

    def _start_sac(self):
        """A internal method to initialize the SAC object."""
        self.sac = SAC(self.state_dimension,
                       self.action_dimension,
                       self.learning_rate,
                       self.gamma,
                       self.alpha)

    def start(self):
        """Start method to train the SAC."""
        self.sac.train(env=self.env,
                       num_episodes=self.num_episodes,
                       max_steps=self.max_steps,
                       batch_size=self.batch_size)

if __name__ == "__main__":
    rospy.init_node("stage_controller_sac_node")
    goals = [[0.0, 5.5], [-2,2], [-4, 4], [1.5, 6], [3.5,2.5], [3,-1], [-3, -1], [4, 4], [-2, 3]]
    env = gym.make("Stage-v0", env_stage=1, continuous=True, goal_list=goals)

    stage_controller_sac = StageControllerSAC(env, goals)
    stage_controller_sac.start()
