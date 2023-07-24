#!/usr/bin/env python3
import rospy

import torch.distributed as dist

import gym
from stage_controller_sac.env import StageEnv
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
        self.hidden_layer = rospy.get_param("~hidden_layer")

    def _start_sac(self):
        """A internal method to initialize the SAC object."""
        self.sac = SAC(self.state_dimension,
                       self.action_dimension,
                       self.learning_rate,
                       self.gamma,
                       self.alpha,
                       self.hidden_layer)

    def start(self, parallel=False):
        """Start method to train the SAC."""
        if parallel:
            dist.init_process_group(backend="nccl")
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            if world_size > 1:
                self.batch_size = self.batch_size // world_size

            self.sac.train_sac_parallel(env=self.env,
                        num_episodes=self.num_episodes,
                        max_steps=self.max_steps,
                        batch_size=self.batch_size)

        else:
            self.sac.train(env=self.env,
                        num_episodes=self.num_episodes,
                        max_steps=self.max_steps,
                        batch_size=self.batch_size)

if __name__ == "__main__":
    rospy.init_node("stage_controller_sac_node")
    goals = [[0.0, 5.5], [-2, 2], [-4, 4], [1.5, 6], [3.5, 2.5], [3,-1], [4, 4], [-3, -1]]
    env = gym.make("Stage-v1", goal_list=goals)

    stage_controller_sac = StageControllerSAC(env, goals)
    stage_controller_sac.start(parallel=False)
