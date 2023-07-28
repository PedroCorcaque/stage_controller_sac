# Stage Controller SAC

The stage_controller_sac package provides a trainable stage simulator environment for reinforcement learning, integrated with a SAC implementation.

___

## Installation

You will need to have [ROS Noetic](wiki.ros.org/noetic/Installation/Ubuntu) installed on your computer.

### Install dependencies

```
pip3 install gym
pip3 install numpy
```

### Download the repository

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/PedroCorcaque/stage_controller_sac.git
```

### Build the package

```
cd ~/catkin_ws/
catkin build
```