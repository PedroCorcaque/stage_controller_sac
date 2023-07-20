from stage_controller_sac.env import StageEnv

from gym.envs.registration import register

register(
    id="Stage-v1",
    entry_point="stage_controller_sac:StageEnv"
)