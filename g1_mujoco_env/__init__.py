from gymnasium.envs.registration import register

register(
    id="g1_mujoco_env/G1Mujoco-v0",
    entry_point="g1_mujoco_env.envs:G1MujocoEnv",
)
