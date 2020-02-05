from gym.envs.registration import register

register(
    id='LidarBat-v0',
    entry_point='environments.bat_flying_env:BatFlyingEnv'
)

register(
    id='LidarBatPolicyMap-v0',
    entry_point='environments.bat_flying_env_policy_visualizer:BatFlyingEnvPolicyVisualizer'
)