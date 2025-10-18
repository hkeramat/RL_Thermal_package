from gymnasium.spaces import Discrete, Box
from thermal_stack_rl.envs.thermal_stack_env import ThermalStackRLLibEnv

def test_env_shapes():
    env = ThermalStackRLLibEnv({})
    assert isinstance(env.action_space, Discrete)
    assert isinstance(env.observation_space, Box)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (6,)

def test_env_termination():
    env = ThermalStackRLLibEnv({})
    obs, _ = env.reset(seed=0)
    done = False
    steps = 0
    while not done and steps < 20:
        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        steps += 1
    assert done
    assert "k_series" in info
