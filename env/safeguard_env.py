import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SafeGuardEnv(gym.Env):
    """
    A custom Gymnasium environment for the Meta OpenEnv Hackathon.
    Simulates an industrial facility where an RL Agent must balance 
    mitigation actions against environmental hazards.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode=None):
        super(SafeGuardEnv, self).__init__()
        self.render_mode = render_mode

        # Actions:
        # 0: Do nothing
        # 1: Turn on Ventilation (Decreases CO2 & PM2.5, but uses energy)
        # 2: Turn on Cooling (Decreases Temp, uses energy)
        # 3: Emergency Shutdown (Forces everything to safe baseline, MASSIVE penalty to production)
        self.action_space = spaces.Discrete(4)

        # Observation Space: [Temp(20-100C), Humidity(0-100%), CO2(300-2000ppm), PM2.5(0-500ug/m3)]
        low = np.array([20.0, 0.0, 300.0, 0.0], dtype=np.float32)
        high = np.array([100.0, 100.0, 2000.0, 500.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Environment configuration
        self.max_steps = 200
        self.current_step = 0
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start at a safe baseline
        self.state = np.array([
            np.random.uniform(22.0, 28.0),   # Temp
            np.random.uniform(40.0, 50.0),   # Humidity
            np.random.uniform(400.0, 450.0), # CO2
            np.random.uniform(10.0, 20.0)    # PM2.5
        ], dtype=np.float32)
        
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Environmental drift (natural accumulation of heat and emissions in a factory)
        self.state[0] += np.random.uniform(0.1, 1.0)  # Temp goes up
        self.state[1] += np.random.uniform(-0.5, 0.5) # Humidity fluctuates
        self.state[2] += np.random.uniform(5.0, 25.0) # CO2 builds up
        self.state[3] += np.random.uniform(1.0, 5.0)  # PM2.5 builds up

        # Apply Agent Actions
        if action == 1:
            # Vent: Large reduction in CO2 and PM, slight energy penalty
            self.state[2] -= 40.0
            self.state[3] -= 10.0
            reward -= 1.0 
        elif action == 2:
            # Cool: Reduction in Temp, slight energy penalty
            self.state[0] -= 2.0
            reward -= 1.0
        elif action == 3:
            # Shutdown: Forces to baseline but massive penalty
            self.state[0] = 25.0
            self.state[2] = 400.0
            self.state[3] = 15.0
            reward -= 100.0

        # Enforce physical bounds
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        # Baseline survival reward
        reward += 2.0 

        # Danger thresholds Penalties
        temp, hum, co2, pm = self.state
        if temp > 40.0:
            reward -= 10.0
        if co2 > 1000.0:
            reward -= 10.0
        if pm > 100.0:
            reward -= 10.0

        # Terminal conditions (Catastrophe)
        if temp >= 90.0 or co2 >= 1900.0 or pm >= 450.0:
            reward -= 500.0
            terminated = True

        return self.state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "console":
            print(f"Step: {self.current_step} | State: {self.state}")

# Register the environment
gym.envs.registration.register(
    id='SafeGuardEnv-v0',
    entry_point='env.safeguard_env:SafeGuardEnv',
    max_episode_steps=200,
)
