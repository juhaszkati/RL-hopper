import gymnasium as gym
import numpy as np


# Building custom environment
class MassActuatorEnv(gym.Env):  # inheriting the class of gym.Env

    # Setting render mode
    metadata = {"render_modes": ["human"],
                "render_fps": 30}  # "human": for console / "rgb_array": NumPy array for Matplotlib

    # Model parameters
    def __init__(self, render_mode=None):
        super().__init__()  # making sure our environment is compatible with gym.Env class
        self.state = None
        self.time = None
        self.termination = None
        self.n_jump = None
        self.jump_start = None
        self.success = None
        self.y_max = None
        self.mass = 1.0  # kg
        self.gravity = 9.81  # m/s^2
        self.max_force = 100  # N actuator
        self.ground = 0.0  # m
        self.goal = 5  # m
        # Change self.dt if training or testing agents with different timesteps
        self.dt = 0.05  # s timestep

        # Observation space definition [y, y_dot]
        observation_low = np.array([0, -100])
        observation_high = np.array([50, 100])
        self.observation_space = gym.spaces.Box(observation_low, observation_high, dtype=np.float32)

        # Action space definition [force] discrete
        self.action_space = gym.spaces.Discrete(3)

        self.reset()

    # Initial state
    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        self.state = np.array(options.get("initial_state", [5, 0]), dtype=np.float32)  # initial y and y_dot
        self.time = 0
        self.n_jump = 0
        self.jump_start = 0
        self.success = 0
        self.y_max = 0
        self.termination = options.get("termination")
        return self.state

    # Action in a step
    def step(self, action):
        y, v = self.state
        force_values = [-self.max_force, 0, self.max_force]
        force = force_values[action]  # chosen force from [-max, 0, max]
        prev_y = self.state[0]
        a = 0

        if y >= 2:
            a = - self.gravity
        elif y < 2:
            a = (force - self.mass * self.gravity) / self.mass

        v = a * self.dt + self.state[1]
        y = a * self.dt ** 2 / 2 + self.state[1] * self.dt + self.state[0]
        self.state = np.array([y, v], dtype=np.float32)
        self.time += self.dt

        if prev_y >= 2.0 > y:
            self.n_jump += 1

        if prev_y < 2.0 <= y:
            self.jump_start += 1

        # Reward
        reward = 0.5 * (1 / ((y - self.goal) ** 2 + 1) - abs(y - self.goal) / 5)
        if (y - self.goal + 0.1) ** 2 < 0.1 ** 2:
            reward += 1.5
        if (y - self.goal) ** 2 < 0.25 ** 2:
            if abs(v) < 0.5 and self.jump_start > self.success:
                self.success += 1
        if abs(v) < 0.5 and self.jump_start > 0:
            self.y_max = y

        terminated = y <= self.termination  # Stops training if termination limit is crossed
        truncated = self.time >= 10 or self.n_jump >= 2
        # Values of truncated in the following cases:
        # - training agents for 1 jump: self.time >= 3 or self.n_jump >= 2
        # - testing agents for 1 jump: self.time >= 10 or self.n_jump >= 2
        # - training agents for 5 jumps: self.time >= 12 or self.n_jump >= 6
        # - testing agents for 5 jumps: self.time >= 20 or self.n_jump >= 6

        return self.state, reward, terminated, truncated, {}

    # Render DQN's actions
    def render(self):
        print(f"Time: {self.time:.2f}, Position: {self.state[0]:.2f} m, Velocity: {self.state[1]:.2f} m/s")
