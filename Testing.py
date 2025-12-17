import numpy as np
import torch

from itertools import count
from Costum_Environment import MassActuatorEnv
from Neural_Network import DQN
from Monitoring import plot_basin_bw, save_basin_bw, save_agent_logs_test,  \
    plot_position_and_force_test_ep, plot_basin_color, save_basin_color

device = torch.device("cpu")

env = MassActuatorEnv()

# Determining actions and observations
n_actions = env.action_space.n  # action space is discrete: .n
n_observations = env.observation_space.shape[0]  # observation space is continuous: .shape[0]


# Testing cycle
def testing_agents(termination, initial_state):
    global success
    # Creating neural networks
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load(f"agents/trained_agent_{termination}.pth"))
    policy_net.eval()   # setting agent in an evaluation mode with no all_ep

    # Initializing the environment
    options = {
        "termination": 0.1,     # fixed physical termination limit for testing agents
        "initial_state": initial_state
    }

    state = env.reset(options=options)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    total_reward = 0
    trajectory = []

    # Simulating steps
    for t in count():
        # Action selection
        with torch.no_grad():
            action = policy_net(state).max(1).indices

        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        total_reward += reward.item()
        trajectory.append((state.clone(), action.item(), reward.item(), terminated))

        # Moving to the next state
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        if done:

            print(env.jump_start, env.n_jump, env.success)
            save_agent_logs_test(trajectory, termination, 1, total_reward, env.dt)
            if terminated:
                success = 1     # 1 if creating Performance Safety Map
            elif env.success > 0:
                success = 0     # 0 if creating Performance Safety Map
            elif truncated and env.success == 0:
                success = 0
                # Uncomment lines below for creating Performance Safety Map
                # min_distance = env.y_max - env.goal
                # success = np.sign(min_distance) * 1 * abs(min_distance)/(1 + abs(min_distance))

            break
    # Uncomment the line below only when running test for a single (or few) initial condition
    # plot_position_and_force_test_ep(termination, trajectory, env.dt, env.max_force)
    return success


# Choose terimination limit of agents to be tested
# terminations = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# terminations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
terminations = [0.3]

# Choose initial conditions
x_values = np.round(np.arange(2.1, 20.1, 0.1), 1)
# x_values = [10]

plot_matrix = np.empty((len(x_values), len(terminations)), dtype=int)     # int if black/white safety map

for j, termination in enumerate(terminations):
    for i, x in enumerate(x_values):
        state = [x, 0]
        print(f"Testing state: {state}")
        success = testing_agents(termination, state)    # Fail: 0, Success: 1
        plot_matrix[i, j] = success
        print(f"[{i}]")
print(plot_matrix)

plot_basin_bw(plot_matrix, 'BatchA1')  # plot_basin_color if creating Performance Safety Map
save_basin_bw(plot_matrix, 'BatchA1')  # save_basin_color if creating Performance Safety Map
