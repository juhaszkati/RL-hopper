import numpy as np
import random
import torch

from itertools import count
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from Costum_Environment import MassActuatorEnv
from Neural_Network import DQN
from Memory import ReplayMemory, Transition
from Device import device
from Monitoring import plot_position_and_force_last_ep, save_agent_logs


env = MassActuatorEnv()


# Training parameters
batch_size = 512
gamma = 0.99  # discount factor
eps_start = 0.99  # exploration rate
eps_end = 0.01
eps_decay = 10000
tau = 0.005  # update rate of target values
lr = 0.0005  # learning rate
memory_len = 10000
num_episodes = 3000

# Determining actions and observations
n_actions = env.action_space.n  # action space is discrete: .n
n_observations = env.observation_space.shape[0]  # observation space is continuous: .shape[0]


# Action selection
def select_action(state, policy_net, steps_done):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)  # eps decay over time

    if sample > eps_threshold:
        with torch.no_grad():  # if not random action no need for gradient calculation
            return policy_net(state).max(1).indices.view(1, 1)  # choosing the action with the highest expected value
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)  # picking random action


# Optimizing
def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < batch_size:
        return None, None, None
    transitions = memory.sample(batch_size)  # sampling random transitions
    batch = Transition(*zip(*transitions))  # transposing the batch to the correct shape

    # Formatting batch
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)  # boolean mask of which transition has next state
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # collecting data using the mask
    state_batch = torch.cat(batch.state)    # assembling the batch tensors
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Computing current Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Computing V(s_{t+1})
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Computing expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    td_error = expected_state_action_values - state_action_values

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()  # combination of MSE and MAE (Mean Squared/Absolute Error)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()   # resetting the gradients to ensure unbiased updates
    loss.backward()         # calculating new gradient values

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10)  # ensuring stability and avoiding gradient explosion
    optimizer.step()        # updating the network weights

    return loss, state_action_values, td_error


# Training cycle
def training_agents(termination, initial_state):
    # Creating neural networks
    policy_net = DQN(n_observations, n_actions).to(device)  # creating and moving net to device (gpu if possible)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())  # ensuring both net starts with the same weights

    # AdamW: Adaptive Moment Estimation W: weight decay regulating apart from derivatives
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr, weight_decay=0.0001)
    memory = ReplayMemory(memory_len)  # setting the replay buffer size

    # Information for monitoring
    steps_done = 0  # tracking the number of steps
    last_episode = []

    filename = f"logs/all_ep/agent_{termination:.2f}_all_episodes.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"")

    recent_rewards = deque([], maxlen=200)
    agent_reward = 0
    success_number = 0

    # Uncomment the line below to monitor agent training
    # writer = SummaryWriter(log_dir=f"runs/agent_{termination:.2f}/")

    for i_episode in range(num_episodes):
        # Initializing the environment
        options = {
            "termination": termination,
            "initial_state": initial_state
        }
        state = env.reset(options=options)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        trajectory = []

        # Simulating steps
        for t in count():
            steps_done += 1

            # Computing the next state
            action = select_action(state, policy_net, steps_done)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            total_reward += reward.item()
            trajectory.append(
                (state.clone(), action.item(), reward.item(), terminated))

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Storing the transition in memory
            memory.push(state, action, next_state, reward)

            # Moving to the next state
            state = next_state

            # Performing one step of the optimization (on the policy network)
            loss, state_action_values, td_error = optimize_model(policy_net, target_net, memory, optimizer)

            # Uncomment the lines below to monitor agent training
            # epsilon = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay)

            # writer.add_scalar("Reward", reward.item(), steps_done)
            # writer.add_scalar("Epsilon", epsilon, steps_done)

            # if loss is not None:
            #    writer.add_scalar("Loss", loss.item(), steps_done)

            # if state_action_values is not None and isinstance(state_action_values, torch.Tensor):
            #    writer.add_scalar("Q_max", state_action_values.max().item(), steps_done)
            #    writer.add_scalar("TD_error", td_error.abs().mean().item(), steps_done)

            # if td_error is not None:
            #    writer.add_histogram("TD_error/distribution", td_error, steps_done)

            # for name, param in policy_net.named_parameters():
            #   if param.grad is not None:
            #       writer.add_scalar(f"Gradients/{name}_mean", param.grad.abs().mean(), steps_done)
            #        writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), steps_done)
            #        writer.add_histogram(f"gradients/{name}", param.grad, steps_done)

            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                if env.success > 0:
                    success_number = 1
                elif env.success == 0:
                    success_number = 0
                recent_rewards.append(success_number)
                agent_reward = sum(recent_rewards)/len(recent_rewards)
                last_episode = trajectory.copy()
                save_agent_logs(trajectory, termination, i_episode+1, total_reward, env.dt)
                print(f"Episode {i_episode + 1}, Total reward: {total_reward}")

                # Uncomment the line below to monitor agent training
                # writer.add_scalar("T_Reward", total_reward, steps_done)
                break

        if agent_reward > 0.8:
            print(f"Agent successfully trained after {i_episode+1} episodes")
            break

    torch.save(policy_net.state_dict(), f"agents/trained_agent_{termination}.pth")
    plot_position_and_force_last_ep(termination, last_episode, env.dt, env.max_force)

    # Uncomment the line below to monitor agent training
    # writer.close()

# Choose terimination limit of agents to be trained
# terminations = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
# terminations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
terminations = [0.3]
initial_states = np.array([5, 0])
for i in terminations:
    training_agents(i, initial_states)
    print(f"Agent {i} complete")

print('Complete')
