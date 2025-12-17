import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_position_and_force_last_ep(termination, episode, dt, max_force):
    if not episode:
        print("No saved episode.")
        return

    time_steps = [i * dt for i in range(len(episode))]

    positions = [s.squeeze().cpu().numpy()[0] for s, _, _, _ in episode]
    actions = [a for _, a, _, _ in episode]
    force_values = [-max_force, 0, max_force]
    forces = [force_values[a] for a in actions]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time_steps, positions, marker='o', linestyle='-', color='blue')
    ax1.set_ylabel("y [m]")
    ax1.set_title("Position and Force over Time")
    ax1.grid(True)
    goal = 5.0
    termination_line = termination
    ax1.axhline(goal, color='black', linestyle='--', linewidth=2)
    ax1.axhline(termination_line, color='red', linestyle='--', linewidth=2)

    ax2.step(time_steps, forces, where='post', color='darkred')
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("F [N]")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"figures/last_ep/agent_{termination:.2f}_last_episode.png")


def save_agent_logs(trajectory, termination, episode_number, total_reward, dt):
    filename = f"logs/all_ep/agent_{termination:.2f}_all_episodes.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\nEpisode: {episode_number} | Total reward: {total_reward}\n")
        f.write(f"{'Step':<6}{'Time':<8}{'x':<12}{'x_dot':<12}{'Action':<8}{'Reward':<10}{'Terminated':<12}\n")
        f.write("-" * 70 + "\n")

        for step_idx, (state, action, reward, terminated) in enumerate(trajectory):
            state_values = state.squeeze().tolist()
            time = step_idx * dt
            f.write(
                f"{step_idx:<6}{time:<8.4f}{state_values[0]:<12.4f}{state_values[1]:<12.4f}"
                f"{action:<8}{reward:<10.4f}{terminated:<12}\n")


def plot_basin_bw(matrix, batch):
    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = ListedColormap(['white', 'black'])

    im = ax.imshow(matrix, cmap=cmap, origin='lower',
                   interpolation='nearest', aspect='equal')

    ax.set_xlabel(r"$y_l$ [m]", fontsize=12)
    ax.set_ylabel(r"$y_0$ [m]", fontsize=12)
    ax.set_title("Binary Safety Map", fontsize=14)

    ax.set_xticks(range(matrix.shape[1]))
    n_positions = matrix.shape[0]
    ytick_labels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ytick_positions = np.linspace(0, n_positions - 1, len(ytick_labels))
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)

    plt.tight_layout()
    plt.savefig(f"figures/basins/agent_{batch}_bw_basin.png", dpi=300)


def save_basin_bw(matrix, batch):
    filename = f"logs/basins/agent_{batch}_bw_episodes.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\nAgents: {batch}\n")
        f.write("-" * 60 + "\n")
        np.savetxt(f, matrix, fmt="%d", delimiter=" ", newline="\n")
        f.write("\n")


def plot_basin_color(matrix, batch):
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(matrix, cmap='twilight', vmin=-1, vmax=1,
                   origin='lower', interpolation='none', aspect='equal')

    ax.set_xlabel(r"$y_l$ [m]", fontsize=12)
    ax.set_ylabel(r"$y_0$ [m]", fontsize=12)
    ax.set_title("Performance Safety Map", fontsize=14)

    ax.set_xticks(range(matrix.shape[1]))
    n_positions = matrix.shape[0]
    ytick_labels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ytick_positions = np.linspace(0, n_positions - 1, len(ytick_labels))
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)

    plt.tight_layout()
    plt.savefig(f"figures/basins/agent_{batch}_basin.png", dpi=300)


def save_basin_color(matrix, batch):
    filename = f"logs/basins/agent_{batch}_episodes.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\nAgents: {batch}\n")
        f.write("-" * 60 + "\n")
        np.savetxt(f, matrix, fmt="%d", delimiter=" ", newline="\n")
        f.write("\n")


def plot_position_and_force_test_ep(termination, episode, dt, max_force):
    if not episode:
        print("No saved episode.")
        return

    time_steps = [i * dt for i in range(len(episode))]

    positions = [s.squeeze().cpu().numpy()[0] for s, _, _, _ in episode]
    actions = [a for _, a, _, _ in episode]
    force_values = [-max_force, 0, max_force]
    forces = [force_values[a] for a in actions]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time_steps, positions, marker='o', linestyle='-', color='blue')
    ax1.set_ylabel("y [m]")
    ax1.set_title("Position and Force over Time")
    ax1.grid(True)
    goal = 5.0
    termination_line = termination
    ax1.axhline(goal, color='black', linestyle='-', linewidth=2)
    ax1.axhline(goal - 0.25, color='grey', linestyle='--', linewidth=2)
    ax1.axhline(goal + 0.25, color='grey', linestyle='--', linewidth=2)
    ax1.axhline(termination_line, color='red', linestyle='--', linewidth=2)

    ax2.step(time_steps, forces, where='post', color='darkred')
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("F [N]")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"figures/test_ep/agent_{termination}_last_episode.png")


def save_agent_logs_test(trajectory, termination, episode_number, total_reward, dt):
    filename = f"logs/test_ep/agent_{termination:.2f}_episodes.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\nEpisode: {episode_number} | Total reward: {total_reward}\n")
        f.write(f"{'Step':<6}{'Time':<8}{'x':<12}{'x_dot':<12}{'Action':<8}{'Reward':<10}{'Terminated':<12}\n")
        f.write("-" * 70 + "\n")

        for step_idx, (state, action, reward, terminated) in enumerate(trajectory):
            state_values = state.squeeze().tolist()
            time = step_idx * dt
            f.write(
                f"{step_idx:<6}{time:<8.2f}{state_values[0]:<12.4f}{state_values[1]:<12.4f}"
                f"{action:<8}{reward:<10.4f}{terminated:<12}\n")
