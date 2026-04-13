"""
Phase 1 — Step 1.3
Evaluation & Visualization Script for the trained PPO policy.

Modes:
  --watch       : Watch the trained agent navigate in real time (render window)
  --report      : Run N episodes and print a full performance report
  --compare     : Side-by-side comparison of trained vs random agent
  --save-gif    : Save a GIF of one episode (requires pillow)

Usage:
  python evaluate_policy.py --watch
  python evaluate_policy.py --report --episodes 20
  python evaluate_policy.py --compare
  python evaluate_policy.py --save-gif

Requirements:
  pip install stable-baselines3 gymnasium numpy matplotlib pillow
"""

import os
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("macosx")         
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from collections import defaultdict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from robot_nav_env import (
    RobotNavEnv, CMD_NAMES, CMD_FORWARD, CMD_LEFT, CMD_RIGHT, CMD_STOP,
    NUM_SECTORS, SECTOR_FOV, MAX_DEPTH, MIN_SAFE_DIST,
    GRID_SIZE, ROBOT_RADIUS, MAX_LIN_VEL, MAX_ANG_VEL,
)

import math

BEST_MODEL_PATH = "./models/best_model/best_model.zip"
VECNORM_PATH    = "./models/vecnormalize.pkl"

matplotlib.use("macosx")

def load_model_and_env(seed: int = 42):
    """Load trained PPO model with its VecNormalize stats."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No model found at {BEST_MODEL_PATH}. Run train_ppo.py first."
        )

    raw_env = make_vec_env(
        lambda: Monitor(RobotNavEnv(num_obstacles=8, random_seed=seed)),
        n_envs=1,
        seed=seed,
    )
    vec_env = VecNormalize.load(VECNORM_PATH, raw_env)
    vec_env.training    = False  
    vec_env.norm_reward = False   

    model = PPO.load(BEST_MODEL_PATH, env=vec_env)
    return model, vec_env


def get_base_env(vec_env) -> RobotNavEnv:
    """Return the underlying RobotNavEnv from a VecEnv wrapper stack."""
    env = vec_env.venv.envs[0]

    while hasattr(env, "env"):
        env = env.env

    return env


class PolicyVisualizer:
    """
    Real-time visualization of the PPO policy navigating.

    Layout:
      Left panel  — 2D top-down map (robot, obstacles, depth rays, trail)
      Right panel — live telemetry (depth bar chart, action values, reward)
    """

    def __init__(self):
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.patch.set_facecolor("#1e1e2e")

        gs = gridspec.GridSpec(3, 2, figure=self.fig,
                               left=0.05, right=0.97,
                               top=0.93, bottom=0.08,
                               wspace=0.3, hspace=0.5)

        self.ax_map    = self.fig.add_subplot(gs[:, 0])   # full left column
        self.ax_depth  = self.fig.add_subplot(gs[0, 1])   # depth sectors
        self.ax_action = self.fig.add_subplot(gs[1, 1])   # action values
        self.ax_reward = self.fig.add_subplot(gs[2, 1])   # reward history

        for ax in [self.ax_map, self.ax_depth, self.ax_action, self.ax_reward]:
            ax.set_facecolor("#2a2a3e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555")

        self.reward_history = []
        self.trail_x        = []
        self.trail_y        = []

        plt.ion()
        plt.show()

    def update(self, env: RobotNavEnv, obs: np.ndarray,
               action: np.ndarray, reward: float, step: int):

        depth_sectors = obs[:9]
        velocity      = obs[9:12]   # normalized — just for display shape
        goal_idx      = int(np.argmax(obs[12:16]))
        self.reward_history.append(reward)

        self.trail_x.append(env.robot_x)
        self.trail_y.append(env.robot_y)
        if len(self.trail_x) > 200:
            self.trail_x.pop(0)
            self.trail_y.pop(0)

        ax = self.ax_map
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect("equal")
        ax.set_title(
            f"Step {step}  |  Goal: {CMD_NAMES[goal_idx].upper()}  |  "
            f"Reward: {reward:+.2f}",
            color="#cdd6f4", fontsize=10, pad=6,
        )
        ax.tick_params(colors="#555")

        # Grid lines
        for i in range(0, int(GRID_SIZE) + 1):
            ax.axhline(i, color="#333", linewidth=0.3, zorder=0)
            ax.axvline(i, color="#333", linewidth=0.3, zorder=0)

        # Robot trail
        if len(self.trail_x) > 1:
            for i in range(1, len(self.trail_x)):
                alpha = 0.1 + 0.6 * (i / len(self.trail_x))
                ax.plot(
                    [self.trail_x[i-1], self.trail_x[i]],
                    [self.trail_y[i-1], self.trail_y[i]],
                    color="#89b4fa", alpha=alpha, linewidth=1.2,
                )

        # Obstacles
        for obs_obj in env.obstacles:
            rect = patches.Rectangle(
                (obs_obj.x - obs_obj.w / 2, obs_obj.y - obs_obj.h / 2),
                obs_obj.w, obs_obj.h,
                linewidth=1, edgecolor="#7f849c", facecolor="#45475a",
            )
            ax.add_patch(rect)

        # Depth rays
        half_fov    = math.radians(SECTOR_FOV / 2)
        sector_step = math.radians(SECTOR_FOV) / (NUM_SECTORS - 1)

        for i in range(NUM_SECTORS):
            rel_angle   = -half_fov + i * sector_step
            world_angle = env.robot_theta + rel_angle
            # Use raw env depth, not normalized obs
            d = env._get_depth_sectors()[i]
            ex = env.robot_x + d * math.cos(world_angle)
            ey = env.robot_y + d * math.sin(world_angle)
            color = "#f38ba8" if d < MIN_SAFE_DIST else \
                    "#fab387" if d < MIN_SAFE_DIST * 2 else "#89dceb"
            ax.plot(
                [env.robot_x, ex], [env.robot_y, ey],
                color=color, alpha=0.5, linewidth=1.2, zorder=3,
            )
            # Endpoint dot
            ax.plot(ex, ey, "o", color=color, markersize=3, zorder=4)

        # Robot body
        robot_circle = plt.Circle(
            (env.robot_x, env.robot_y), ROBOT_RADIUS,
            color="#a6e3a1", zorder=6,
        )
        ax.add_patch(robot_circle)

        # Heading arrow
        hx = env.robot_x + ROBOT_RADIUS * 2.2 * math.cos(env.robot_theta)
        hy = env.robot_y + ROBOT_RADIUS * 2.2 * math.sin(env.robot_theta)
        ax.annotate(
            "", xy=(hx, hy), xytext=(env.robot_x, env.robot_y),
            arrowprops=dict(arrowstyle="->", color="#cdd6f4", lw=2),
            zorder=7,
        )

        # Legend
        ax.text(0.2, 9.7, "safe", color="#89dceb", fontsize=7)
        ax.text(1.2, 9.7, "near", color="#fab387", fontsize=7)
        ax.text(2.2, 9.7, "danger", color="#f38ba8", fontsize=7)

        # ---- DEPTH BAR CHART ----
        ax = self.ax_depth
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        raw_depths = env._get_depth_sectors()
        colors = [
            "#f38ba8" if d < MIN_SAFE_DIST else
            "#fab387" if d < MIN_SAFE_DIST * 2 else
            "#89dceb"
            for d in raw_depths
        ]
        bars = ax.bar(range(NUM_SECTORS), raw_depths, color=colors, width=0.7)
        ax.axhline(MIN_SAFE_DIST, color="#f38ba8", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Safety ({MIN_SAFE_DIST}m)")
        ax.set_ylim(0, MAX_DEPTH + 0.3)
        ax.set_xticks(range(NUM_SECTORS))
        ax.set_xticklabels([f"S{i}" for i in range(NUM_SECTORS)],
                           color="#cdd6f4", fontsize=8)
        ax.set_ylabel("Depth (m)", color="#cdd6f4", fontsize=8)
        ax.set_title("Depth sectors", color="#cdd6f4", fontsize=9)
        ax.tick_params(colors="#555")
        ax.legend(fontsize=7, labelcolor="#cdd6f4",
                  facecolor="#2a2a3e", edgecolor="#555")

        # Value labels on bars
        for i, (bar, d) in enumerate(zip(bars, raw_depths)):
            ax.text(i, d + 0.1, f"{d:.1f}", ha="center",
                    color="#cdd6f4", fontsize=6)

        # ---- ACTION BAR CHART ----
        ax = self.ax_action
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        action_labels = ["vx", "vy", "ω"]
        action_colors = ["#89b4fa", "#a6e3a1", "#cba6f7"]
        ax.bar(action_labels, action, color=action_colors, width=0.5)
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("Normalised value", color="#cdd6f4", fontsize=8)
        ax.set_title("Agent actions", color="#cdd6f4", fontsize=9)
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        for i, (label, val) in enumerate(zip(action_labels, action)):
            ax.text(i, val + (0.05 if val >= 0 else -0.12),
                    f"{val:+.2f}", ha="center",
                    color="#cdd6f4", fontsize=8)

        # ---- REWARD HISTORY ----
        ax = self.ax_reward
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        if len(self.reward_history) > 1:
            xs = range(len(self.reward_history))
            ax.plot(list(xs), self.reward_history,
                    color="#a6e3a1", linewidth=1)
            ax.fill_between(
                list(xs), self.reward_history, 0,
                alpha=0.2, color="#a6e3a1",
            )
            cumulative = np.cumsum(self.reward_history)
            ax.plot(list(xs), cumulative / (np.arange(len(cumulative)) + 1),
                    color="#89b4fa", linewidth=1, linestyle="--",
                    label="running mean")
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_ylabel("Reward", color="#cdd6f4", fontsize=8)
        ax.set_title("Reward per step", color="#cdd6f4", fontsize=9)
        ax.tick_params(colors="#cdd6f4")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=7, labelcolor="#cdd6f4",
                      facecolor="#2a2a3e", edgecolor="#555")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

        self.fig.suptitle(
            "RaspBot PPO Policy — Live Evaluation",
            color="#cdd6f4", fontsize=12, fontweight="bold",
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

def watch(n_episodes: int = 3, delay: float = 0.02):
    """Watch the trained agent navigate with full telemetry dashboard."""
    print("\n=== WATCH MODE ===")
    print("Close the window or press Ctrl+C to stop.\n")

    model, vec_env = load_model_and_env(seed=42)

    # Get the underlying raw env for direct state access
    raw_env = get_base_env(vec_env)

    viz = PolicyVisualizer()

    try:
        for ep in range(n_episodes):
            obs = vec_env.reset()
            done  = False
            ep_reward = 0.0
            step  = 0

            print(f"Episode {ep + 1}/{n_episodes} — "
                  f"Goal: {CMD_NAMES[raw_env.goal_cmd].upper()}")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)

                ep_reward += float(reward[0])
                step += 1

                # Pass raw (unnormalized) obs to visualizer for depth display
                raw_obs = vec_env.get_original_obs()[0]
                viz.update(
                    env    = raw_env,
                    obs    = raw_obs,
                    action = action[0],
                    reward = float(reward[0]),
                    step   = step,
                )
                time.sleep(delay)

            print(f"  Finished: reward={ep_reward:+.2f}  "
                  f"length={step}  "
                  f"collision={info[0].get('collision', False)}")
            viz.reward_history.clear()
            viz.trail_x.clear()
            viz.trail_y.clear()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        viz.close()
        vec_env.close()


def report(n_episodes: int = 20):
    """Run N episodes and print a detailed performance report."""
    print(f"\n=== PERFORMANCE REPORT ({n_episodes} episodes) ===\n")

    model, vec_env = load_model_and_env(seed=0)

    stats = defaultdict(list)

    for ep in range(n_episodes):
        obs       = vec_env.reset()
        raw_env   = get_base_env(vec_env)
        goal_cmd  = CMD_NAMES[raw_env.goal_cmd]
        done      = False
        ep_reward = 0.0
        ep_steps  = 0
        min_depth_seen = MAX_DEPTH
        collided  = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1

            raw_obs = vec_env.get_original_obs()[0]
            min_d   = float(raw_obs[:9].min())
            if min_d < min_depth_seen:
                min_depth_seen = min_d
            if info[0].get("collision", False):
                collided = True

        stats["reward"].append(ep_reward)
        stats["length"].append(ep_steps)
        stats["min_depth"].append(min_depth_seen)
        stats["collided"].append(int(collided))
        stats["survived"].append(int(ep_steps >= 999))
        stats["goal"].append(goal_cmd)

        print(
            f"  Ep {ep+1:2d} | goal={goal_cmd:7s} | "
            f"reward={ep_reward:+7.1f} | "
            f"steps={ep_steps:4d} | "
            f"min_depth={min_depth_seen:.2f}m | "
            f"{'COLLISION' if collided else 'safe':9s}"
        )

    print("\n" + "-" * 60)
    print(f"  Mean reward     : {np.mean(stats['reward']):.2f} "
          f"± {np.std(stats['reward']):.2f}")
    print(f"  Mean ep length  : {np.mean(stats['length']):.0f} steps")
    print(f"  Survival rate   : "
          f"{100 * np.mean(stats['survived']):.0f}%  "
          f"({sum(stats['survived'])}/{n_episodes} episodes)")
    print(f"  Collision rate  : "
          f"{100 * np.mean(stats['collided']):.0f}%")
    print(f"  Mean min depth  : {np.mean(stats['min_depth']):.2f}m  "
          f"(safety threshold: {MIN_SAFE_DIST}m)")

    # Per-command breakdown
    print("\n  Per-command breakdown:")
    for cmd in CMD_NAMES:
        idxs = [i for i, g in enumerate(stats["goal"]) if g == cmd]
        if not idxs:
            continue
        cmd_rewards = [stats["reward"][i] for i in idxs]
        cmd_survived = [stats["survived"][i] for i in idxs]
        print(
            f"    {cmd:8s} : mean_reward={np.mean(cmd_rewards):+.1f}  "
            f"survival={100*np.mean(cmd_survived):.0f}%  "
            f"(n={len(idxs)})"
        )

    print("-" * 60)
    vec_env.close()


def compare(n_episodes: int = 10):
    """Compare trained PPO agent vs random action baseline."""
    print(f"\n=== TRAINED vs RANDOM ({n_episodes} episodes each) ===\n")

    def run_episodes(use_model: bool, n: int):
        env = RobotNavEnv(num_obstacles=8, random_seed=7)
        if use_model:
            model, vec_env = load_model_and_env(seed=7)

        rewards, lengths, collisions = [], [], []

        for ep in range(n):
            if use_model:
                obs = vec_env.reset()
                done = False
                ep_r, ep_l, ep_c = 0.0, 0, False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, done, info = vec_env.step(action)
                    ep_r += float(r[0])
                    ep_l += 1
                    if info[0].get("collision", False):
                        ep_c = True
            else:
                obs, _ = env.reset()
                done = False
                ep_r, ep_l, ep_c = 0.0, 0, False
                while not done:
                    action = env.action_space.sample()
                    obs, r, term, trunc, info = env.step(action)
                    ep_r += r
                    ep_l += 1
                    done = term or trunc
                    if info.get("collision", False):
                        ep_c = True

            rewards.append(ep_r)
            lengths.append(ep_l)
            collisions.append(int(ep_c))

        if use_model:
            vec_env.close()
        else:
            env.close()

        return rewards, lengths, collisions

    print("Running trained agent...")
    t_r, t_l, t_c = run_episodes(use_model=True,  n=n_episodes)
    print("Running random agent...")
    r_r, r_l, r_c = run_episodes(use_model=False, n=n_episodes)

    print("\n" + "=" * 50)
    print(f"{'Metric':<22} {'Trained':>12} {'Random':>12}")
    print("-" * 50)
    print(f"{'Mean reward':<22} {np.mean(t_r):>+11.1f} {np.mean(r_r):>+11.1f}")
    print(f"{'Std reward':<22} {np.std(t_r):>11.1f} {np.std(r_r):>11.1f}")
    print(f"{'Mean ep length':<22} {np.mean(t_l):>11.0f} {np.mean(r_l):>11.0f}")
    print(f"{'Collision rate':<22} {100*np.mean(t_c):>10.0f}% {100*np.mean(r_c):>10.0f}%")
    print("=" * 50)


def save_gif(output_path: str = "policy_episode.gif", max_steps: int = 300):
    """Save one episode as an animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("pillow not installed. Run: pip install pillow")
        return

    print(f"\nSaving GIF to: {output_path}")

    model, vec_env = load_model_and_env(seed=5)
    raw_env = get_base_env(vec_env)

    # Use rgb_array render mode on a fresh env
    render_env = RobotNavEnv(render_mode="rgb_array", num_obstacles=8,
                             random_seed=5)
    render_env.reset()
    render_env.goal_cmd    = raw_env.goal_cmd
    render_env.obstacles   = raw_env.obstacles
    render_env.robot_x     = raw_env.robot_x
    render_env.robot_y     = raw_env.robot_y
    render_env.robot_theta = raw_env.robot_theta

    frames = []
    obs    = vec_env.reset()
    done   = False
    step   = 0

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = vec_env.step(action)

        # Mirror state to render env
        render_env.robot_x     = raw_env.robot_x
        render_env.robot_y     = raw_env.robot_y
        render_env.robot_theta = raw_env.robot_theta
        render_env.robot_vx    = raw_env.robot_vx
        render_env.robot_vy    = raw_env.robot_vy
        render_env.robot_omega = raw_env.robot_omega
        render_env.step_count  = raw_env.step_count
        render_env.goal_cmd    = raw_env.goal_cmd

        frame = render_env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
        step += 1

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=40,    # ms per frame = ~25 fps
            loop=0,
        )
        print(f"Saved {len(frames)} frames → {output_path}")
    else:
        print("No frames captured — check render mode.")

    render_env.close()
    vec_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO policy on RobotNavEnv"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch agent navigate in real time",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print full performance report",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare trained vs random agent",
    )
    parser.add_argument(
        "--save-gif", action="store_true",
        help="Save one episode as a GIF",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes for --report or --compare (default: 10)",
    )
    args = parser.parse_args()

    if args.watch:
        watch(n_episodes=3)
    elif args.report:
        report(n_episodes=args.episodes)
    elif args.compare:
        compare(n_episodes=args.episodes)
    elif args.save_gif:
        save_gif()
    else:
        # Default: run report then compare
        report(n_episodes=args.episodes)
        compare(n_episodes=args.episodes)