"""
Phase 2 — Habitat Policy Viewer
Watch the trained PPO policy navigate in the real Habitat scene.

Shows:
  - Left : RGB camera view from Habitat (what the robot sees)
  - Right : Depth sector bar + action values + reward history

Usage:
  python watch_habitat.py
  python watch_habitat.py --episodes 5

Requirements:
  conda activate habitat_m1
  export KMP_DUPLICATE_LIB_OK=TRUE
"""

import os
import sys
import time
import argparse
import math
import numpy as np

import matplotlib
matplotlib.use("macosx")   # change to TkAgg if macosx fails
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from habitat_nav_env import (
    HabitatNavEnv, CMD_NAMES,
    NUM_SECTORS, MAX_DEPTH, MIN_SAFE_DIST,
    MAX_LIN_VEL, MAX_ANG_VEL,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCENE_PATH      = "hm3d-example-habitat/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb"
BEST_MODEL_PATH = "./models/best_model/best_model.zip"
VECNORM_PATH    = "./models/vecnormalize.pkl"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model_and_env(seed: int = 42):
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"No model found at {BEST_MODEL_PATH}")
        print("Run train_ppo_habitat.py first.")
        sys.exit(1)

    raw_env = make_vec_env(
        lambda: Monitor(
            HabitatNavEnv(scene_path=SCENE_PATH, random_seed=seed)
        ),
        n_envs=1,
        seed=seed,
    )
    vec_env = VecNormalize.load(VECNORM_PATH, raw_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = PPO.load(BEST_MODEL_PATH, env=vec_env)
    return model, vec_env


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class HabitatPolicyViewer:

    def __init__(self):
        self.fig = plt.figure(figsize=(14, 6))
        self.fig.patch.set_facecolor("#1e1e2e")

        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            left=0.04, right=0.97,
            top=0.92, bottom=0.08,
            wspace=0.35, hspace=0.55,
        )

        self.ax_rgb    = self.fig.add_subplot(gs[:, 0])
        self.ax_depth  = self.fig.add_subplot(gs[0, 1])
        self.ax_action = self.fig.add_subplot(gs[1, 1])
        self.ax_reward = self.fig.add_subplot(gs[2, 1])

        for ax in [self.ax_rgb, self.ax_depth,
                   self.ax_action, self.ax_reward]:
            ax.set_facecolor("#2a2a3e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555")

        self.reward_history = []
        plt.ion()
        plt.show()

    def update(
        self,
        rgb_frame: np.ndarray,
        raw_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        step: int,
        goal_cmd: str,
    ):
        depth_sectors = raw_obs[:9]
        self.reward_history.append(reward)

        # ---- RGB camera view ----
        ax = self.ax_rgb
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        if rgb_frame is not None:
            ax.imshow(rgb_frame)
        ax.set_title(
            f"Step {step}  |  Goal: {goal_cmd.upper()}  |  "
            f"Reward: {reward:+.2f}",
            color="#cdd6f4", fontsize=10, pad=6,
        )
        ax.axis("off")

        # Overlay depth sector markers on the RGB image
        if rgb_frame is not None:
            h, w = rgb_frame.shape[:2]
            sector_w = w / NUM_SECTORS
            for i, d in enumerate(depth_sectors):
                color = "red" if d < MIN_SAFE_DIST else \
                        "orange" if d < MIN_SAFE_DIST * 2 else "cyan"
                cx = int((i + 0.5) * sector_w)
                ax.axvline(
                    cx, color=color, alpha=0.25, linewidth=1
                )
                ax.text(
                    cx, h - 12, f"{d:.1f}",
                    ha="center", color=color, fontsize=6,
                )

        # ---- Depth bar chart ----
        ax = self.ax_depth
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        colors = [
            "#f38ba8" if d < MIN_SAFE_DIST else
            "#fab387" if d < MIN_SAFE_DIST * 2 else
            "#89dceb"
            for d in depth_sectors
        ]
        bars = ax.bar(range(NUM_SECTORS), depth_sectors,
                      color=colors, width=0.7)
        ax.axhline(
            MIN_SAFE_DIST, color="#f38ba8",
            linestyle="--", linewidth=1, alpha=0.7,
        )
        ax.set_ylim(0, MAX_DEPTH + 0.3)
        ax.set_xticks(range(NUM_SECTORS))
        ax.set_xticklabels(
            [f"S{i}" for i in range(NUM_SECTORS)],
            color="#cdd6f4", fontsize=7,
        )
        ax.set_ylabel("Depth (m)", color="#cdd6f4", fontsize=8)
        ax.set_title("Depth sectors", color="#cdd6f4", fontsize=9)
        ax.tick_params(colors="#555")
        for i, (bar, d) in enumerate(zip(bars, depth_sectors)):
            ax.text(i, d + 0.1, f"{d:.1f}", ha="center",
                    color="#cdd6f4", fontsize=6)

        # ---- Action bars ----
        ax = self.ax_action
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        labels = ["vx", "vy", "ω"]
        colors = ["#89b4fa", "#a6e3a1", "#cba6f7"]
        ax.bar(labels, action, color=colors, width=0.5)
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title("Agent actions", color="#cdd6f4", fontsize=9)
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        for i, (lbl, val) in enumerate(zip(labels, action)):
            ax.text(
                i, val + (0.05 if val >= 0 else -0.13),
                f"{val:+.2f}", ha="center",
                color="#cdd6f4", fontsize=8,
            )

        # ---- Reward history ----
        ax = self.ax_reward
        ax.clear()
        ax.set_facecolor("#2a2a3e")
        if len(self.reward_history) > 1:
            xs = list(range(len(self.reward_history)))
            ax.plot(xs, self.reward_history,
                    color="#a6e3a1", linewidth=1)
            ax.fill_between(
                xs, self.reward_history, 0,
                alpha=0.15, color="#a6e3a1",
            )
            cumulative = np.cumsum(self.reward_history)
            running_mean = cumulative / (np.arange(
                len(cumulative)) + 1)
            ax.plot(xs, running_mean, color="#89b4fa",
                    linewidth=1, linestyle="--",
                    label="running mean")
            ax.legend(fontsize=7, labelcolor="#cdd6f4",
                      facecolor="#2a2a3e", edgecolor="#555")
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_ylabel("Reward", color="#cdd6f4", fontsize=8)
        ax.set_title("Reward per step", color="#cdd6f4", fontsize=9)
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

        self.fig.suptitle(
            "RaspBot PPO — Habitat-Sim Live View",
            color="#cdd6f4", fontsize=12, fontweight="bold",
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def reset_episode(self):
        self.reward_history.clear()

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def watch(n_episodes: int = 3, delay: float = 0.03):
    print("\n=== Habitat Policy Viewer ===")
    print("Close the window or Ctrl+C to stop.\n")

    model, vec_env = load_model_and_env(seed=42)
    raw_env = vec_env.venv.envs[0].env.env    # unwrap to HabitatNavEnv

    viewer = HabitatPolicyViewer()

    try:
        for ep in range(n_episodes):
            obs       = vec_env.reset()
            done      = False
            ep_reward = 0.0
            step      = 0
            goal_cmd  = CMD_NAMES[raw_env.goal_cmd]

            print(f"Episode {ep+1}/{n_episodes} — "
                  f"Goal: {goal_cmd.upper()}")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)

                ep_reward += float(reward[0])
                step      += 1

                # Get RGB frame directly from Habitat
                rgb_frame = raw_env.render()

                # Get raw (unnormalized) observation for depth display
                raw_obs = vec_env.get_original_obs()[0]

                viewer.update(
                    rgb_frame = rgb_frame,
                    raw_obs   = raw_obs,
                    action    = action[0],
                    reward    = float(reward[0]),
                    step      = step,
                    goal_cmd  = CMD_NAMES[raw_env.goal_cmd],
                )
                time.sleep(delay)

            print(f"  Episode {ep+1} done: "
                  f"reward={ep_reward:+.2f}  "
                  f"steps={step}  "
                  f"collision={info[0].get('collision', False)}")
            viewer.reset_episode()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()
        vec_env.close()


# ---------------------------------------------------------------------------
# Quick performance report
# ---------------------------------------------------------------------------

def report(n_episodes: int = 10):
    print(f"\n=== Habitat Performance Report ({n_episodes} eps) ===\n")

    model, vec_env = load_model_and_env(seed=0)
    raw_env = vec_env.venv.envs[0].env

    rewards, lengths, collisions = [], [], []

    for ep in range(n_episodes):
        obs      = vec_env.reset()
        done     = False
        ep_r     = 0.0
        ep_l     = 0
        ep_col   = False
        goal_cmd = CMD_NAMES[raw_env.goal_cmd]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = vec_env.step(action)
            ep_r += float(r[0])
            ep_l += 1
            if info[0].get("collision", False):
                ep_col = True

        rewards.append(ep_r)
        lengths.append(ep_l)
        collisions.append(int(ep_col))

        print(
            f"  Ep {ep+1:2d} | goal={goal_cmd:7s} | "
            f"reward={ep_r:+7.1f} | steps={ep_l:4d} | "
            f"{'COLLISION' if ep_col else 'safe'}"
        )

    print(f"\n  Mean reward   : {np.mean(rewards):.2f} "
          f"± {np.std(rewards):.2f}")
    print(f"  Mean length   : {np.mean(lengths):.0f} steps")
    print(f"  Survival rate : "
          f"{100*(1-np.mean(collisions)):.0f}%")
    vec_env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--report", action="store_true",
        help="Run performance report instead of watch mode",
    )
    args = parser.parse_args()

    if args.report:
        report(n_episodes=args.episodes)
    else:
        watch(n_episodes=args.episodes)