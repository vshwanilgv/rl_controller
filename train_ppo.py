"""
Phase 1 — Step 1.2
PPO Training Script for RobotNavEnv using Stable-Baselines3.

Features:
  - Vectorized parallel environments for faster data collection
  - VecNormalize for observation and reward normalization
  - TensorBoard logging
  - Checkpoint saving every N steps
  - Resume training from checkpoint with --resume flag

Usage:
  # Fresh training
  python train_ppo.py

  # Resume from checkpoint
  python train_ppo.py --resume

  # Custom timesteps
  python train_ppo.py --timesteps 500000

Requirements:
  pip install stable-baselines3 gymnasium numpy matplotlib tensorboard
"""

import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from robot_nav_env import RobotNavEnv

SAVE_DIR        = "./models"
LOG_DIR         = "./logs/ppo_robotnav"
CHECKPOINT_DIR  = "./models/checkpoints"
BEST_MODEL_PATH = "./models/best_model"
VECNORM_PATH    = "./models/vecnormalize.pkl"

os.makedirs(SAVE_DIR, dtype=None, exist_ok=True) if False else None
os.makedirs(SAVE_DIR,        exist_ok=True)
os.makedirs(LOG_DIR,         exist_ok=True)
os.makedirs(CHECKPOINT_DIR,  exist_ok=True)
os.makedirs(BEST_MODEL_PATH, exist_ok=True)


PPO_HYPERPARAMS = dict(
    # --- Core PPO ---
    learning_rate    = 3e-4,       # Adam LR — standard starting point
    n_steps          = 2048,       # steps per env before each update
    batch_size       = 64,         # minibatch size for gradient updates
    n_epochs         = 10,         # passes over collected data per update
    gamma            = 0.99,       # discount factor
    gae_lambda       = 0.95,       # GAE lambda for advantage estimation
    clip_range       = 0.2,        # PPO clipping parameter
    ent_coef         = 0.01,       # entropy bonus — encourages exploration
    vf_coef          = 0.5,        # value function loss weight
    max_grad_norm    = 0.5,        # gradient clipping

    # --- Network architecture ---
    # MLP with two hidden layers of 256 units each
    # Input: 16-dim obs → 256 → 256 → action (3) / value (1)
    policy_kwargs    = dict(
        net_arch = dict(pi=[256, 256], vf=[256, 256]),
    ),

    # --- Logging ---
    verbose          = 1,
    tensorboard_log  = LOG_DIR,
)

# Number of parallel environments
# M1 Mac: use 4 (CPU cores)
# Colab T4: use 8
N_ENVS = 4

# Total training timesteps
# ~300k is enough for basic navigation on M1 Mac (~20-30 min)
# Use 1M+ on Colab T4 for a well-converged policy
DEFAULT_TIMESTEPS = 300_000

# Save a checkpoint every this many steps (per env, so multiply by N_ENVS)
CHECKPOINT_FREQ = 50_000

def make_env(seed: int = 0):
    """Factory function — creates a single monitored environment."""
    def _init():
        env = RobotNavEnv(num_obstacles=8, random_seed=seed)
        env = Monitor(env)   # wraps env to log episode rewards/lengths
        return env
    return _init


def build_vec_env(n_envs: int = N_ENVS, seed: int = 0) -> VecNormalize:
    """
    Build a vectorized, normalized environment stack.
    VecNormalize tracks running mean/std of observations and rewards,
    normalizing them on the fly — critical for stable PPO training.
    """
    vec_env = make_vec_env(
        make_env(seed=seed),
        n_envs=n_envs,
        seed=seed,
    )
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,       # normalize observations
        norm_reward=True,    # normalize rewards
        clip_obs=10.0,       # clip normalized obs to [-10, 10]
        clip_reward=10.0,    # clip normalized rewards
        gamma=PPO_HYPERPARAMS["gamma"],
    )
    return vec_env


def build_eval_env(seed: int = 99) -> VecNormalize:
    """
    Separate evaluation environment.
    norm_reward=False so eval rewards are in original scale (interpretable).
    """
    vec_env = make_vec_env(
        make_env(seed=seed),
        n_envs=1,
        seed=seed,
    )
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,   # keep rewards unscaled for evaluation
        clip_obs=10.0,
        gamma=PPO_HYPERPARAMS["gamma"],
    )
    return vec_env

def build_callbacks(eval_env: VecNormalize) -> CallbackList:
    # Save a checkpoint every CHECKPOINT_FREQ steps
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // N_ENVS,  # per-env frequency
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_robotnav",
        save_vecnormalize=True,   # also saves VecNormalize stats
        verbose=1,
    )

    # Evaluate on a separate env every 10k steps, save best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_PATH,
        log_path=LOG_DIR,
        eval_freq=10_000 // N_ENVS,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    return CallbackList([checkpoint_cb, eval_cb])

def train(total_timesteps: int = DEFAULT_TIMESTEPS, resume: bool = False):
    print("\n" + "=" * 55)
    print("  Phase 1 — Step 1.2: PPO Training")
    print("=" * 55)
    print(f"  Environments  : {N_ENVS} parallel")
    print(f"  Total steps   : {total_timesteps:,}")
    print(f"  Resume        : {resume}")
    print(f"  Logs          : {LOG_DIR}")
    print(f"  Checkpoints   : {CHECKPOINT_DIR}")
    print("=" * 55 + "\n")

    # --- Build environments ---
    print("Building environments...")
    train_env = build_vec_env(n_envs=N_ENVS, seed=0)
    eval_env  = build_eval_env(seed=99)

    # --- Build or load model ---
    resume_path = os.path.join(SAVE_DIR, "ppo_robotnav_final.zip")

    if resume and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        model = PPO.load(
            resume_path,
            env=train_env,
            tensorboard_log=LOG_DIR,
            verbose=1,
        )
        # Restore VecNormalize running stats
        if os.path.exists(VECNORM_PATH):
            train_env = VecNormalize.load(VECNORM_PATH, train_env.venv)
            print(f"VecNormalize stats restored from: {VECNORM_PATH}")
    else:
        if resume:
            print("No checkpoint found — starting fresh training.")
        print("Initializing new PPO model...")
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            **PPO_HYPERPARAMS,
        )

    print(f"\nPolicy network:\n{model.policy}\n")

    # --- Sync eval env obs normalization with train env ---
    eval_env.obs_rms = train_env.obs_rms

    # --- Build callbacks ---
    callbacks = build_callbacks(eval_env)

    # --- Train ---
    print("Starting training... (view live in TensorBoard)\n")
    print(f"  tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps      = total_timesteps,
        callback             = callbacks,
        reset_num_timesteps  = not resume,
        progress_bar         = True,
    )

    # --- Save final model and VecNormalize stats ---
    final_path = os.path.join(SAVE_DIR, "ppo_robotnav_final")
    model.save(final_path)
    train_env.save(VECNORM_PATH)

    print(f"\nTraining complete.")
    print(f"  Final model : {final_path}.zip")
    print(f"  VecNormalize: {VECNORM_PATH}")
    print(f"  Best model  : {BEST_MODEL_PATH}/best_model.zip")

    train_env.close()
    eval_env.close()
    return model


def evaluate(n_episodes: int = 5):
    """Run a quick evaluation of the saved best model."""
    model_path   = os.path.join(BEST_MODEL_PATH, "best_model.zip")
    vecnorm_path = VECNORM_PATH

    if not os.path.exists(model_path):
        print("No saved model found. Run training first.")
        return

    print(f"\nEvaluating best model: {model_path}")

    # Build a fresh eval env
    raw_env = make_vec_env(make_env(seed=42), n_envs=1, seed=42)
    eval_env = VecNormalize.load(vecnorm_path, raw_env)
    eval_env.training = False     # freeze normalization stats during eval
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env)

    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            ep_length += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        print(
            f"  Episode {ep + 1}: "
            f"reward={ep_reward:+.2f}  length={ep_length} steps"
        )

    print(f"\n  Mean reward : {np.mean(episode_rewards):.2f} "
          f"± {np.std(episode_rewards):.2f}")
    print(f"  Mean length : {np.mean(episode_lengths):.0f} steps")

    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on RobotNavEnv")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last saved checkpoint",
    )
    parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and only run evaluation on saved model",
    )
    args = parser.parse_args()

    if args.eval_only:
        evaluate()
    else:
        train(total_timesteps=args.timesteps, resume=args.resume)
        evaluate(n_episodes=5)