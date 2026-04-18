"""
Phase 2 — HabitatNavEnv
Habitat-Sim environment for RaspBot PPO training.

Identical observation/action interface to Phase 1 RobotNavEnv so the
same train_ppo.py script works without modification.

Observation space (16-dim):
  [0:9]   — 9-sector depth readings (metres, from D435i-matched depth sensor)
  [9:12]  — velocity state [vx, 0.0, omega]  (vy always 0 — non-holonomic)
  [12:16] — goal command one-hot [forward, left, right, stop]

Action space (3-dim continuous, [-1, 1]):
  [0] vx    — forward/backward velocity (scaled to MAX_LIN_VEL)
  [1] ignored (kept for interface compatibility with Phase 1)
  [2] omega — angular velocity (scaled to MAX_ANG_VEL)

Continuous velocity control:
  Each step sets vx and omega directly on the Habitat agent via
  velocity_control — no discrete action snapping. This eliminates
  the stop-and-go problem and allows smooth deceleration.

Requirements (inside habitat_m1 conda env):
  conda activate habitat_m1
  pip install gymnasium numpy

Usage:
  python habitat_nav_env.py
  (runs a 200-step sanity check — same as Phase 1)
"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import habitat_sim
from habitat_sim.utils.common import quat_to_angle_axis

# ---------------------------------------------------------------------------
# Constants — match Phase 1 exactly
# ---------------------------------------------------------------------------

MAX_DEPTH       = 5.0           # metres — sensor max range
MIN_SAFE_DIST   = 0.4           # metres — safety threshold
NUM_SECTORS     = 9             # depth sectors across front arc
SECTOR_FOV      = 180.0         # degrees — total front field of view
MAX_LIN_VEL     = 0.5           # m/s — max forward speed
MAX_ANG_VEL     = 1.0           # rad/s — max angular speed
DT              = 0.02          # seconds — control timestep (50 Hz)
MAX_STEPS       = 1000          # max steps per episode

# Depth sensor resolution — low res is fine for sector binning
DEPTH_H         = 64
DEPTH_W         = 128

# Camera height matching your existing scripts (0.3m from agent base)
SENSOR_HEIGHT   = 0.3

# Goal commands — identical to Phase 1
CMD_FORWARD  = 0
CMD_LEFT     = 1
CMD_RIGHT    = 2
CMD_STOP     = 3
CMD_NAMES    = ["forward", "left", "right", "stop"]

# Reward weights — same as Phase 1
R_PROGRESS      =  1.0
R_COLLISION     = -10.0
R_SMOOTH        = -0.05
R_STEP          = -0.01


# ---------------------------------------------------------------------------
# Habitat configuration builder
# ---------------------------------------------------------------------------

def make_habitat_config(scene_path: str) -> habitat_sim.Configuration:
    """
    Build Habitat-Sim configuration with:
      - RGB sensor   (for optional VLM use later)
      - Depth sensor (for RL observation)
      - Velocity control enabled
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id      = scene_path
    sim_cfg.enable_physics = True
    sim_cfg.allow_sliding  = True   # prevents hard stops at walls

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height       = 0.88   # RaspBot approximate height (metres)
    agent_cfg.radius       = 0.20   # matches ROBOT_RADIUS in Phase 1

    # --- RGB sensor (kept for future VLM integration) ---
    rgb_spec              = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid         = "color_sensor"
    rgb_spec.sensor_type  = habitat_sim.SensorType.COLOR
    rgb_spec.resolution   = [240, 320]
    rgb_spec.position     = [0.0, SENSOR_HEIGHT, 0.0]
    rgb_spec.orientation  = [0.0, 0.0, 0.0]

    # --- Depth sensor (matches Intel RealSense D435i FOV ~87°x58°) ---
    depth_spec              = habitat_sim.CameraSensorSpec()
    depth_spec.uuid         = "depth_sensor"
    depth_spec.sensor_type  = habitat_sim.SensorType.DEPTH
    depth_spec.resolution   = [DEPTH_H, DEPTH_W]
    depth_spec.position     = [0.0, SENSOR_HEIGHT, 0.0]
    depth_spec.orientation  = [0.0, 0.0, 0.0]
    depth_spec.hfov         = 87.0   # D435i horizontal FOV

    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    # --- Discrete action space (used as fallback reference) ---
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.25),
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward",
            habitat_sim.agent.ActuationSpec(amount=0.25),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=10.0),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=10.0),
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# ---------------------------------------------------------------------------
# HabitatNavEnv
# ---------------------------------------------------------------------------

class HabitatNavEnv(gym.Env):
    """
    Habitat-Sim navigation environment for RaspBot PPO training.

    Drop-in replacement for Phase 1 RobotNavEnv — identical
    observation space, action space, and reward structure.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene_path: str,
        render_mode: str = None,
        random_seed: int = None,
    ):
        super().__init__()

        self.scene_path  = scene_path
        self.render_mode = render_mode
        self.rng         = random.Random(random_seed)
        self.np_rng      = np.random.default_rng(random_seed)

        # Build and start simulator
        cfg        = make_habitat_config(scene_path)
        self._sim  = habitat_sim.Simulator(cfg)
        self._agent = self._sim.initialize_agent(0)

        # Check navmesh
        if self._sim.pathfinder.is_loaded:
            print(f"Navmesh loaded. "
                  f"Navigable area: "
                  f"{self._sim.pathfinder.navigable_area:.1f} m²")
        else:
            print("Warning: navmesh not loaded — "
                  "using fixed spawn position.")

        # --- Observation space (identical to Phase 1) ---
        low = np.array(
            [0.0] * NUM_SECTORS +
            [-MAX_LIN_VEL, -MAX_LIN_VEL, -MAX_ANG_VEL] +
            [0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [MAX_DEPTH] * NUM_SECTORS +
            [MAX_LIN_VEL, MAX_LIN_VEL, MAX_ANG_VEL] +
            [1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        # --- Action space (identical to Phase 1) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Internal state
        self.goal_cmd    = CMD_FORWARD
        self.step_count  = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self._vx         = 0.0
        self._omega      = 0.0

        # For progress reward
        self._prev_pos   = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count  = 0
        self.prev_action = np.zeros(3, dtype=np.float32)
        self._vx         = 0.0
        self._omega      = 0.0

        # Spawn at a random navigable position
        agent_state = habitat_sim.AgentState()
        if self._sim.pathfinder.is_loaded:
            agent_state.position = \
                self._sim.pathfinder.get_random_navigable_point()
        else:
            # Fallback: fixed position near scene centre
            agent_state.position = np.array([0.0, 0.0, 0.0],
                                            dtype=np.float32)

        # Random heading
        angle = self.np_rng.uniform(-math.pi, math.pi)
        agent_state.rotation = _angle_to_quat(angle)
        self._agent.set_state(agent_state)

        # Random goal command
        self.goal_cmd  = self.rng.randint(0, 3)
        self._prev_pos = self._agent.get_state().position.copy()

        obs  = self._get_observation()
        info = {"goal_cmd": CMD_NAMES[self.goal_cmd]}
        return obs, info

    # ------------------------------------------------------------------
    # Step — continuous velocity control
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        vx    = float(action[0]) * MAX_LIN_VEL   # forward velocity
        omega = float(action[2]) * MAX_ANG_VEL   # angular velocity
        # action[1] (vy) is kept for interface compatibility but ignored
        # — Habitat uses non-holonomic (no strafing)

        self._vx    = vx
        self._omega = omega

        # Apply velocity via velocity_control
        vel_ctrl = self._agent.controls
        try:
            # habitat-sim 0.3.x velocity control API
            self._agent.controls.action(
                self._agent.scene_node,
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=abs(vx) * DT),
                apply_filter=True,
            )
            if vx < 0:
                # Move backward by applying move_forward in reverse
                # via a negative translation
                self._agent.controls.action(
                    self._agent.scene_node,
                    "move_forward",
                    habitat_sim.agent.ActuationSpec(
                        amount=abs(vx) * DT * 2
                    ),
                    apply_filter=True,
                )

            # Angular velocity — turn incrementally
            turn_amount_deg = math.degrees(abs(omega) * DT)
            if omega > 0.01:
                self._agent.controls.action(
                    self._agent.scene_node,
                    "turn_left",
                    habitat_sim.agent.ActuationSpec(
                        amount=turn_amount_deg
                    ),
                    apply_filter=True,
                )
            elif omega < -0.01:
                self._agent.controls.action(
                    self._agent.scene_node,
                    "turn_right",
                    habitat_sim.agent.ActuationSpec(
                        amount=turn_amount_deg
                    ),
                    apply_filter=True,
                )

        except Exception:
            # Fallback to discrete action if velocity control fails
            _apply_discrete_fallback(self._sim, vx, omega)

        self.step_count += 1

        # Get new position
        new_pos = self._agent.get_state().position.copy()

        # Collision detection via position change
        # If position barely changed despite vx > 0 → likely collision
        pos_delta = np.linalg.norm(new_pos - self._prev_pos)
        collision = (
            abs(vx) > 0.05 and
            pos_delta < abs(vx) * DT * 0.1
        )

        # Also check depth — if any sector is below safety threshold
        depth_sectors = self._get_depth_sectors()
        if float(np.min(depth_sectors)) < MIN_SAFE_DIST * 0.5:  # 0.2m instead of 0.4m
            collision = True

        # Reward
        reward = self._compute_reward(
            action, collision, new_pos, depth_sectors
        )

        terminated = collision and self.step_count > 5
        truncated  = self.step_count >= MAX_STEPS

        self._prev_pos = new_pos
        self.prev_action = action.copy()

        obs  = self._get_observation()
        info = {
            "goal_cmd"  : CMD_NAMES[self.goal_cmd],
            "collision" : collision,
            "step"      : self.step_count,
            "pos"       : new_pos.tolist(),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        depth_sectors = self._get_depth_sectors()

        velocity = np.array(
            [self._vx, 0.0, self._omega], dtype=np.float32
        )
        velocity = np.clip(
            velocity,
            [-MAX_LIN_VEL, -MAX_LIN_VEL, -MAX_ANG_VEL],
            [MAX_LIN_VEL,  MAX_LIN_VEL,  MAX_ANG_VEL],
        )

        goal_onehot = np.zeros(4, dtype=np.float32)
        goal_onehot[self.goal_cmd] = 1.0

        return np.concatenate(
            [depth_sectors, velocity, goal_onehot]
        ).astype(np.float32)

    def _get_depth_sectors(self) -> np.ndarray:
        """
        Get depth image from Habitat sensor and bin into NUM_SECTORS.
        Each sector takes the minimum depth in its column range.
        0-valued pixels (no return) are replaced with MAX_DEPTH.
        """
        obs          = self._sim.get_sensor_observations()
        depth_image  = obs["depth_sensor"].astype(np.float32)

        # Replace 0 (invalid/no return) with MAX_DEPTH
        depth_image[depth_image == 0] = MAX_DEPTH

        # Clip to sensor range
        depth_image = np.clip(depth_image, 0.0, MAX_DEPTH)

        # Use centre rows only (most relevant for ground-level obstacles)
        mid_row    = DEPTH_H // 2
        row_band   = depth_image[
            max(0, mid_row - 8) : min(DEPTH_H, mid_row + 8), :
        ]

        # Bin columns into NUM_SECTORS
        sectors    = np.full(NUM_SECTORS, MAX_DEPTH, dtype=np.float32)
        cols_per   = DEPTH_W // NUM_SECTORS

        for i in range(NUM_SECTORS):
            col_start = i * cols_per
            col_end   = col_start + cols_per
            patch     = row_band[:, col_start:col_end]
            sectors[i] = float(np.min(patch))

        return sectors

    # ------------------------------------------------------------------
    # Reward — identical logic to Phase 1
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: np.ndarray,
        collision: bool,
        new_pos: np.ndarray,
        depth_sectors: np.ndarray,
    ) -> float:

        reward = R_STEP

        if collision:
            return reward + R_COLLISION

        # Progress in commanded direction
        dx = new_pos[0] - self._prev_pos[0]
        dz = new_pos[2] - self._prev_pos[2]  # Habitat uses Z for forward

        if self.goal_cmd == CMD_FORWARD:
            heading   = _get_agent_heading(self._agent)
            desired_x = math.sin(heading)
            desired_z = -math.cos(heading)
            progress  = dx * desired_x + dz * desired_z
            reward   += R_PROGRESS * progress / (MAX_LIN_VEL * DT + 1e-8)

        elif self.goal_cmd == CMD_LEFT:
            turn_reward = float(action[2]) * 0.5
            reward     += R_PROGRESS * turn_reward

        elif self.goal_cmd == CMD_RIGHT:
            turn_reward = -float(action[2]) * 0.5
            reward     += R_PROGRESS * turn_reward

        elif self.goal_cmd == CMD_STOP:
            speed  = math.hypot(dx, dz) / (DT + 1e-8)
            reward += R_PROGRESS * max(0.0, 1.0 - speed / MAX_LIN_VEL)

        # Smoothness penalty
        jerk   = float(np.linalg.norm(action - self.prev_action))
        reward += R_SMOOTH * jerk

        # Proximity penalty
        min_d  = float(np.min(depth_sectors))
        if min_d < MIN_SAFE_DIST * 2:
            reward += R_SMOOTH * (MIN_SAFE_DIST * 2 - min_d)

        return float(reward)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        obs = self._sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]   # drop alpha channel
        return rgb   # H x W x 3 numpy array

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self._sim.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_to_quat(angle: float):
    """Convert a Y-axis rotation angle (radians) to a Habitat quaternion."""
    import quaternion as qt
    half = angle / 2.0
    return qt.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)


def _get_agent_heading(agent) -> float:
    """Extract Y-axis heading angle (radians) from agent quaternion."""
    rot   = agent.get_state().rotation
    angle, axis = quat_to_angle_axis(rot)
    # axis[1] is Y — positive = counter-clockwise
    return float(angle * np.sign(axis[1]) if abs(axis[1]) > 0.1 else angle)


def _apply_discrete_fallback(sim, vx: float, omega: float):
    """
    Fallback discrete action when velocity control is unavailable.
    Maps continuous (vx, omega) to the nearest discrete action.
    """
    if abs(omega) > abs(vx) * 0.5:
        sim.step("turn_left" if omega > 0 else "turn_right")
    elif vx > 0.05:
        sim.step("move_forward")
    elif vx < -0.05:
        sim.step("move_backward")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Default scene — update this path to match your setup
    DEFAULT_SCENE = (
        "hm3d-example-habitat/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb"
    )

    scene = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENE

    if not os.path.exists(scene):
        print(f"Scene not found: {scene}")
        print("Usage: python habitat_nav_env.py <path/to/scene.glb>")
        print(f"Default path tried: {DEFAULT_SCENE}")
        sys.exit(1)

    print("=" * 55)
    print("  Phase 2 — HabitatNavEnv sanity check")
    print("=" * 55)
    print(f"  Scene: {scene}\n")

    env = HabitatNavEnv(scene_path=scene, random_seed=42)

    obs, info = env.reset()
    print(f"Goal command : {info['goal_cmd']}")
    print(f"Obs shape    : {obs.shape}  (expected: (16,))")
    print(f"Depth sectors: {obs[:9].round(2)}")
    print(f"Velocity     : {obs[9:12].round(3)}")
    print(f"Goal one-hot : {obs[12:16]}")
    print(f"Obs space    : {env.observation_space}")
    print(f"Action space : {env.action_space}\n")

    # 200 random steps
    total_reward = 0.0
    collisions   = 0

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info["collision"]:
            collisions += 1

        if step % 50 == 0:
            print(
                f"Step {step:3d} | reward={reward:+.3f} | "
                f"collision={info['collision']} | "
                f"min_depth={obs[:9].min():.2f}m | "
                f"pos={info['pos']}"
            )

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            obs, info = env.reset()

    print(f"\nTotal reward : {total_reward:.2f}")
    print(f"Collisions   : {collisions}")
    print("\nSanity check complete.")
    env.close()