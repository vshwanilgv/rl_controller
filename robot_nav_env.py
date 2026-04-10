"""
Phase 1 — Step 1.1
RobotNavEnv: Custom Gymnasium environment for RaspBot low-level navigation.

Observation space (16-dim vector):
  [0:9]   — 9-sector depth readings (meters, clipped to MAX_DEPTH)
  [9:12]  — velocity state [vx, vy, omega]
  [12:16] — goal command one-hot [forward, left, right, stop]

Action space (3-dim continuous):
  [vx, vy, omega] each in [-1, 1], scaled to MAX_VEL

Coordinate system:
  x → right,  y → up (in 2D top-down view)
  robot heading angle θ: 0 = facing right, increases counter-clockwise

Requirements:
  pip install gymnasium numpy matplotlib
"""
import matplotlib
matplotlib.use("macosx")
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE       = 10.0          # metres — square world
MAX_DEPTH       = 5.0           # metres — sensor max range
MIN_SAFE_DIST   = 0.4           # metres — safety threshold (matches your diagram)
NUM_SECTORS     = 9             # depth sectors across 180° front arc
SECTOR_FOV      = 180.0         # degrees — total front field of view
MAX_LIN_VEL     = 0.5           # m/s — max linear speed
MAX_ANG_VEL     = 1.0           # rad/s — max angular speed
ROBOT_RADIUS    = 0.2           # metres — robot footprint
DT              = 0.02          # seconds — timestep (50 Hz)
MAX_STEPS       = 1000          # max steps per episode

# Goal commands (matches your VLM dataset labels)
CMD_FORWARD  = 0
CMD_LEFT     = 1
CMD_RIGHT    = 2
CMD_STOP     = 3
CMD_NAMES    = ["forward", "left", "right", "stop"]

# Reward weights
R_PROGRESS      =  1.0    # reward for moving in commanded direction
R_COLLISION     = -10.0   # penalty for hitting obstacle
R_SMOOTH        = -0.05   # penalty for large velocity changes (jerk)
R_GOAL_REACHED  =  20.0   # bonus for completing command successfully
R_STEP          = -0.01   # small time penalty to encourage efficiency


# ---------------------------------------------------------------------------
# Helper — 2D Obstacle (axis-aligned rectangle)
# ---------------------------------------------------------------------------

class Obstacle:
    def __init__(self, x, y, w, h):
        """Rectangle obstacle. (x,y) is centre, w/h are full width/height."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def intersects_circle(self, cx, cy, r):
        """True if a circle (cx, cy, r) overlaps this rectangle."""
        closest_x = max(self.x - self.w / 2, min(cx, self.x + self.w / 2))
        closest_y = max(self.y - self.h / 2, min(cy, self.y + self.h / 2))
        dist = math.hypot(cx - closest_x, cy - closest_y)
        return dist < r

    def raycast(self, ox, oy, angle):
        """
        Cast a ray from (ox, oy) at `angle` radians.
        Returns distance to this obstacle, or MAX_DEPTH if no hit.
        Slab method for AABB ray intersection.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)

        x_min = self.x - self.w / 2
        x_max = self.x + self.w / 2
        y_min = self.y - self.h / 2
        y_max = self.y + self.h / 2

        t_min = 0.0
        t_max = MAX_DEPTH

        for (o_comp, d_comp, b_min, b_max) in [
            (ox, dx, x_min, x_max),
            (oy, dy, y_min, y_max),
        ]:
            if abs(d_comp) < 1e-9:
                if o_comp < b_min or o_comp > b_max:
                    return MAX_DEPTH
            else:
                t1 = (b_min - o_comp) / d_comp
                t2 = (b_max - o_comp) / d_comp
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return MAX_DEPTH

        if t_min < 0:
            return MAX_DEPTH
        return min(t_min, MAX_DEPTH)


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class RobotNavEnv(gym.Env):
    """
    2D top-down navigation environment for RaspBot PPO training.

    The robot must follow a high-level direction command (from the VLM tier)
    while avoiding obstacles using its depth sensor readings.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, num_obstacles=8, random_seed=None):
        super().__init__()

        self.render_mode   = render_mode
        self.num_obstacles = num_obstacles
        self.rng           = random.Random(random_seed)
        self.np_rng        = np.random.default_rng(random_seed)

        # --- Observation space ---
        # [9 depth sectors | 3 velocity | 4 goal one-hot]
        low  = np.zeros(16, dtype=np.float32)
        high = np.array(
            [MAX_DEPTH] * NUM_SECTORS +       # depth sectors
            [MAX_LIN_VEL, MAX_LIN_VEL, MAX_ANG_VEL] +  # velocity
            [1.0, 1.0, 1.0, 1.0],            # goal one-hot
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # --- Action space ---
        # [vx, vy, omega] normalised to [-1, 1], scaled in step()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Internal state (initialised in reset)
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_theta = 0.0   # heading in radians
        self.robot_vx    = 0.0
        self.robot_vy    = 0.0
        self.robot_omega = 0.0
        self.goal_cmd    = CMD_FORWARD
        self.obstacles   : list[Obstacle] = []
        self.step_count  = 0
        self.prev_action = np.zeros(3, dtype=np.float32)

        # For progress reward — track displacement along commanded direction
        self.prev_x = 0.0
        self.prev_y = 0.0

        # Matplotlib figure (lazy init)
        self._fig = None
        self._ax  = None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        self.step_count  = 0
        self.prev_action = np.zeros(3, dtype=np.float32)

        # Spawn robot near centre with random heading
        self.robot_x     = self.np_rng.uniform(3.0, 7.0)
        self.robot_y     = self.np_rng.uniform(3.0, 7.0)
        self.robot_theta = self.np_rng.uniform(-math.pi, math.pi)
        self.robot_vx    = 0.0
        self.robot_vy    = 0.0
        self.robot_omega = 0.0

        # Random goal command (forward / left / right / stop)
        self.goal_cmd = self.rng.randint(0, 3)

        # Generate random obstacles — ensure no spawn collision
        self.obstacles = self._generate_obstacles()

        self.prev_x = self.robot_x
        self.prev_y = self.robot_y

        obs = self._get_observation()
        info = {"goal_cmd": CMD_NAMES[self.goal_cmd]}
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Scale actions to physical velocities
        vx    = float(action[0]) * MAX_LIN_VEL
        vy    = float(action[1]) * MAX_LIN_VEL
        omega = float(action[2]) * MAX_ANG_VEL

        # --- Kinematics update (simple unicycle in world frame) ---
        # Rotate local velocities to world frame using current heading
        cos_t = math.cos(self.robot_theta)
        sin_t = math.sin(self.robot_theta)
        world_vx = vx * cos_t - vy * sin_t
        world_vy = vx * sin_t + vy * cos_t

        new_x     = self.robot_x     + world_vx * DT
        new_y     = self.robot_y     + world_vy * DT
        new_theta = self.robot_theta + omega * DT

        # Clamp to world bounds
        new_x = np.clip(new_x, ROBOT_RADIUS, GRID_SIZE - ROBOT_RADIUS)
        new_y = np.clip(new_y, ROBOT_RADIUS, GRID_SIZE - ROBOT_RADIUS)

        # Wrap heading to [-pi, pi]
        new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi

        # Check collision
        collision = self._check_collision(new_x, new_y)

        if not collision:
            self.robot_x     = new_x
            self.robot_y     = new_y
            self.robot_theta = new_theta

        self.robot_vx    = vx
        self.robot_vy    = vy
        self.robot_omega = omega
        self.step_count += 1

        # --- Reward calculation ---
        reward = self._compute_reward(action, collision)

        # --- Termination ---
        terminated = collision
        truncated  = self.step_count >= MAX_STEPS

        self.prev_x      = self.robot_x
        self.prev_y      = self.robot_y
        self.prev_action = action.copy()

        obs  = self._get_observation()
        info = {
            "goal_cmd"  : CMD_NAMES[self.goal_cmd],
            "collision" : collision,
            "step"      : self.step_count,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        depth_sectors = self._get_depth_sectors()

        velocity = np.array(
            [self.robot_vx, self.robot_vy, self.robot_omega], dtype=np.float32
        )

        goal_onehot = np.zeros(4, dtype=np.float32)
        goal_onehot[self.goal_cmd] = 1.0

        obs = np.concatenate([depth_sectors, velocity, goal_onehot])
        return obs.astype(np.float32)

    def _get_depth_sectors(self) -> np.ndarray:
        """
        Cast NUM_SECTORS rays across a SECTOR_FOV-degree front arc.
        Each ray returns the minimum distance to any obstacle or world wall.
        Returns a numpy array of shape (NUM_SECTORS,).
        """
        sectors = np.full(NUM_SECTORS, MAX_DEPTH, dtype=np.float32)

        half_fov   = math.radians(SECTOR_FOV / 2)
        sector_step = math.radians(SECTOR_FOV) / (NUM_SECTORS - 1)

        for i in range(NUM_SECTORS):
            # Angle relative to robot heading
            rel_angle = -half_fov + i * sector_step
            world_angle = self.robot_theta + rel_angle

            # Minimum distance across all obstacles
            min_dist = MAX_DEPTH

            # Check each obstacle
            for obs in self.obstacles:
                d = obs.raycast(self.robot_x, self.robot_y, world_angle)
                if d < min_dist:
                    min_dist = d

            # Check world boundary walls
            wall_dist = self._wall_raycast(world_angle)
            if wall_dist < min_dist:
                min_dist = wall_dist

            sectors[i] = min_dist

        return sectors

    def _wall_raycast(self, angle: float) -> float:
        """Distance to the nearest world boundary wall along `angle`."""
        dx = math.cos(angle)
        dy = math.sin(angle)
        distances = []

        if abs(dx) > 1e-9:
            t = (GRID_SIZE - self.robot_x) / dx if dx > 0 else -self.robot_x / dx
            if t > 0:
                distances.append(t)
        if abs(dy) > 1e-9:
            t = (GRID_SIZE - self.robot_y) / dy if dy > 0 else -self.robot_y / dy
            if t > 0:
                distances.append(t)

        return min(distances) if distances else MAX_DEPTH

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, action: np.ndarray, collision: bool) -> float:
        reward = R_STEP  # small step penalty

        if collision:
            return reward + R_COLLISION

        # Progress reward: how much did we move in the commanded direction?
        dx = self.robot_x - self.prev_x
        dy = self.robot_y - self.prev_y

        # Desired movement direction based on goal command and current heading
        if self.goal_cmd == CMD_FORWARD:
            desired_dx = math.cos(self.robot_theta)
            desired_dy = math.sin(self.robot_theta)
        elif self.goal_cmd == CMD_LEFT:
            # Turn left → positive omega is good
            turn_reward = float(action[2]) * 0.5   # omega component
            reward += R_PROGRESS * turn_reward
            desired_dx, desired_dy = 0.0, 0.0
        elif self.goal_cmd == CMD_RIGHT:
            turn_reward = -float(action[2]) * 0.5  # negative omega = right turn
            reward += R_PROGRESS * turn_reward
            desired_dx, desired_dy = 0.0, 0.0
        else:  # CMD_STOP
            speed = math.hypot(dx, dy) / DT
            reward += R_PROGRESS * max(0, 1.0 - speed / MAX_LIN_VEL)
            desired_dx, desired_dy = 0.0, 0.0

        if self.goal_cmd == CMD_FORWARD:
            progress = dx * desired_dx + dy * desired_dy  # dot product
            reward += R_PROGRESS * progress / (MAX_LIN_VEL * DT + 1e-8)

        # Smoothness penalty: penalise large action changes (jerk)
        jerk = np.linalg.norm(action - self.prev_action)
        reward += R_SMOOTH * jerk

        # Safety bonus: reward keeping distance from obstacles
        depth_sectors = self._get_depth_sectors()
        min_depth = float(np.min(depth_sectors))
        if min_depth < MIN_SAFE_DIST * 2:
            # Approaching danger zone — negative reward proportional to closeness
            reward += R_SMOOTH * (MIN_SAFE_DIST * 2 - min_depth)

        return float(reward)

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_collision(self, x: float, y: float) -> bool:
        # World boundary
        if x < ROBOT_RADIUS or x > GRID_SIZE - ROBOT_RADIUS:
            return True
        if y < ROBOT_RADIUS or y > GRID_SIZE - ROBOT_RADIUS:
            return True
        # Obstacles
        for obs in self.obstacles:
            if obs.intersects_circle(x, y, ROBOT_RADIUS):
                return True
        return False

    # ------------------------------------------------------------------
    # Obstacle generation
    # ------------------------------------------------------------------

    def _generate_obstacles(self) -> list:
        obstacles = []
        attempts  = 0
        while len(obstacles) < self.num_obstacles and attempts < 200:
            attempts += 1
            w = self.np_rng.uniform(0.4, 1.2)
            h = self.np_rng.uniform(0.4, 1.2)
            x = self.np_rng.uniform(w / 2 + 0.5, GRID_SIZE - w / 2 - 0.5)
            y = self.np_rng.uniform(h / 2 + 0.5, GRID_SIZE - h / 2 - 0.5)
            candidate = Obstacle(x, y, w, h)
            # Don't spawn on top of the robot (keep 1.5m clear zone)
            if math.hypot(x - self.robot_x, y - self.robot_y) < 1.5:
                continue
            # Don't overlap existing obstacles
            overlap = any(
                abs(x - o.x) < (w + o.w) / 2 + 0.3 and
                abs(y - o.y) < (h + o.h) / 2 + 0.3
                for o in obstacles
            )
            if not overlap:
                obstacles.append(candidate)
        return obstacles

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return None

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1, figsize=(7, 7))

        ax = self._ax
        ax.clear()
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect("equal")
        ax.set_title(
            f"Step {self.step_count} | Goal: {CMD_NAMES[self.goal_cmd].upper()} | "
            f"vx={self.robot_vx:.2f} vy={self.robot_vy:.2f} ω={self.robot_omega:.2f}",
            fontsize=10,
        )
        ax.set_facecolor("#f5f5f5")

        # Draw obstacles
        for obs in self.obstacles:
            rect = patches.Rectangle(
                (obs.x - obs.w / 2, obs.y - obs.h / 2),
                obs.w, obs.h,
                linewidth=1, edgecolor="#555", facecolor="#aaa",
            )
            ax.add_patch(rect)

        # Draw depth rays
        half_fov    = math.radians(SECTOR_FOV / 2)
        sector_step = math.radians(SECTOR_FOV) / (NUM_SECTORS - 1)
        depth_sectors = self._get_depth_sectors()

        for i in range(NUM_SECTORS):
            rel_angle   = -half_fov + i * sector_step
            world_angle = self.robot_theta + rel_angle
            d = depth_sectors[i]
            ex = self.robot_x + d * math.cos(world_angle)
            ey = self.robot_y + d * math.sin(world_angle)
            color = "#e74c3c" if d < MIN_SAFE_DIST else "#3498db"
            ax.plot(
                [self.robot_x, ex], [self.robot_y, ey],
                color=color, alpha=0.4, linewidth=1,
            )

        # Draw robot body
        robot_circle = plt.Circle(
            (self.robot_x, self.robot_y), ROBOT_RADIUS,
            color="#2ecc71", zorder=5,
        )
        ax.add_patch(robot_circle)

        # Draw heading arrow
        hx = self.robot_x + ROBOT_RADIUS * 1.8 * math.cos(self.robot_theta)
        hy = self.robot_y + ROBOT_RADIUS * 1.8 * math.sin(self.robot_theta)
        ax.annotate(
            "", xy=(hx, hy), xytext=(self.robot_x, self.robot_y),
            arrowprops=dict(arrowstyle="->", color="#1a1a1a", lw=2),
        )

        # Draw depth sector bar at bottom
        bar_y = 0.3
        bar_w = GRID_SIZE / NUM_SECTORS * 0.85
        for i, d in enumerate(depth_sectors):
            bar_x    = (i + 0.5) * GRID_SIZE / NUM_SECTORS
            norm_d   = d / MAX_DEPTH
            bar_color = "#e74c3c" if d < MIN_SAFE_DIST else "#3498db"
            bar = patches.Rectangle(
                (bar_x - bar_w / 2, 0.05), bar_w, 0.25 * norm_d * 0.8,
                facecolor=bar_color, alpha=0.7,
            )
            ax.add_patch(bar)
            ax.text(
                bar_x, 0.08, f"S{i}", ha="center", va="bottom",
                fontsize=6, color="#333",
            )

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

        if self.render_mode == "rgb_array":
            import numpy as np
            self._fig.canvas.draw()
            buf = self._fig.canvas.tostring_rgb()
            w, h = self._fig.canvas.get_width_height()
            return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to test the environment
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== RobotNavEnv sanity check ===\n")

    env = RobotNavEnv(render_mode="human", num_obstacles=8, random_seed=42)

    obs, info = env.reset()
    print(f"Goal command : {info['goal_cmd']}")
    print(f"Obs shape    : {obs.shape}  (expected: (16,))")
    print(f"Depth sectors: {obs[:9].round(2)}")
    print(f"Velocity     : {obs[9:12].round(3)}")
    print(f"Goal one-hot : {obs[12:16]}")
    print(f"\nAction space : {env.action_space}")
    print(f"Obs space    : {env.observation_space}\n")

    # Run 200 random steps
    total_reward = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 50 == 0:
            print(
                f"Step {step:3d} | reward={reward:+.3f} | "
                f"collision={info['collision']} | "
                f"min_depth={obs[:9].min():.2f}m"
            )
        if terminated or truncated:
            print(f"\nEpisode ended at step {step} — collision={info['collision']}")
            obs, info = env.reset()

        time.sleep(0.01)

    print(f"\nTotal reward over 200 steps: {total_reward:.2f}")
    print("\nSanity check complete.")
    env.close()