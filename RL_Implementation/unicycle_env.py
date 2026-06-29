import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from gymnasium import spaces
from math import pi
from typing import Optional, Tuple, Dict, Any
from matplotlib.patches import Rectangle, Circle
from plot import plot_polygons

import time

class UnicycleGymEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self,
                 state_space: Tuple[float, float] = (10, 10),
                 initial_state: Tuple[float, float, float] = (2.0, 2.0),
                 goal_set: np.ndarray = [[np.array([0, 6]), 4, 4]],
                 goal_reward: float = 1.0,
                 crash_cost: float = 0.0,
                 stage_cost: np.ndarray = np.array([[-1, 0],[0, 0]]),
                 input_constraints: np.ndarray = np.array([[0, 3], [-pi, pi]]),
                 max_steps: int = 100,
                 random_init: float = 0.0,
                 dist_mean: Tuple[float, float] = (0.0, 0.0),
                 dist_var: Tuple[float, float] = (0.4, 0.4),
                 input_mean: float = 0.0,
                 input_var: float = 0.2,
                 render_mode: Optional[str] = None,
                 seed: int = 2701
                 ):
        
        super().__init__()

        # Input and Action spaces

        xmax, ymax = state_space

        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([xmax, ymax], dtype=np.float32)

        self.observation_space = spaces.Box(low = low, high = high, dtype=np.float32)

        act_low = np.array([input_constraints[0,0], input_constraints[1, 0]], dtype=np.float32)
        act_high = np.array([input_constraints[0,1], input_constraints[1,1]], dtype=np.float32)

        self.action_space = spaces.Box(low = act_low, high = act_high, dtype=np.float32)

        # Other stuff

        self.render_mode = render_mode
        self._render_fig = None
        self._render_ax = None

        self.seed = seed

        self.max_steps = max_steps

        # Store trajectory of episode
        # if self.render_mode is not None:
        self.trajectory = []

        # Simulation

        self.initial_state = initial_state
        self.state_space = np.array(state_space)
        self.state = initial_state
        self.goal_set = goal_set
        self.goal_reward = goal_reward
        self.crash_cost = crash_cost
        self.stage_cost = stage_cost
        self.random_init = random_init
        self.dist_mean = np.array(dist_mean)
        self.dist_std = np.sqrt(np.array(dist_var))
        self.input_mean = input_mean
        self.input_std = np.sqrt(input_var)

        self.input_constraints = input_constraints

        self.step_count = 0

        self.rng = np.random.default_rng(self.seed)

        # Lists of unsafe sets, empty at initialisation
        self.unsafe_rect = []
        self.unsafe_circ = []
        self.unsafe_polygons = []

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        self.step_count += 1

        action = np.clip(action, self.input_constraints[:,0], self.input_constraints[:,1])

        if np.any(self.dist_std != 0):
            disturbance = self.rng.normal(self.dist_mean, self.dist_std)
        else:
            disturbance = self.dist_mean

        if self.input_std != 0:
            input_disturbance = self.rng.normal(self.input_mean, self.input_std)
        else:
            input_disturbance = self.input_mean

        next_state = self.state + np.array([action[0] * np.cos(action[1] + input_disturbance), action[0] * np.sin(action[1] + input_disturbance)]) + disturbance

        success = False
        crash = False
        truncated = False

        # Check time out
        if self.step_count >= self.max_steps:
            truncated = True
            reward = 0

        # Crash if next state unsafe
        elif self.is_unsafe(next_state, self.state):
            crash = True
            reward = self.crash_cost - abs(action[0])*self.stage_cost

        # Check success
        # elif np.all(next_state - self.goal_set[0] <= np.array([self.goal_set[1], self.goal_set[2]])) and np.all(next_state - self.goal_set[0] >= np.zeros(2)):
        #     success = True
        #     reward = self.goal_reward - abs(action[0])*self.stage_cost
        for goal in self.goal_set:
            lower_left = goal[0]
            size = np.array([goal[1], goal[2]])
            
            if np.all(next_state - lower_left >= 0) and np.all(next_state - lower_left <= size):
                success = True
                reward = self.goal_reward - abs(action[0]) * self.stage_cost
                break  # stop at first goal reached

        # Continue and stage cost if next state okay and not timed out
        else:
            reward = -abs(action[0])*self.stage_cost

        self.state = next_state
        info = {"success": success, "crash": crash, "step count": self.step_count}

        reward = float(reward)

        return next_state, reward, success or crash, truncated, info
    

    def is_unsafe(self, next_state, state):
        p = state
        q = next_state

        # --- helper functions ---
        def inside_rect(point, rect):
            return (
                rect[0] <= point[0] <= rect[0] + rect[2] and
                rect[1] <= point[1] <= rect[1] + rect[3]
            )

        # orientation test
        def orient(a, b, c):
            return np.cross(b - a, c - a)

        # segment–segment intersection
        def seg_intersect(a, b, c, d):
            o1 = orient(a, b, c)
            o2 = orient(a, b, d)
            o3 = orient(c, d, a)
            o4 = orient(c, d, b)

            # proper intersection
            if (o1 * o2 < 0) and (o3 * o4 < 0):
                return True

            # collinear or touching cases
            def on_seg(a, b, p):
                return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                        min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

            eps = 1e-12
            if abs(o1) < eps and on_seg(a, b, c): return True
            if abs(o2) < eps and on_seg(a, b, d): return True
            if abs(o3) < eps and on_seg(c, d, a): return True
            if abs(o4) < eps and on_seg(c, d, b): return True

            return False

        # --- rectangles ---
        for rect in self.unsafe_rect:
            x, y, w, h = rect

            # 4 corners
            c1 = np.array([x,     y    ])
            c2 = np.array([x+w,   y    ])
            c3 = np.array([x+w,   y+h  ])
            c4 = np.array([x,     y+h  ])

            # 4 edges as segments
            edges = [
                (c1, c2),
                (c2, c3),
                (c3, c4),
                (c4, c1)
            ]

            # 1. endpoint inside rectangle
            if inside_rect(p, rect) or inside_rect(q, rect):
                return True

            # 2. segment intersects any rectangle edge
            for a, b in edges:
                if seg_intersect(p, q, a, b):
                    return True
                
        # --- circles ---
        def segment_circle_collision(p0, p1, center, radius):
            d = p1 - p0          # segment direction
            f = p0 - center      # from center to start

            # Project center onto segment
            t = -np.dot(f, d) / np.dot(d, d)

            # Clamp to segment
            t = np.clip(t, 0.0, 1.0)

            closest = p0 + t * d

            # Distance to center
            return np.linalg.norm(closest - center) <= radius

        for circle in self.unsafe_circ:
            if np.linalg.norm(circle[0:2] - next_state) <= circle[2] + 0.0001:
                self.is_crashed = True
                return True
            
            if segment_circle_collision(state, next_state, circle[0:2], circle[2]):
                self.is_crashed = True
                return True
            
        # --- Convex Polygons ---    
        def cross(a, b):
            return a[0]*b[1] - a[1]*b[0]

        def polygon_intersect(p1, p2, q1, q2):
            r = p2 - p1
            s = q2 - q1

            denom = cross(r, s)
            if denom == 0:
                return False  # parallel

            t = cross(q1 - p1, s) / denom
            u = cross(q1 - p1, r) / denom

            return (0 <= t <= 1) and (0 <= u <= 1)


        def point_in_convex_polygon(point, polygon):
            # polygon: (N,2), ordered (CW or CCW)
            sign = None

            for i in range(len(polygon)):
                a = polygon[i]
                b = polygon[(i+1) % len(polygon)]

                edge = b - a
                to_point = point - a

                c = cross(edge, to_point)

                if c != 0:
                    if sign is None:
                        sign = np.sign(c)
                    elif np.sign(c) != sign:
                        return False

            return True


        def segment_polygon_collision(p0, p1, polygon):
            # 1. endpoint inside polygon
            if point_in_convex_polygon(p0, polygon) or point_in_convex_polygon(p1, polygon):
                return True

            # 2. segment intersects any edge
            for i in range(len(polygon)):
                q0 = polygon[i]
                q1 = polygon[(i+1) % len(polygon)]

                if polygon_intersect(p0, p1, q0, q1):
                    return True

            return False
        
        for poly in self.unsafe_polygons:
            if segment_polygon_collision(state, next_state, poly):
                self.is_crashed = True
                return True
        
        # --- boundary violation ---
        if np.any(next_state != np.clip(next_state, [0, 0], self.state_space)):
            return True

        return False
    
    def reset(self):
        self.step_count = 0
        random_cost = 0

        random_init = self.rng.choice([True, False], p=[self.random_init, 1 - self.random_init])

        if random_init:
            unsafe = True
            in_target = True            

            while(unsafe or in_target):
                for goal in self.goal_set:
                    self.state = np.multiply(self.rng.uniform(size=2), self.state_space)
                    unsafe = self.is_unsafe(self.state, self.state+0.001) or np.all(self.state - goal[0] <= np.array([goal[1], goal[2]])) and np.all(self.state - goal[0] >= np.zeros(2))
                    in_target = goal[0][0] < self.state[0] < goal[0][0] + goal[1] and goal[0][1] < self.state[1] < goal[2]

            random_cost = self.rng.random()

        else:
            self.state = self.initial_state

        # if self.render_mode is not None:
        self.trajectory = [self.state[:2]]

        info = {"random cost": random_cost}

        return self.state, info

    # Unused
    def render(self, render = False):

        if render != self.render_mode:
            mode = render
        else:
            mode = self.render_mode

        if mode is None:
            return
        
        if self._render_fig is None or self._render_ax is None:
            self._render_fig, self._render_ax = plt.subplots()

        ax = self._render_ax
        ax.clear()

        # Plot obstacles
        for rect in self.unsafe_rect:
            ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3], alpha = 0.5))
        for circ in self.unsafe_circ:
            ax.add_patch(Circle((circ[0], circ[1]), circ[2], alpha = 0.5))

        # Plot trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], "-b")

        # Plot robot
        x, y = self.state
        ax.plot(x, y, marker = "o")

        # Plot target area
        # Plot target
        g = self.goal_set
        ax.add_patch(Rectangle((g[0][0], g[0][1]), g[1], g[2], fill = False, linestyle = "--"))

        ax.set_xlim(0, max(10, float(self.state_space[0]) if np.isfinite(self.state_space[0]) else 10))
        ax.set_ylim(0, max(10, float(self.state_space[1]) if np.isfinite(self.state_space[1]) else 10))
        ax.set_aspect('equal')
        ax.set_title(f"Step {self.step_count}")

        if mode == "human":
            plt.pause(0.001)
            plt.draw()
            return None
        elif mode == "rgb_array":
            # Return RGB array of the figure canvas
            self._render_fig.canvas.draw()
            import numpy as _np
            w, h = self._render_fig.canvas.get_width_height()
            buf = _np.frombuffer(self._render_fig.canvas.tostring_rgb(), dtype=_np.uint8)
            buf = buf.reshape((h, w, 3))
            return buf
        
    # Append rectangular obstacle to list of rectangular obstacles, form (ndarray w. shape (2,), float, float)
    def add_rect_set(self, btml_corner, width, height):
        assert btml_corner.shape == (2,)
        self.unsafe_rect.append(np.append(btml_corner, [width, height]))

    # Append circular obstacle to list of circular obstacles, form (ndarray w. shape (2,), float)
    def add_circ_set(self, center, radius):
        assert center.shape == (2,)
        self.unsafe_circ.append(np.append(center, radius))

    def add_poly_set(self, corners):
        self.unsafe_polygons.append(corners)

    def sample_action_space(self):
        low = self.input_constraints[:,0]
        high = self.input_constraints[:,1]
        action = self.rng.uniform(low, high)
        return action