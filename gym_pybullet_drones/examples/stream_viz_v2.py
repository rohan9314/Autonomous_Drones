"""stream_viz_v2.py — real-time Vispy visualiser for ObstacleAviaryV2.

Headless PyBullet physics + Vispy 3-D window. Handles:
  - Variable-size box/cylinder obstacles (queries PyBullet for actual geometry)
  - VecNormalize observation wrapper (pass the wrapped venv, not the raw env)
  - Lidar ray overlay (optional, toggled with L key)

Install deps
------------
    pip install "vispy[pyqt5]" pyopengl
    # or: pip install "vispy[pyside6]" pyopengl

Standalone usage
----------------
    python stream_viz_v2.py                            # auto-find latest v2 model
    python stream_viz_v2.py --model_path <zip>
    python stream_viz_v2.py --difficulty 2 --seed 3
    python stream_viz_v2.py --difficulty 3 --episodes 3   # loop N episodes

Integration
-----------
    from stream_viz_v2 import DroneVizV2
    viz = DroneVizV2(inner_env, venv, title="demo")
    viz.run(model=model, seed=0)
"""

import argparse
import glob
import os

import numpy as np

try:
    from vispy import app, scene
    from vispy.scene import visuals
except ImportError as exc:
    raise SystemExit(
        "Vispy not found. Install with:\n"
        '  pip install "vispy[pyqt5]" pyopengl'
    ) from exc

import pybullet as p

from gym_pybullet_drones.envs.ObstacleAviaryV2 import (
    ObstacleAviaryV2,
    DIFFICULTY_OBSTACLE_COUNTS,
    N_LIDAR_HORIZ, N_LIDAR_VERT, N_LIDAR_TOTAL, LIDAR_RANGE,
)
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

TRAIL_LEN  = 300
ROTOR_SEGS = 28


# ── geometry helpers ───────────────────────────────────────────────────────────

def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p_, y = rpy
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p_), np.sin(p_)
    cr, sr = np.cos(r), np.sin(r)
    return (np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]) @
            np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]) @
            np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]))


def _circle_pts(center: np.ndarray, radius: float, n: int = ROTOR_SEGS) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.zeros((n + 1, 3))
    pts[:, 0] = center[0] + radius * np.cos(t)
    pts[:, 1] = center[1] + radius * np.sin(t)
    pts[:, 2] = center[2]
    return pts


def _box_edge_pts(cx, cy, cz, hx, hy, hz) -> np.ndarray:
    corners = np.array([
        [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
        [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
        [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
        [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts = np.empty((len(edges) * 2, 3))
    for i, (a, b) in enumerate(edges):
        pts[2*i], pts[2*i+1] = corners[a], corners[b]
    return pts


def _ground_grid(extent: int = 3, step: float = 0.5) -> np.ndarray:
    lines = []
    vals  = np.arange(-extent, extent + step, step)
    for v in vals:
        lines += [[-extent, v, 0], [extent, v, 0]]
        lines += [[v, -extent, 0], [v, extent, 0]]
    return np.array(lines)


def _obstacle_edge_pts(oid: int, client_id: int) -> tuple:
    """Return (edge_pts array, is_cylinder) for one obstacle body."""
    pos_w, _ = p.getBasePositionAndOrientation(oid, physicsClientId=client_id)
    data = p.getCollisionShapeData(oid, -1, physicsClientId=client_id)
    if not data:
        he = 0.15
        return _box_edge_pts(*pos_w, he, he, he), False

    shape_type, dims = data[0][2], data[0][3]
    ox, oy, oz = pos_w

    if shape_type == p.GEOM_BOX:
        hx, hy, hz = dims[0], dims[1], dims[2]
        return _box_edge_pts(ox, oy, oz, hx, hy, hz), False
    elif shape_type == p.GEOM_CYLINDER:
        r, h = dims[0], dims[1]  # pybullet: dims=(radius, height, ...)
        return _box_edge_pts(ox, oy, oz, r, r, h * 0.5), True

    he = 0.15
    return _box_edge_pts(ox, oy, oz, he, he, he), False


# ── lidar ray geometry ─────────────────────────────────────────────────────────

def _lidar_ray_pts(drone_pos: np.ndarray, lidar_vals: np.ndarray) -> np.ndarray:
    """(N*2, 3) segment array: drone_pos → hit point for each lidar ray."""
    h_angles = np.linspace(0.0, 2.0 * np.pi, N_LIDAR_HORIZ, endpoint=False)
    h_dirs   = np.column_stack([np.cos(h_angles), np.sin(h_angles), np.zeros(N_LIDAR_HORIZ)])

    v_angles = np.linspace(-np.pi / 4.0, np.pi / 4.0, N_LIDAR_VERT)
    v_dirs   = np.column_stack([np.cos(v_angles), np.zeros(N_LIDAR_VERT), np.sin(v_angles)])

    all_dirs = np.vstack([h_dirs, v_dirs])  # (48, 3)

    pts = np.empty((N_LIDAR_TOTAL * 2, 3))
    for i, (d, frac) in enumerate(zip(all_dirs, lidar_vals)):
        pts[2*i]     = drone_pos
        pts[2*i + 1] = drone_pos + d * LIDAR_RANGE * frac
    return pts


# ── main visualiser class ──────────────────────────────────────────────────────

class DroneVizV2:
    """Real-time Vispy visualiser for ObstacleAviaryV2.

    Parameters
    ----------
    inner_env : ObstacleAviaryV2
        The unwrapped environment (needed for state access).
    venv : DummyVecEnv | VecNormalize
        The wrapped env used for stepping (handles obs normalization).
    title : str
    show_lidar : bool
        Start with lidar ray overlay enabled (toggle with L key).
    """

    def __init__(self, inner_env: ObstacleAviaryV2, venv,
                 title: str = "V2 Drone Viz", show_lidar: bool = True):
        self.inner_env   = inner_env
        self.venv        = venv
        self._trail_buf  = []
        self._step       = 0
        self._ep_reward  = 0.0
        self._done       = False
        self._obs        = None
        self._action_fn  = None
        self._timer      = None
        self._show_lidar = show_lidar

        # ── canvas ────────────────────────────────────────────────────────────
        self.canvas = scene.SceneCanvas(
            title=title, size=(1280, 720), bgcolor="#0e0e1a",
            keys="interactive", show=True
        )
        self.canvas.events.key_press.connect(self._on_key)
        vb = self.canvas.central_widget.add_view()
        vb.camera = scene.TurntableCamera(elevation=22, azimuth=-50, distance=5.5, fov=50)
        self._vb = vb

        # ── static scene elements ─────────────────────────────────────────────
        visuals.Line(pos=_ground_grid(), color=(0.22, 0.22, 0.32, 0.55),
                     connect="segments", parent=vb.scene)
        visuals.XYZAxis(parent=vb.scene)

        # ── drone body ────────────────────────────────────────────────────────
        self._body_mkr = visuals.Markers(parent=vb.scene)
        self._body_mkr.set_data(np.zeros((1, 3)),
                                face_color=(0.15, 0.65, 1.0, 1.0),
                                edge_color=(1.0, 1.0, 1.0, 0.9), size=14, edge_width=1.5)

        L = float(getattr(inner_env, "L", 0.0397))
        s = L / np.sqrt(2)
        self._rotor_body = np.array([[ s,-s,0], [-s,-s,0], [-s, s,0], [ s, s,0]])
        self._rotor_r    = L * 0.75

        self._arm_lines = [
            visuals.Line(pos=np.zeros((2, 3)), color=(1.0, 0.45, 0.15, 1.0),
                         width=3.0, connect="segments", parent=vb.scene)
            for _ in range(4)
        ]
        self._rotor_rings = [
            visuals.Line(pos=_circle_pts(np.zeros(3), 0.04),
                         color=(0.9, 0.85, 0.1, 0.85), width=1.8, connect="strip",
                         parent=vb.scene)
            for _ in range(4)
        ]

        # ── trail ─────────────────────────────────────────────────────────────
        self._trail_line = visuals.Line(pos=np.zeros((2, 3)), color=np.zeros((2, 4)),
                                        width=2.0, connect="strip", antialias=True,
                                        parent=vb.scene)

        # ── target ────────────────────────────────────────────────────────────
        tgt = getattr(inner_env, "TARGET_POS", np.array([1.0, 0.0, 1.0]))
        self._target_mkr = visuals.Markers(parent=vb.scene)
        self._target_mkr.set_data(tgt.reshape(1, 3),
                                  face_color=(0.1, 0.95, 0.35, 1.0),
                                  edge_color=(1.0, 1.0, 1.0, 0.9),
                                  size=20, edge_width=2.0, symbol="star")

        # ── obstacle wireframes (rebuilt per episode) ─────────────────────────
        self._obs_visuals = []

        # ── lidar ray overlay ─────────────────────────────────────────────────
        self._lidar_lines = visuals.Line(
            pos=np.zeros((N_LIDAR_TOTAL * 2, 3)),
            color=(0.3, 0.95, 0.3, 0.35),
            connect="segments", width=1.0, parent=vb.scene
        )

        # ── HUD ───────────────────────────────────────────────────────────────
        self._hud = visuals.Text("", color=(1.0, 1.0, 1.0, 0.9),
                                 font_size=11, anchor_x="left", anchor_y="top",
                                 parent=self.canvas.scene)
        self._hud.pos = (16, 16)

    # ── keyboard toggle ───────────────────────────────────────────────────────

    def _on_key(self, event):
        if event.key.name.upper() == "L":
            self._show_lidar = not self._show_lidar
            self._lidar_lines.visible = self._show_lidar

    # ── per-episode scene setup ───────────────────────────────────────────────

    def _rebuild_obstacles(self):
        for v in self._obs_visuals:
            v.parent = None
        self._obs_visuals.clear()

        env = self.inner_env
        for oid in env.obstacle_ids:
            pts, is_cyl = _obstacle_edge_pts(oid, env.CLIENT)
            color = (0.2, 0.5, 1.0, 0.75) if is_cyl else (0.9, 0.2, 0.2, 0.75)
            ln = visuals.Line(pos=pts, color=color, width=2.0,
                              connect="segments", parent=self._vb.scene)
            self._obs_visuals.append(ln)

        tgt = getattr(env, "TARGET_POS", np.array([1.0, 0.0, 1.0]))
        self._target_mkr.set_data(tgt.reshape(1, 3),
                                  face_color=(0.1, 0.95, 0.35, 1.0),
                                  edge_color=(1.0, 1.0, 1.0, 0.9),
                                  size=20, edge_width=2.0, symbol="star")

    # ── per-step refresh ──────────────────────────────────────────────────────

    def _refresh_drone(self, pos: np.ndarray, rpy: np.ndarray):
        R = _rpy_matrix(rpy)
        self._body_mkr.set_data(pos.reshape(1, 3),
                                face_color=(0.15, 0.65, 1.0, 1.0),
                                edge_color=(1.0, 1.0, 1.0, 0.9), size=14, edge_width=1.5)
        for i, off_b in enumerate(self._rotor_body):
            hub = pos + R @ off_b
            self._arm_lines[i].set_data(pos=np.array([pos, hub]),
                                        color=(1.0, 0.45, 0.15, 1.0))
            self._rotor_rings[i].set_data(pos=_circle_pts(hub, self._rotor_r),
                                          color=(0.9, 0.85, 0.1, 0.85))

    def _refresh_trail(self):
        n = len(self._trail_buf)
        if n < 2:
            return
        pts    = np.array(self._trail_buf)
        alpha  = np.linspace(0.05, 0.9, n)
        colors = np.zeros((n, 4))
        colors[:, 0] = 0.15; colors[:, 1] = 0.55; colors[:, 2] = 1.0
        colors[:, 3] = alpha
        self._trail_line.set_data(pos=pts, color=colors, connect="strip")

    def _refresh_lidar(self, pos: np.ndarray):
        if not self._show_lidar:
            return
        env   = self.inner_env
        lidar = env._last_lidar if hasattr(env, "_last_lidar") else np.ones(N_LIDAR_TOTAL)
        pts   = _lidar_ray_pts(pos, lidar)
        # colour by proximity: close hits → red, clear → green
        colors = np.zeros((N_LIDAR_TOTAL * 2, 4))
        for i, frac in enumerate(lidar):
            c = np.array([1.0 - frac, frac * 0.8, 0.2, 0.30 + 0.35 * (1.0 - frac)])
            colors[2*i] = c; colors[2*i+1] = c
        self._lidar_lines.set_data(pos=pts, color=colors, connect="segments")

    def _refresh_hud(self, info: dict = None):
        env   = self.inner_env
        dist  = info.get("dist_to_goal", 0.0) if info else 0.0
        pos   = env.pos[0]
        n_obs = len(env.obstacle_ids)
        diff  = env.difficulty
        lo, hi = DIFFICULTY_OBSTACLE_COUNTS.get(diff, (0, 0))
        self._hud.text = (
            f"step      {self._step:4d}\n"
            f"reward  {self._ep_reward:+7.3f}\n"
            f"dist      {dist:.3f} m\n"
            f"xyz    {pos[0]:+.2f}  {pos[1]:+.2f}  {pos[2]:+.2f}\n"
            f"diff      {diff}  ({n_obs} obstacles)\n"
            f"[L] lidar {'ON' if self._show_lidar else 'OFF'}"
        )

    # ── episode run ───────────────────────────────────────────────────────────

    def run(self, model=None, action_fn=None, seed: int = 0, n_episodes: int = 1):
        """Block until window closed or all episodes complete."""
        if model is not None:
            self._action_fn = lambda o: model.predict(o, deterministic=True)[0]
        elif action_fn is not None:
            self._action_fn = action_fn
        else:
            raise ValueError("Supply model= or action_fn=")

        self._remaining_episodes = n_episodes
        self._seed_base          = seed
        self._current_seed       = seed
        self._start_episode()

        interval = float(getattr(self.inner_env, "CTRL_TIMESTEP", 1 / 30.0))
        self._timer = app.Timer(interval=interval, connect=self._on_tick, start=True)
        app.run()

    def _start_episode(self):
        self._obs = self.venv.reset()
        self._trail_buf = [self.inner_env.pos[0].copy()]
        self._step = 0
        self._ep_reward = 0.0
        self._done = False
        self._rebuild_obstacles()
        self._refresh_drone(self.inner_env.pos[0], self.inner_env.rpy[0])
        self._refresh_hud()

    def _on_tick(self, _event):
        if self._done:
            self._remaining_episodes -= 1
            if self._remaining_episodes <= 0:
                self._timer.stop()
                app.quit()
                return
            # next episode
            self._current_seed += 1
            self._done = False
            self._start_episode()
            return

        action = self._action_fn(self._obs)
        self._obs, reward, done, info = self.venv.step(action)
        self._step      += 1
        self._ep_reward += float(reward[0])

        pos = self.inner_env.pos[0].copy()
        rpy = self.inner_env.rpy[0].copy()

        self._trail_buf.append(pos)
        if len(self._trail_buf) > TRAIL_LEN:
            self._trail_buf.pop(0)

        self._refresh_drone(pos, rpy)
        self._refresh_trail()
        self._refresh_lidar(pos)
        self._refresh_hud(info[0])

        if done[0]:
            self._done = True
            success = info[0].get("success", False)
            outcome = "SUCCESS" if success else "timeout/crash"
            lo, hi  = DIFFICULTY_OBSTACLE_COUNTS.get(self.inner_env.difficulty, (0, 0))
            print(f"[stream_viz_v2] ep ended — {outcome} | "
                  f"diff={self.inner_env.difficulty} ({lo}–{hi} obs) | "
                  f"steps={self._step} | reward={self._ep_reward:.2f}")


# ── convenience wrapper ────────────────────────────────────────────────────────

def run_visual_episode_vispy_v2(model, vec_norm_path, difficulty: int = 1,
                                 seed: int = 0, n_episodes: int = 1):
    """Standalone entry point — creates env, runs Vispy demo, closes."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _make():
        return ObstacleAviaryV2(obs=ObservationType("kin"), act=ActionType("pid"),
                                difficulty=difficulty, gui=False)

    venv = DummyVecEnv([_make])
    if vec_norm_path and os.path.exists(vec_norm_path):
        venv = VecNormalize.load(vec_norm_path, venv)
        venv.training = False
        venv.norm_reward = False

    inner_env = venv.envs[0] if hasattr(venv, "envs") else venv.venv.envs[0]
    lo, hi = DIFFICULTY_OBSTACLE_COUNTS.get(difficulty, (0, 0))
    viz = DroneVizV2(
        inner_env, venv,
        title=f"V2 Obstacle Avoidance  |  difficulty {difficulty}  ({lo}–{hi} obstacles)"
    )
    try:
        viz.run(model=model, seed=seed, n_episodes=n_episodes)
    finally:
        venv.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _find_latest_model(results_dir: str = "results") -> str:
    candidates = sorted(glob.glob(
        os.path.join(results_dir, "obstacle_v2-*", "best_model.zip")
    ))
    if not candidates:
        raise FileNotFoundError(f"No v2 model found under '{results_dir}'.")
    return candidates[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vispy real-time v2 drone visualiser")
    parser.add_argument("--model_path", default=None, type=str, metavar="")
    parser.add_argument("--difficulty", default=2,    type=int, metavar="")
    parser.add_argument("--seed",       default=0,    type=int, metavar="")
    parser.add_argument("--episodes",   default=1,    type=int, metavar="")
    parser.add_argument("--output",     default="results", type=str, metavar="")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    mp = args.model_path or _find_latest_model(args.output)
    print(f"[INFO] Model: {mp}")

    model = PPO.load(mp)
    vec_norm = os.path.join(os.path.dirname(mp), "vec_normalize.pkl")
    if not os.path.exists(vec_norm):
        vec_norm = None
        print("[WARN] vec_normalize.pkl not found — running without obs normalization")

    run_visual_episode_vispy_v2(
        model, vec_norm,
        difficulty=args.difficulty,
        seed=args.seed,
        n_episodes=args.episodes,
    )
