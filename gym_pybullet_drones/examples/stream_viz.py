"""stream_viz.py — real-time 3-D drone visualiser using Vispy.

PyBullet runs fully headless (DIRECT mode); Vispy owns the window and render
loop.  State (position, orientation, obstacles, target) is pulled from the
environment every control tick and rendered without touching the physics.

Install deps
------------
    pip install "vispy[pyqt5]" pyopengl
    # or: pip install "vispy[pyside6]" pyopengl

Standalone usage
----------------
    python stream_viz.py                            # auto-find latest model
    python stream_viz.py --model_path <zip>
    python stream_viz.py --difficulty 2 --seed 3

Integration with play_obstacle.py
----------------------------------
    from stream_viz import run_visual_episode_vispy
    run_visual_episode_vispy(model, difficulty=1)   # drop-in for run_visual_episode
"""

import argparse
import glob
import os

import numpy as np

# ── Vispy import with helpful error ───────────────────────────────────────────
try:
    from vispy import app, scene
    from vispy.scene import visuals
except ImportError as exc:
    raise SystemExit(
        "Vispy not found. Install it with:\n"
        '  pip install "vispy[pyqt5]" pyopengl\n'
        "or use pyside6 / pyqt6 instead of pyqt5."
    ) from exc

from gym_pybullet_drones.envs.ObstacleAviary import ObstacleAviary, OBSTACLE_HALF_EXTENT
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ── tunables ──────────────────────────────────────────────────────────────────
TRAIL_LEN   = 250    # number of past positions to draw
ROTOR_SEGS  = 28     # polygon segments per rotor disc

# ── geometry helpers ──────────────────────────────────────────────────────────

def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw → 3×3 rotation matrix (ZYX / extrinsic convention)."""
    r, p, y = rpy
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [0, 0, 1]])
    Ry = np.array([[cp,  0, sp], [ 0,   1, 0], [-sp, 0, cp]])
    Rx = np.array([[ 1,  0,  0], [ 0,  cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _circle_pts(center: np.ndarray, radius: float, n: int = ROTOR_SEGS) -> np.ndarray:
    """(n+1, 3) array tracing a circle in the body XY plane."""
    t   = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.zeros((n + 1, 3))
    pts[:, 0] = center[0] + radius * np.cos(t)
    pts[:, 1] = center[1] + radius * np.sin(t)
    pts[:, 2] = center[2]
    return pts


def _box_edge_pts(cx: float, cy: float, cz: float, h: float) -> np.ndarray:
    """(24, 3) array: 12 edges × 2 endpoints for a wireframe axis-aligned cube."""
    d       = h
    corners = np.array([
        [cx-d, cy-d, cz-d], [cx+d, cy-d, cz-d],
        [cx+d, cy+d, cz-d], [cx-d, cy+d, cz-d],
        [cx-d, cy-d, cz+d], [cx+d, cy-d, cz+d],
        [cx+d, cy+d, cz+d], [cx-d, cy+d, cz+d],
    ])
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # bottom face
        (4,5),(5,6),(6,7),(7,4),   # top face
        (0,4),(1,5),(2,6),(3,7),   # verticals
    ]
    pts = np.empty((len(edges) * 2, 3))
    for i, (a, b) in enumerate(edges):
        pts[2*i],   pts[2*i+1] = corners[a], corners[b]
    return pts


def _ground_grid(extent: int = 3, step: float = 0.5) -> np.ndarray:
    """(N, 3) segment-pair array forming a flat grid in the z=0 plane."""
    lines = []
    vals  = np.arange(-extent, extent + step, step)
    for v in vals:
        lines += [[-extent, v, 0], [extent, v, 0]]
        lines += [[v, -extent, 0], [v,  extent, 0]]
    return np.array(lines)


# ── main visualiser ────────────────────────────────────────────────────────────

class DroneViz:
    """Real-time Vispy visualiser for a single-drone gym-pybullet-drones env.

    Parameters
    ----------
    env   : any single-drone BaseAviary subclass (must be created with gui=False)
    title : window title string
    """

    def __init__(self, env, title: str = "Drone Stream Viz"):
        self.env         = env
        self._trail_buf  = []
        self._step       = 0
        self._ep_reward  = 0.0
        self._done       = False
        self._obs        = None
        self._action_fn  = None
        self._timer      = None

        # ── canvas & camera ───────────────────────────────────────────────────
        self.canvas = scene.SceneCanvas(
            title=title, size=(1280, 720), bgcolor="#12121e",
            keys="interactive", show=True
        )
        vb = self.canvas.central_widget.add_view()
        vb.camera = scene.TurntableCamera(
            elevation=20, azimuth=-55, distance=5.0, fov=50
        )
        self._vb = vb

        # ── ground grid ───────────────────────────────────────────────────────
        visuals.Line(
            pos=_ground_grid(), color=(0.25, 0.25, 0.35, 0.6),
            connect="segments", parent=vb.scene
        )

        # ── coordinate axis (small, at origin) ───────────────────────────────
        visuals.XYZAxis(parent=vb.scene)

        # ── drone body centre sphere ──────────────────────────────────────────
        self._body_mkr = visuals.Markers(parent=vb.scene)
        self._body_mkr.set_data(
            np.zeros((1, 3)),
            face_color=(0.15, 0.65, 1.0, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=14, edge_width=1.5
        )

        # ── four arm lines (centre → rotor hub) ──────────────────────────────
        self._arm_lines = [
            visuals.Line(
                pos=np.zeros((2, 3)), color=(1.0, 0.45, 0.15, 1.0),
                width=3.0, connect="segments", parent=vb.scene
            )
            for _ in range(4)
        ]

        # ── four rotor rings ──────────────────────────────────────────────────
        self._rotor_rings = [
            visuals.Line(
                pos=_circle_pts(np.zeros(3), 0.04),
                color=(0.9, 0.85, 0.1, 0.85),
                width=1.8, connect="strip", parent=vb.scene
            )
            for _ in range(4)
        ]

        # ── position trail ────────────────────────────────────────────────────
        self._trail_line = visuals.Line(
            pos=np.zeros((2, 3)),
            color=np.zeros((2, 4)),
            width=1.8, connect="strip", antialias=True,
            parent=vb.scene
        )

        # ── target marker (star) ──────────────────────────────────────────────
        tgt = getattr(env, "TARGET_POS", np.array([1.0, 0.0, 1.0]))
        self._target_mkr = visuals.Markers(parent=vb.scene)
        self._target_mkr.set_data(
            tgt.reshape(1, 3),
            face_color=(0.1, 0.95, 0.35, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=18, edge_width=2.0, symbol="star"
        )

        # ── obstacle wireframes (rebuilt each episode) ────────────────────────
        self._obs_visuals = []

        # ── HUD text overlay ──────────────────────────────────────────────────
        self._hud = visuals.Text(
            "", color=(1.0, 1.0, 1.0, 0.9),
            font_size=12, anchor_x="left", anchor_y="top",
            parent=self.canvas.scene
        )
        self._hud.pos = (16, 16)

        # ── rotor geometry (derived from arm length) ──────────────────────────
        L = float(getattr(env, "L", 0.0397))
        s = L / np.sqrt(2)
        # body-frame rotor centres: front-right, rear-right, rear-left, front-left
        self._rotor_body = np.array([
            [ s, -s, 0.0],
            [-s, -s, 0.0],
            [-s,  s, 0.0],
            [ s,  s, 0.0],
        ])
        self._rotor_r = L * 0.75

    # ── obstacle / goal setup ─────────────────────────────────────────────────

    def _rebuild_scene_objects(self):
        """Reconstruct obstacle wireframes and goal marker for a new episode."""
        for v in self._obs_visuals:
            v.parent = None
        self._obs_visuals.clear()

        env = self.env
        if hasattr(env, "obstacle_ids"):
            import pybullet as p
            for oid in env.obstacle_ids:
                pos_w, _ = p.getBasePositionAndOrientation(
                    oid, physicsClientId=env.CLIENT
                )
                pts = _box_edge_pts(*pos_w, OBSTACLE_HALF_EXTENT)
                ln = visuals.Line(
                    pos=pts, color=(0.9, 0.2, 0.2, 0.75),
                    width=2.0, connect="segments", parent=self._vb.scene
                )
                self._obs_visuals.append(ln)

        tgt = getattr(env, "TARGET_POS", np.array([1.0, 0.0, 1.0]))
        self._target_mkr.set_data(
            tgt.reshape(1, 3),
            face_color=(0.1, 0.95, 0.35, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=18, edge_width=2.0, symbol="star"
        )

    # ── per-step visual updates ───────────────────────────────────────────────

    def _refresh_drone(self, pos: np.ndarray, rpy: np.ndarray):
        R = _rpy_matrix(rpy)

        self._body_mkr.set_data(
            pos.reshape(1, 3),
            face_color=(0.15, 0.65, 1.0, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=14, edge_width=1.5
        )

        for i, offset_b in enumerate(self._rotor_body):
            hub = pos + R @ offset_b
            self._arm_lines[i].set_data(
                pos=np.array([pos, hub]),
                color=(1.0, 0.45, 0.15, 1.0)
            )
            # rotor ring in world XY (un-tilted intentionally — easier to read)
            self._rotor_rings[i].set_data(
                pos=_circle_pts(hub, self._rotor_r),
                color=(0.9, 0.85, 0.1, 0.85)
            )

    def _refresh_trail(self):
        n = len(self._trail_buf)
        if n < 2:
            return
        pts    = np.array(self._trail_buf)
        alpha  = np.linspace(0.05, 0.85, n)
        colors = np.zeros((n, 4))
        colors[:, 0] = 0.15
        colors[:, 1] = 0.55
        colors[:, 2] = 1.0
        colors[:, 3] = alpha
        self._trail_line.set_data(pos=pts, color=colors, connect="strip")

    def _refresh_hud(self, info: dict = None):
        env  = self.env
        tgt  = getattr(env, "TARGET_POS", None)
        if info is not None:
            dist = info.get("dist_to_goal", 0.0)
        elif tgt is not None:
            dist = float(np.linalg.norm(env.pos[0] - tgt))
        else:
            dist = 0.0

        pos = env.pos[0]
        self._hud.text = (
            f"step    {self._step:4d}\n"
            f"reward  {self._ep_reward:+7.3f}\n"
            f"dist    {dist:.3f} m\n"
            f"xyz     {pos[0]:+.2f}  {pos[1]:+.2f}  {pos[2]:+.2f}"
        )

    # ── episode entry point ───────────────────────────────────────────────────

    def run(self, model=None, action_fn=None, seed: int = 0):
        """Run one episode.  Blocks until window is closed or episode ends.

        Supply either a stable-baselines3 ``model`` or a raw callable
        ``action_fn(obs) → action``.
        """
        if model is not None:
            self._action_fn = lambda o: model.predict(o, deterministic=True)[0]
        elif action_fn is not None:
            self._action_fn = action_fn
        else:
            raise ValueError("Provide model= or action_fn=")

        self._obs, _    = self.env.reset(seed=seed)
        self._trail_buf = [self.env.pos[0].copy()]
        self._step      = 0
        self._ep_reward = 0.0
        self._done      = False

        self._rebuild_scene_objects()
        self._refresh_drone(self.env.pos[0], self.env.rpy[0])
        self._refresh_hud()

        interval = float(getattr(self.env, "CTRL_TIMESTEP", 1 / 30.0))
        self._timer = app.Timer(
            interval=interval, connect=self._on_tick, start=True
        )
        app.run()

    # ── timer callback (sim + render per tick) ────────────────────────────────

    def _on_tick(self, _event):
        if self._done:
            self._timer.stop()
            return

        action                                              = self._action_fn(self._obs)
        self._obs, reward, terminated, truncated, info     = self.env.step(action)
        self._step      += 1
        self._ep_reward += float(reward)

        pos = self.env.pos[0].copy()
        rpy = self.env.rpy[0].copy()

        self._trail_buf.append(pos)
        if len(self._trail_buf) > TRAIL_LEN:
            self._trail_buf.pop(0)

        self._refresh_drone(pos, rpy)
        self._refresh_trail()
        self._refresh_hud(info)

        if terminated or truncated:
            self._done = True
            outcome = "success" if terminated else "timeout/crash"
            print(f"[stream_viz] Episode ended — {outcome} "
                  f"| steps={self._step} | reward={self._ep_reward:.2f}")


# ── convenience wrapper ────────────────────────────────────────────────────────

def run_visual_episode_vispy(model, difficulty: int = 1, seed: int = 0):
    """Drop-in replacement for play_obstacle.run_visual_episode using Vispy.

    Creates the environment headlessly, runs one episode in the Vispy window,
    then closes the env.  Call this instead of run_visual_episode(model, ...)
    when you want the upgraded renderer.
    """
    env = ObstacleAviary(
        obs=ObservationType("kin"),
        act=ActionType("pid"),
        difficulty=difficulty,
        gui=False,
    )
    viz = DroneViz(env, title=f"Obstacle Avoidance  |  difficulty {difficulty}")
    try:
        viz.run(model=model, seed=seed)
    finally:
        env.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _find_latest_model(results_dir: str = "results") -> str:
    candidates = sorted(glob.glob(os.path.join(results_dir, "obstacle-*", "best_model.zip")))
    if not candidates:
        raise FileNotFoundError(
            f"No obstacle model found under '{results_dir}'. "
            "Train one with learn_obstacle.py first, or pass --model_path."
        )
    return candidates[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vispy real-time drone visualiser")
    parser.add_argument("--model_path", default=None, type=str, metavar="")
    parser.add_argument("--difficulty", default=1,    type=int, metavar="")
    parser.add_argument("--seed",       default=0,    type=int, metavar="")
    parser.add_argument("--output",     default="results", type=str, metavar="")
    args = parser.parse_args()

    mp = args.model_path or _find_latest_model(args.output)
    print(f"[INFO] Loading model: {mp}")

    from stable_baselines3 import PPO
    run_visual_episode_vispy(PPO.load(mp), difficulty=args.difficulty, seed=args.seed)
