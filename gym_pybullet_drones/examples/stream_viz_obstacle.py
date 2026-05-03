"""stream_viz_obstacle.py — real-time Vispy visualiser for the obstacle-avoidance branch.

PyBullet runs headless (DIRECT mode); Vispy owns the window and render loop.
Supports variable-size box obstacles (queried from PyBullet each episode).

Usage
-----
    python stream_viz_obstacle.py --model results/obstacle_<ts>/best_model.zip
    python stream_viz_obstacle.py --model ... --num_obstacles 6 --seed 3
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

from gym_pybullet_drones.envs.ObstacleAviary import ObstacleAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ── tunables ───────────────────────────────────────────────────────────────────
TRAIL_LEN  = 300
ROTOR_SEGS = 28

# ── geometry helpers ───────────────────────────────────────────────────────────

def _rpy_matrix(rpy):
    r, p_, y = rpy
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p_), np.sin(p_)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _circle_pts(center, radius, n=ROTOR_SEGS):
    t = np.linspace(0, 2 * np.pi, n + 1)
    pts = np.zeros((n + 1, 3))
    pts[:, 0] = center[0] + radius * np.cos(t)
    pts[:, 1] = center[1] + radius * np.sin(t)
    pts[:, 2] = center[2]
    return pts


def _box_edge_pts(cx, cy, cz, hx, hy, hz):
    """24-point segment array for a wireframe box with independent half-extents."""
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


def _ground_grid(extent=3, step=0.5):
    lines = []
    for v in np.arange(-extent, extent + step, step):
        lines += [[-extent, v, 0], [extent, v, 0]]
        lines += [[v, -extent, 0], [v, extent, 0]]
    return np.array(lines)


def _obstacle_half_extents(oid, client_id):
    """Query PyBullet collision shape for a box's (hx, hy, hz)."""
    data = p.getCollisionShapeData(oid, -1, physicsClientId=client_id)
    if not data:
        return (0.15, 0.15, 0.15)
    return tuple(float(x) for x in data[0][3])   # GEOM_BOX dims = halfExtents


# ── visualiser ─────────────────────────────────────────────────────────────────

class DroneVizObstacle:
    """Real-time Vispy visualiser for the obstacle-avoidance ObstacleAviary.

    Parameters
    ----------
    env   : ObstacleAviary created with gui=False
    venv  : optional VecEnv wrapper (for SB3 predict); if None, env is used raw
    title : window title
    """

    def __init__(self, env, venv=None, title="Drone Obstacle Avoidance"):
        self.env        = env
        self.venv       = venv
        self._trail_buf = []
        self._step      = 0
        self._ep_reward = 0.0
        self._done      = False
        self._obs       = None
        self._action_fn = None
        self._timer     = None

        # ── canvas & camera ───────────────────────────────────────────────────
        self.canvas = scene.SceneCanvas(
            title=title, size=(1280, 720), bgcolor="#12121e",
            keys="interactive", show=True,
        )
        vb = self.canvas.central_widget.add_view()
        vb.camera = scene.TurntableCamera(elevation=22, azimuth=-50, distance=6.0, fov=50)
        self._vb = vb

        # ── ground grid ───────────────────────────────────────────────────────
        visuals.Line(
            pos=_ground_grid(), color=(0.25, 0.25, 0.35, 0.6),
            connect="segments", parent=vb.scene,
        )
        visuals.XYZAxis(parent=vb.scene)

        # ── drone body ────────────────────────────────────────────────────────
        self._body_mkr = visuals.Markers(parent=vb.scene)
        self._body_mkr.set_data(
            np.zeros((1, 3)),
            face_color=(0.15, 0.65, 1.0, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=14, edge_width=1.5,
        )

        # ── four arm lines ────────────────────────────────────────────────────
        self._arm_lines = [
            visuals.Line(
                pos=np.zeros((2, 3)), color=(1.0, 0.45, 0.15, 1.0),
                width=3.0, connect="segments", parent=vb.scene,
            )
            for _ in range(4)
        ]

        # ── four rotor rings ──────────────────────────────────────────────────
        self._rotor_rings = [
            visuals.Line(
                pos=_circle_pts(np.zeros(3), 0.04),
                color=(0.9, 0.85, 0.1, 0.85),
                width=1.8, connect="strip", parent=vb.scene,
            )
            for _ in range(4)
        ]

        # ── position trail ────────────────────────────────────────────────────
        self._trail_line = visuals.Line(
            pos=np.zeros((2, 3)), color=np.zeros((2, 4)),
            width=2.0, connect="strip", antialias=True, parent=vb.scene,
        )

        # ── target marker ─────────────────────────────────────────────────────
        self._target_mkr = visuals.Markers(parent=vb.scene)
        self._target_mkr.set_data(
            env.TARGET_POS.reshape(1, 3),
            face_color=(0.1, 0.95, 0.35, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=20, edge_width=2.0, symbol="star",
        )

        # ── obstacle wireframes (rebuilt each episode) ────────────────────────
        self._obs_visuals = []

        # ── HUD overlay ───────────────────────────────────────────────────────
        self._hud = visuals.Text(
            "", color=(1.0, 1.0, 1.0, 0.9),
            font_size=12, anchor_x="left", anchor_y="top",
            parent=self.canvas.scene,
        )
        self._hud.pos = (16, 16)

        # ── rotor geometry ────────────────────────────────────────────────────
        L = float(getattr(env, "L", 0.0397))
        s = L / np.sqrt(2)
        self._rotor_body = np.array([
            [ s, -s, 0.0], [-s, -s, 0.0],
            [-s,  s, 0.0], [ s,  s, 0.0],
        ])
        self._rotor_r = L * 0.75

    # ── scene setup per episode ────────────────────────────────────────────────

    def _rebuild_scene_objects(self):
        for v in self._obs_visuals:
            v.parent = None
        self._obs_visuals.clear()

        for oid in self.env._obstacle_ids:
            pos_w, _ = p.getBasePositionAndOrientation(oid, physicsClientId=self.env.CLIENT)
            hx, hy, hz = _obstacle_half_extents(oid, self.env.CLIENT)
            pts = _box_edge_pts(pos_w[0], pos_w[1], pos_w[2], hx, hy, hz)
            ln = visuals.Line(
                pos=pts, color=(0.9, 0.35, 0.1, 0.8),
                width=2.0, connect="segments", parent=self._vb.scene,
            )
            self._obs_visuals.append(ln)

        self._target_mkr.set_data(
            self.env.TARGET_POS.reshape(1, 3),
            face_color=(0.1, 0.95, 0.35, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=20, edge_width=2.0, symbol="star",
        )

    # ── per-step visuals ───────────────────────────────────────────────────────

    def _refresh_drone(self, pos, rpy):
        R = _rpy_matrix(rpy)
        self._body_mkr.set_data(
            pos.reshape(1, 3),
            face_color=(0.15, 0.65, 1.0, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=14, edge_width=1.5,
        )
        for i, offset_b in enumerate(self._rotor_body):
            hub = pos + R @ offset_b
            self._arm_lines[i].set_data(pos=np.array([pos, hub]), color=(1.0, 0.45, 0.15, 1.0))
            self._rotor_rings[i].set_data(pos=_circle_pts(hub, self._rotor_r), color=(0.9, 0.85, 0.1, 0.85))

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

    def _refresh_hud(self, dist=None, colliding=False):
        pos = self.env.pos[0]
        dist_str = f"{dist:.3f} m" if dist is not None else "—"
        col_flag = "  [COLLISION]" if colliding else ""
        self._hud.text = (
            f"step    {self._step:4d}{col_flag}\n"
            f"reward  {self._ep_reward:+7.3f}\n"
            f"dist    {dist_str}\n"
            f"xyz     {pos[0]:+.2f}  {pos[1]:+.2f}  {pos[2]:+.2f}"
        )

    # ── episode entry point ────────────────────────────────────────────────────

    def run(self, model=None, action_fn=None, seed=0, n_episodes=1):
        """Run n_episodes in the Vispy window. Blocks until window closes."""
        if model is not None:
            self._action_fn = lambda o: model.predict(o, deterministic=True)[0]
        elif action_fn is not None:
            self._action_fn = action_fn
        else:
            raise ValueError("Provide model= or action_fn=")

        self._n_episodes  = n_episodes
        self._ep_count    = 0
        self._seed        = seed
        self._start_episode(seed)

        interval = float(getattr(self.env, "CTRL_TIMESTEP", 1 / 30.0))
        self._timer = app.Timer(interval=interval, connect=self._on_tick, start=True)
        app.run()

    def _start_episode(self, seed):
        self._obs, _ = self.env.reset(seed=seed)
        self._trail_buf = [self.env.pos[0].copy()]
        self._step      = 0
        self._ep_reward = 0.0
        self._done      = False
        self._rebuild_scene_objects()
        self._refresh_drone(self.env.pos[0], self.env.rpy[0])
        self._refresh_hud()

    # ── timer callback ─────────────────────────────────────────────────────────

    def _on_tick(self, _event):
        if self._done:
            self._ep_count += 1
            if self._ep_count < self._n_episodes:
                self._start_episode(seed=self._seed + self._ep_count)
            else:
                self._timer.stop()
                print("[stream_viz] All episodes done.")
            return

        action                                          = self._action_fn(self._obs)
        self._obs, reward, terminated, truncated, info = self.env.step(action)
        self._step      += 1
        self._ep_reward += float(reward)

        pos = self.env.pos[0].copy()
        rpy = self.env.rpy[0].copy()

        self._trail_buf.append(pos)
        if len(self._trail_buf) > TRAIL_LEN:
            self._trail_buf.pop(0)

        self._refresh_drone(pos, rpy)
        self._refresh_trail()
        self._refresh_hud(
            dist=info.get("dist_to_target"),
            colliding=info.get("colliding", False),
        )

        if terminated or truncated:
            self._done = True
            if info.get("success"):
                outcome = "SUCCESS"
            elif info.get("colliding"):
                outcome = "COLLISION"
            else:
                outcome = "TIMEOUT"
            print(
                f"[stream_viz] Ep {self._ep_count+1}  {outcome}"
                f"  steps={self._step}  reward={self._ep_reward:.2f}"
                f"  dist={info.get('dist_to_target', 0):.3f} m"
            )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _find_latest_model(results_dir="results"):
    candidates = sorted(glob.glob(os.path.join(results_dir, "obstacle_*", "best_model.zip")))
    if not candidates:
        raise FileNotFoundError(f"No obstacle model found under '{results_dir}'.")
    return candidates[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vispy visualiser for obstacle-avoidance branch")
    parser.add_argument("--model",         default=None, type=str)
    parser.add_argument("--num_obstacles", default=4,    type=int)
    parser.add_argument("--n_episodes",    default=3,    type=int)
    parser.add_argument("--seed",          default=0,    type=int)
    parser.add_argument("--output",        default="results", type=str)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    mp = args.model or _find_latest_model(args.output)
    print(f"[INFO] Model: {mp}")
    model = PPO.load(mp)

    env = ObstacleAviary(
        obs=ObservationType.KIN,
        act=ActionType.VEL,
        num_obstacles=args.num_obstacles,
        gui=False,
    )
    viz = DroneVizObstacle(
        env,
        title=f"Obstacle Avoidance  |  {args.num_obstacles} obstacles",
    )
    try:
        viz.run(model=model, seed=args.seed, n_episodes=args.n_episodes)
    finally:
        env.close()
