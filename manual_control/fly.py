#!/usr/bin/env python3
"""
Manual keyboard control for a single Crazyflie drone in PyBullet.

Run from the repo root:
    python manual_control/fly.py

Controls
--------
  Arrow Up / Down    — Pitch forward / backward  (body frame)
  Arrow Left / Right — Strafe left / right        (body frame)
  E / X              — Ascend / Descend
  A / D              — Yaw left / right
  R                  — Reset drone to origin
  Q / ESC            — Quit

Note: W/S/Z/C/V/G are reserved by PyBullet's built-in visualization shortcuts.
"""

import sys
import os
import time
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

# ── Tunable parameters ───────────────────────────────────────────────────────
DRONE     = DroneModel.CF2X
PYB_FREQ  = 240          # Hz — physics steps per second
CTRL_FREQ = 48           # Hz — control loop frequency
MAX_SPEED = 0.6          # m/s — max speed per axis from key input
MAX_YAW   = np.pi / 3   # rad/s — max yaw rate from key input
HOVER_ALT = 0.5          # m — initial hover altitude
ALT_FLOOR = 0.05         # m — hard floor on target altitude
ALT_CEIL  = 3.0          # m — hard ceiling on target altitude
# ────────────────────────────────────────────────────────────────────────────


def _read_keys(client: int, yaw: float) -> np.ndarray:
    """Return a world-frame velocity command [vx, vy, vz, yaw_rate] from key state.

    Arrow keys are interpreted in the drone's body frame then rotated into
    world frame using the current yaw, so forward/back/strafe feel natural
    regardless of heading.
    """
    keys = p.getKeyboardEvents(physicsClientId=client)

    vx_b, vy_b, vz, yr = 0.0, 0.0, 0.0, 0.0

    if p.B3G_UP_ARROW    in keys and keys[p.B3G_UP_ARROW]    & p.KEY_IS_DOWN:
        vx_b += MAX_SPEED
    if p.B3G_DOWN_ARROW  in keys and keys[p.B3G_DOWN_ARROW]  & p.KEY_IS_DOWN:
        vx_b -= MAX_SPEED
    if p.B3G_LEFT_ARROW  in keys and keys[p.B3G_LEFT_ARROW]  & p.KEY_IS_DOWN:
        vy_b += MAX_SPEED
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        vy_b -= MAX_SPEED
    if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
        vz += MAX_SPEED
    if ord('x') in keys and keys[ord('x')] & p.KEY_IS_DOWN:
        vz -= MAX_SPEED
    if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
        yr += MAX_YAW
    if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
        yr -= MAX_YAW

    # Rotate body-frame lateral commands into world frame
    c, s   = np.cos(yaw), np.sin(yaw)
    vx_w   = c * vx_b - s * vy_b
    vy_w   = s * vx_b + c * vy_b

    return np.array([vx_w, vy_w, vz, yr])


def _should_quit(client: int) -> bool:
    keys = p.getKeyboardEvents(physicsClientId=client)
    return (27      in keys and keys[27]      & p.KEY_WAS_TRIGGERED) or \
           (ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED)


def _should_reset(client: int) -> bool:
    keys = p.getKeyboardEvents(physicsClientId=client)
    return ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED


def main():
    env = CtrlAviary(
        drone_model=DRONE,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, HOVER_ALT]]),
        physics=Physics.PYB,
        pyb_freq=PYB_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=True,
        obstacles=False,
        user_debug_gui=False,
    )
    CLIENT = env.getPyBulletClient()
    ctrl   = DSLPIDControl(drone_model=DRONE)

    obs, _ = env.reset()

    target_pos = np.array([0.0, 0.0, HOVER_ALT])
    target_yaw = 0.0
    action     = np.zeros((1, 4))
    dt         = env.CTRL_TIMESTEP

    # HUD: static help line
    p.addUserDebugText(
        "↑↓ fwd/back  ←→ strafe  E/X alt  A/D yaw  R reset  Q quit",
        textPosition=[0, 0, HOVER_ALT + 1.4],
        textColorRGB=[1.0, 0.9, 0.0],
        textSize=1.1,
        physicsClientId=CLIENT,
    )
    # HUD: live position/yaw readout (updated each loop)
    hud_state = p.addUserDebugText(
        "pos: (+0.00, +0.00, 0.50)  yaw: +0°",
        textPosition=[0, 0, HOVER_ALT + 1.0],
        textColorRGB=[0.7, 0.9, 1.0],
        textSize=1.0,
        physicsClientId=CLIENT,
    )

    print("\n[manual_control] PyBullet window open — click the window first to capture keys.")
    print("  Arrow Up/Down   : forward / backward  (body frame)")
    print("  Arrow Left/Right: strafe left / right  (body frame)")
    print("  E / X           : ascend / descend")
    print("  A / D           : yaw left / right")
    print("  R               : reset to origin")
    print("  Q / ESC         : quit\n")

    step  = 0
    START = time.time()

    try:
        while True:
            try:
                if _should_quit(CLIENT):
                    break

                if _should_reset(CLIENT):
                    obs, _ = env.reset()
                    ctrl.reset()
                    target_pos = np.array([0.0, 0.0, HOVER_ALT])
                    target_yaw = 0.0
                    step       = 0
                    START      = time.time()
                    continue

                # ── Read keys, build velocity command ──────────────────────
                cur_yaw = float(obs[0][9])   # obs[0][9] = yaw (radians)
                vel_cmd = _read_keys(CLIENT, cur_yaw)
                vx, vy, vz, yr = vel_cmd

                # Integrate velocity command into a position setpoint
                target_pos[0] += vx * dt
                target_pos[1] += vy * dt
                target_pos[2]  = np.clip(target_pos[2] + vz * dt, ALT_FLOOR, ALT_CEIL)
                target_yaw    += yr * dt

                # ── PID → RPMs ───────────────────────────────────────────
                action[0, :], _, _ = ctrl.computeControlFromState(
                    control_timestep=dt,
                    state=obs[0],
                    target_pos=target_pos,
                    target_rpy=np.array([0.0, 0.0, target_yaw]),
                )

                obs, _, _, _, _ = env.step(action)

                # ── Update live HUD every 10 steps ───────────────────────
                if step % 10 == 0:
                    px, py, pz = obs[0][0], obs[0][1], obs[0][2]
                    hud_state = p.addUserDebugText(
                        f"pos: ({px:+.2f}, {py:+.2f}, {pz:.2f})  yaw: {np.degrees(cur_yaw):+.0f}°",
                        textPosition=[0, 0, HOVER_ALT + 1.0],
                        textColorRGB=[0.7, 0.9, 1.0],
                        textSize=1.0,
                        replaceItemUniqueId=hud_state,
                        physicsClientId=CLIENT,
                    )

                sync(step, START, dt)
                step += 1

            except p.error as e:
                if "Not connected" in str(e):
                    break   # window was closed — exit cleanly
                raise

    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("[manual_control] Closed.")


if __name__ == "__main__":
    main()
