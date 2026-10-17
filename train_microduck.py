#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "fire",
#     "imageio",
#     "imageio-ffmpeg",
#     "mujoco",
#     "numpy",
#     "torch",
#     "tqdm",
#     "x-evolution",
# ]
# [tool.uv.sources]
# x-evolution = { path = ".", editable = true }
# ///

from __future__ import annotations

import math
import os

os.environ.setdefault('ACCELERATE_USE_CPU', '1')

import shutil
import subprocess
from pathlib import Path

import fire
import imageio
import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import Module
from tqdm import tqdm

from x_evolution import EvoStrategy

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def parse_command(command):
    if not exists(command):
        return None

    if isinstance(command, str):
        command = [part for part in command.replace(' ', '').split(',') if part]

    return tuple(float(x) for x in command)

# central pattern generator

def to_phase(x):
    return torch.tensor(x, dtype = torch.float64) if isinstance(x, (int, float)) else torch.as_tensor(x)

class CPG:
    """Phase clock plus a sine-basis waveform."""

    def __init__(self, freq = 2.0, dt = 0.02):
        self.freq, self.dt = freq, dt

    def waveform(self, phi, amp, phase = 0.0, rectify = False, bias = 0.0):
        # phi (...,) + phase (..., dim) -> (..., dim)
        phi = to_phase(phi)
        amp, phase, bias = (torch.as_tensor(x, dtype = phi.dtype, device = phi.device) for x in (amp, phase, bias))
        rectify = torch.as_tensor(rectify).to(device = phi.device)

        s = torch.sin(phi[..., None] + phase)
        s = torch.where(rectify, s.clamp(min = 0.0), s)

        return bias + amp * s

    def advance(self, phi, freq = None):
        return (to_phase(phi) + 2.0 * math.pi * default(freq, self.freq) * self.dt) % (2.0 * math.pi)

def microduck_leg_waveform(clock, phi, cpg_action, swing, knee_lag):
    # slots -> left hip, left knee, right hip, right knee
    knee_scale = 1.0 + float(np.clip(cpg_action[1], -1.0, 1.0))
    phase_l, phase_r = cpg_action[2] * np.pi, cpg_action[3] * np.pi

    amp = np.array([swing, swing * knee_scale, swing, swing * knee_scale])
    phase = np.array([phase_l, phase_l - knee_lag, phase_r, phase_r - knee_lag])

    return clock.waveform(phi, amp, phase).numpy()

# ensure microduck_rl repository is cloned

def ensure_microduck_repo(path = 'microduck_rl') -> Path:
    subpath = Path('src/mjlab_microduck/robot/microduck/scene_walk.xml')

    for candidate in (Path(path), Path(f'../{path}')):
        scene = candidate / subpath
        if scene.exists():
            return scene

    repo = Path(path)
    if not repo.exists():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/pollen-robotics/microduck_rl.git', str(repo)], check = True)

    assert (repo / subpath).exists(), f'failed to find {repo / subpath}'
    return repo / subpath

# microduck environment

class MicroduckGaitEnv:
    def __init__(
        self,
        mjcf_path: Path | str | None = None,
        *,
        kp_mult: float = 3.0,
        kv: float = 0.1,
        horizon: int = 400,
        seed: int = 0,
        forced_swing: float | None = None,
        command: tuple[float, float] | None = (0.15, 0.0),
        command_scale: float = 5.0,
        control_hz: int = 50,
        n_substeps: int = 5,
        action_scale: float = 0.5,
        policy_range: tuple[float, float] = (-1.0, 1.0),
        cpg_freq: float = 2.0,
        cpg_swing_max: float = 0.5,
        cpg_knee_lag: float = np.pi / 2,
        track_weight: float = 2.0,
        track_std: tuple[float, float] = (0.02, 0.15),
        progress_weight: float = 20.0,
        distance_weight: float = 10.0,
        upright_weight: float = 2.0,
        upright_std: float = 0.05,
        pose_weight: float = 1.0,
        action_rate_weight: float = -0.1,
        air_time_weight: float = 3.0,
        air_time_min: float = 0.10,
        air_time_max: float = 0.25,
        air_time_ramp_episodes: int = 960,
        foot_height_weight: float = 3.0,
        foot_height_clear: float = 0.01,
        foot_height_cap: float = 0.02,
        nan_penalty: float = -10.0,
        floor_clip_penalty: float = -5.0,
        body_ang_vel_weight: float = -0.05,
        angular_momentum_weight: float = -0.02,
        foot_slip_weight: float = -0.1,
        foot_clearance_weight: float = -0.5,
        foot_clearance_target: float = 0.02,
        self_collision_weight: float = -1.0,
        min_trunk_z: float = 0.06,
        min_uprightness: float = 0.5,
        standing_trunk_z: float = 0.095,
        standing_uprightness: float = 0.85,
        floor_clip_eps: float = 0.02,
        reset_z_range: tuple[float, float] = (0.0, 0.01),
        reset_joint_noise: float = 0.05,
        reset_vel_noise: float = 0.1,
        lin_vel_range: tuple[float, float] = (-0.2, 0.2),
        ang_vel_range: tuple[float, float] = (-0.4, 0.4),
    ):
        mjcf_path = default(mjcf_path, ensure_microduck_repo)
        if callable(mjcf_path):
            mjcf_path = mjcf_path()

        self.mjcf_path = Path(mjcf_path)
        self.control_hz = control_hz
        self.n_substeps = n_substeps
        self.physics_dt = 1.0 / (control_hz * n_substeps)
        self.control_dt = 1.0 / control_hz
        self.action_scale = action_scale
        self.policy_range = policy_range

        self.cpg_freq = cpg_freq
        self.cpg_swing_max = cpg_swing_max
        self.cpg_knee_lag = cpg_knee_lag
        self.cpg_clock = CPG(freq = cpg_freq, dt = self.control_dt)
        self.cpg_phi = 0.0

        self.track_weight = track_weight
        self.track_std = track_std
        self.progress_weight = progress_weight
        self.distance_weight = distance_weight
        self.upright_weight = upright_weight
        self.upright_std = upright_std
        self.pose_weight = pose_weight
        self.action_rate_weight = action_rate_weight

        self.air_time_weight = air_time_weight
        self.air_time_min = air_time_min
        self.air_time_max = air_time_max
        self.air_time_ramp_episodes = air_time_ramp_episodes

        self.foot_height_weight = foot_height_weight
        self.foot_height_clear = foot_height_clear
        self.foot_height_cap = foot_height_cap

        self.nan_penalty = nan_penalty
        self.floor_clip_penalty = floor_clip_penalty
        self.body_ang_vel_weight = body_ang_vel_weight
        self.angular_momentum_weight = angular_momentum_weight
        self.foot_slip_weight = foot_slip_weight
        self.foot_clearance_weight = foot_clearance_weight
        self.foot_clearance_target = foot_clearance_target
        self.self_collision_weight = self_collision_weight

        self.min_trunk_z = min_trunk_z
        self.min_uprightness = min_uprightness
        self.standing_trunk_z = standing_trunk_z
        self.standing_uprightness = standing_uprightness
        self.floor_clip_eps = floor_clip_eps
        self.reset_z_range = reset_z_range
        self.reset_joint_noise = reset_joint_noise
        self.reset_vel_noise = reset_vel_noise
        self.lin_vel_range = lin_vel_range
        self.ang_vel_range = ang_vel_range

        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.model.opt.timestep = self.physics_dt

        self.model.actuator_gainprm[:, 0] *= kp_mult
        self.model.actuator_biasprm[:, 1] = -self.model.actuator_gainprm[:, 0]
        self.model.actuator_biasprm[:, 2] = -kv

        self.data = mujoco.MjData(self.model)
        stand = self.model.key('STAND')
        self.home_qpos = stand.qpos.copy()
        self.home_ctrl = stand.ctrl.copy()

        self.horizon = horizon
        self.action_dim = self.model.nu
        self.obs_dim = 3 + 3 + 3 * self.action_dim + 2
        self.forced_swing = forced_swing
        command = parse_command(command)
        self.fixed_command = np.asarray(command, dtype = np.float64) if exists(command) else None
        self.command_scale = command_scale

        self.rng = np.random.default_rng(seed)
        self.last_action = np.zeros(self.action_dim)
        self.command = np.zeros(2)
        self.steps = 0
        self.episodes = 0
        self.foot_air = np.zeros(2)

        self.trunk_body = 1
        self.world_up = np.array([0.0, 0.0, 1.0])

        self.foot_geoms = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'{side}_foot_collision')
            for side in ('left', 'right')
        ])
        self.foot_bodies = np.array([self.model.geom_bodyid[g] for g in self.foot_geoms])
        self.non_foot_bodies = np.array([i for i in range(self.model.nbody) if i not in self.foot_bodies and i != 0])

        actuator_id = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        self.cpg_action_dim = 4
        self.cpg_ctrl = np.array([actuator_id(n) for n in ('left_hip_pitch', 'left_knee', 'right_hip_pitch', 'right_knee')])
        self.direct_ctrl = np.setdiff1d(np.arange(self.action_dim), self.cpg_ctrl)

        self.pose_stds = np.array([
            0.30, 0.05, 0.40, 0.40, 0.25,
            0.35, 0.35, 0.35, 0.35,
            0.30, 0.05, 0.40, 0.40, 0.25,
        ])
        self.leg_mask = np.array([True] * 5 + [False] * 4 + [True] * 5)

    def velocity(self, local: bool = False) -> np.ndarray:
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.trunk_body, v, flg_local = local)
        return v

    def observe(self, safe: bool = False) -> np.ndarray:
        if safe:
            return np.concatenate([
                np.zeros(6),
                np.zeros(self.action_dim * 2),
                self.last_action,
                self.command * self.command_scale,
            ]).astype(np.float32)

        g = -self.data.xmat[self.trunk_body].reshape(3, 3).T @ self.world_up

        return np.concatenate([
            g,
            self.velocity(local = True)[:3],
            self.data.qpos[7:] - self.home_qpos[7:],
            self.data.qvel[6:],
            self.last_action,
            self.command * self.command_scale,
        ]).astype(np.float32)

    def reset(
        self,
        seed: int | None = None,
        command: tuple[float, float] | None = None
    ):
        if exists(seed):
            self.rng = np.random.default_rng(seed)

        self.data.qpos[:] = self.home_qpos
        self.data.qpos[2] += self.rng.uniform(*self.reset_z_range)
        self.data.qpos[7:] += self.rng.uniform(-self.reset_joint_noise, self.reset_joint_noise, size = self.action_dim)
        self.data.qvel[:] = self.rng.uniform(-self.reset_vel_noise, self.reset_vel_noise, size = self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        if exists(command):
            self.command = np.array(command, dtype = np.float32)
        elif exists(self.fixed_command):
            self.command = self.fixed_command.copy()
        elif self.rng.uniform() < 0.25:
            self.command = np.zeros(2, dtype = np.float32)
        else:
            self.command = np.array([
                self.rng.uniform(*self.lin_vel_range),
                self.rng.uniform(*self.ang_vel_range)
            ], dtype = np.float32)

        self.last_action[:] = 0.
        self.steps = 0
        self.cpg_phi = 0.0
        self.foot_air[:] = 0
        self.episodes += 1

        return self.observe()

    def step(self, action: np.ndarray):
        action = np.nan_to_num(action, nan = 0.0, posinf = 0.0, neginf = 0.0)
        action_eff = np.clip(action, *self.policy_range)
        action_rate = float(np.mean((action_eff - np.clip(self.last_action, *self.policy_range)) ** 2))

        # compute control targets

        cpg_action, direct = action[:self.cpg_action_dim], action[self.cpg_action_dim:]

        # action[0] modulates swing around the nominal half-amplitude
        swing = default(
            self.forced_swing,
            0.5 * self.cpg_swing_max * (1.0 + np.clip(cpg_action[0], -1.0, 1.0))
        )

        ctrl = self.home_ctrl.copy()
        ctrl[self.cpg_ctrl] += microduck_leg_waveform(
            self.cpg_clock, self.cpg_phi, cpg_action, swing, self.cpg_knee_lag
        )
        ctrl[self.direct_ctrl] += self.action_scale * np.clip(direct, *self.policy_range)

        self.cpg_phi = float(self.cpg_clock.advance(self.cpg_phi))
        self.data.ctrl[:] = ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        self.last_action[:] = action

        # check numerical stability

        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            return self.observe(safe = True), self.nan_penalty, True, False, {}

        # kinematics & orientation

        w, x, y, z = self.data.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v_world = self.velocity(local = False)
        vx = np.cos(yaw) * v_world[3] + np.sin(yaw) * v_world[4]
        wz = self.velocity(local = True)[2]

        uprightness = self.data.xmat[self.trunk_body][8]
        terminated = bool(uprightness < self.min_uprightness or self.data.qpos[2] < self.min_trunk_z)
        truncated = self.steps >= self.horizon

        body_z = self.data.xpos[self.non_foot_bodies, 2]
        min_body_z = float(body_z.min())
        floor_clip_depth = max(0.0, -min_body_z)

        if floor_clip_depth > self.floor_clip_eps:
            terminated = True

        # rewards

        reward = 0.0

        if floor_clip_depth > 0.:
            reward += self.floor_clip_penalty * floor_clip_depth * 100.

        r_track_lin = self.track_weight * np.exp(-((vx - self.command[0]) ** 2) / self.track_std[0])
        r_track_ang = self.track_weight * np.exp(-((wz - self.command[1]) ** 2) / self.track_std[1])
        r_upright = self.upright_weight * np.exp(-((1.0 - uprightness) ** 2) / self.upright_std)

        cmd_moving = abs(self.command[0]) > 0.01 or abs(self.command[1]) > 0.01
        pose_scale = 1.5 if cmd_moving else 0.5
        r_pose = self.pose_weight * np.exp(-(((self.data.qpos[7:] - self.home_qpos[7:])[self.leg_mask] / (self.pose_stds[self.leg_mask] * pose_scale)) ** 2).mean())
        r_action_rate = self.action_rate_weight * action_rate

        reward += r_track_lin + r_track_ang + r_upright + r_pose + r_action_rate

        if abs(self.command[0]) > 0.01:
            reward += self.progress_weight * vx * np.sign(self.command[0])

        is_standing_gait = bool((uprightness >= self.standing_uprightness) and (self.data.qpos[2] >= self.standing_trunk_z) and (floor_clip_depth == 0.0))

        if cmd_moving and is_standing_gait:
            geom1, geom2 = self.data.contact.geom1, self.data.contact.geom2

            for i, geom in enumerate(self.foot_geoms):
                if (geom1 == geom).any() or (geom2 == geom).any():
                    self.foot_air[i] = 0
                else:
                    self.foot_air[i] += 1

            air_time = self.foot_air * self.control_dt
            ramp = min(1.0, self.episodes / self.air_time_ramp_episodes)
            reward += self.air_time_weight * ramp * ((air_time > self.air_time_min) & (air_time < self.air_time_max)).sum()
        else:
            self.foot_air[:] = 0

        foot_z = self.data.geom_xpos[self.foot_geoms][:, 2]

        if is_standing_gait:
            reward += self.foot_height_weight * np.clip((foot_z - self.foot_height_clear) / self.foot_height_cap, 0., 1.).sum()

        ang_vel = self.velocity(local = True)[:3]
        reward += self.body_ang_vel_weight * float(np.sum(ang_vel ** 2))

        i_comp = self.data.cinert[self.trunk_body][:6]
        inertia = np.array([
            [i_comp[0], i_comp[3], i_comp[4]],
            [i_comp[3], i_comp[1], i_comp[5]],
            [i_comp[4], i_comp[5], i_comp[2]],
        ])
        reward += self.angular_momentum_weight * float(np.sum((inertia @ ang_vel) ** 2))

        self_hits = sum(1 for i in range(self.data.ncon) if 0 not in (int(self.data.contact[i].geom1), int(self.data.contact[i].geom2)))
        reward += self.self_collision_weight * float(self_hits > 0)

        if cmd_moving:
            slip = 0.0
            for body in self.foot_bodies:
                v = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, v, flg_local = 0)
                slip += v[3] ** 2 + v[4] ** 2

            reward += self.foot_slip_weight * slip + self.foot_clearance_weight * np.clip((self.foot_clearance_target - foot_z) / self.foot_clearance_target, 0., 1.).mean()

        if truncated and abs(self.command[0]) > 0.01:
            reward += self.distance_weight * float(self.data.qpos[0]) * np.sign(self.command[0])

        return self.observe(), float(reward), terminated, truncated, {}

    def close(self):
        pass

# evaluation & rendering

@torch.no_grad()
def evaluate_gait(
    policy,
    seeds: tuple[int, ...] = (0, 1, 2),
    horizon: int = 400,
    forced_swing: float | None = None,
    command: tuple[float, float] = (0.15, 0.0),
    command_scale: float = 5.0,
    kp_mult: float = 3.0,
    mjcf_path: str | Path | None = None,
):
    was_training = policy.training
    policy.eval()
    outs = []

    for seed in seeds:
        env = MicroduckGaitEnv(
            mjcf_path = mjcf_path,
            kp_mult = kp_mult,
            horizon = horizon,
            seed = seed,
            command = command,
            command_scale = command_scale,
            forced_swing = forced_swing
        )
        obs = env.reset(seed = seed, command = command)
        total, steps = 0.0, 0

        for _ in range(horizon):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            action = policy(obs_tensor)
            action_np = action.squeeze(0).cpu().numpy()

            obs, reward, term, trunc, _ = env.step(action_np)
            total += float(reward)
            steps += 1

            if term or trunc:
                break

        outs.append((steps, float(env.data.qpos[0]), total))
        env.close()

    policy.train(was_training)
    return tuple(float(np.mean([o[i] for o in outs])) for i in range(3))

def add_box(scene, pos, mat, half, rgba):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, half, pos, mat.reshape(-1), rgba)
    geom.emission = 1.0
    scene.ngeom += 1

def add_command_arrow(scene, env, command):
    command = parse_command(command) if exists(command) else tuple(env.command)
    vx, wz = command

    if abs(vx) < 0.01 and abs(wz) < 0.01:
        return

    head_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'neck_pitch')
    head_body = head_body if head_body >= 0 else env.trunk_body

    w, x, y, z = env.data.qpos[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    if abs(vx) > 0.01:
        theta = yaw + (0.0 if vx > 0 else math.pi) + wz * 0.8
    else:
        theta = yaw + (math.pi / 2 if wz > 0 else -math.pi / 2)

    direction = np.array([math.cos(theta), math.sin(theta), 0.0])
    side = np.array([-math.sin(theta), math.cos(theta), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    mat = np.stack([direction, side, up], axis = 1)

    length, width, head_len, head_w = 0.10, 0.008, 0.045, 0.010
    base = env.data.xpos[head_body] + np.array([0.0, 0.0, 0.12])
    tip = base + direction * length
    rgba = np.array([1.0, 0.82, 0.2, 0.95], np.float32)

    add_box(scene, base + direction * (length / 2), mat, np.array([length / 2, width, width]), rgba)

    for sign in (-1.0, 1.0):
        head_theta = theta + sign * math.pi / 4
        head_dir = np.array([math.cos(head_theta), math.sin(head_theta), 0.0])
        head_side = np.array([-math.sin(head_theta), math.cos(head_theta), 0.0])
        head_mat = np.stack([head_dir, head_side, up], axis = 1)
        add_box(
            scene, tip - head_dir * (head_len / 2), head_mat,
            np.array([head_len / 2, head_w, head_w]), rgba
        )

@torch.no_grad()
def render_gait(
    policy,
    mjcf_path: str | Path | None = None,
    *,
    kp_mult: float = 3.0,
    kv: float = 0.1,
    horizon: int = 400,
    out_path: str = 'videos/microduck_gait.mp4',
    forced_swing: float | None = None,
    command: tuple[float, float] = (0.15, 0.0),
    command_scale: float = 5.0,
    show_command_arrow: bool = True,
    camera_lookat: tuple[float, float, float] = (0.3, 0.0, 0.15),
    camera_distance: float = 1.0,
    camera_azimuth: float = 150.0,
    camera_elevation: float = -15.0,
    video_size: tuple[int, int] = (480, 640),
    control_hz: int = 50,
    seed: int = 0
):
    env = MicroduckGaitEnv(
        mjcf_path = mjcf_path,
        kp_mult = kp_mult,
        kv = kv,
        horizon = horizon,
        seed = seed,
        command = command,
        command_scale = command_scale,
        forced_swing = forced_swing,
        control_hz = control_hz,
    )

    renderer = mujoco.Renderer(env.model, height = video_size[0], width = video_size[1])
    camera = mujoco.MjvCamera()
    camera.lookat[:] = camera_lookat
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation

    obs = env.reset(seed = seed, command = command)
    frames = []

    for _ in tqdm(range(horizon), desc = 'rendering', leave = False):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        action = policy(obs_tensor)
        action_np = action.squeeze(0).cpu().numpy()

        obs, _, term, trunc, _ = env.step(action_np)

        renderer.update_scene(env.data, camera = camera)

        if show_command_arrow:
            add_command_arrow(renderer.scene, env, command)

        frames.append(renderer.render())

        if term or trunc:
            break

    out = Path(out_path)
    out.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(str(out), frames, fps = control_hz)
    print(f'saved gait video to {out}')

    renderer.close()
    env.close()

# policy model

class MicroduckPolicy(Module):
    def __init__(
        self,
        obs_dim: int = 50,
        action_dim: int = 14,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # initialize output projection near zero so initial posture holds home stance
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        return self.net(obs).clamp(-1.0, 1.0)


class MicroduckEnvironment(Module):
    def __init__(
        self,
        mjcf_path: Path | str | None = None,
        horizon: int = 400,
        forced_swing: float | None = None,
        kp_mult: float = 3.0,
        command: tuple[float, float] | None = None,
        anchor_command: tuple[float, float] | None = (0.15, 0.0),
        repeats: int = 2,
        seed: int = 42
    ):
        super().__init__()
        self.horizon = horizon
        self.command = command
        self.anchor_command = anchor_command
        self.repeats = repeats
        self.env = MicroduckGaitEnv(
            mjcf_path = mjcf_path,
            kp_mult = kp_mult,
            horizon = horizon,
            seed = seed,
            command = command,
            forced_swing = forced_swing
        )
        self.obs_dim = self.env.obs_dim
        self.action_dim = self.env.action_dim

    def forward(self, model):
        device = next(model.parameters()).device
        total_reward = 0.0

        for repeat in range(self.repeats):
            seed = torch.randint(0, int(1e9), ()).item()

            if exists(self.command):
                command = self.command
            elif exists(self.anchor_command):
                command = self.anchor_command if repeat % 2 == 0 else None
            else:
                command = None

            obs = self.env.reset(seed = seed, command = command)
            ep_reward = 0.0

            for _ in range(self.horizon):
                obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                action = model(obs_tensor)
                action_np = action.squeeze(0).detach().cpu().numpy()

                obs, reward, term, trunc, _ = self.env.step(action_np)
                ep_reward += float(reward)

                if term or trunc:
                    break

            total_reward += ep_reward

        return total_reward / self.repeats

# main training function

def train(
    num_generations: int = 10_000,
    noise_population_size: int = 128,
    noise_scale: float = 0.02,
    learning_rate: float = 3e-3,
    mirror_sampling: bool = True,
    hidden_dim: int = 128,
    horizon: int = 400,
    eval_horizon: int | None = None,
    forced_swing: float | None = None,
    kp_mult: float = 3.0,
    command: tuple[float, float] | None = None,
    eval_command: tuple[float, float] = (0.15, 0.0),
    eval_every: int = 50,
    eval_video: bool = True,
    checkpoint_dir: str = 'checkpoints-microduck',
    checkpoint_every: int = 100,
    video_dir: str = 'videos',
    seed: int = 42,
    repeats: int = 2,
    cpu: bool = True,
    init_policy: str | None = None,
    resume: bool = False,
    smoke: bool = False,
    clear_videos: bool = True,
):
    if smoke:
        num_generations = 2
        noise_population_size = 4
        horizon = 50
        eval_horizon = 50
        eval_every = 1
        checkpoint_every = 1

    eval_horizon = default(eval_horizon, horizon)

    torch.manual_seed(seed)
    np.random.seed(seed)

    command = parse_command(command)
    eval_command = parse_command(eval_command)

    is_main = int(os.environ.get('RANK', '0')) == 0

    ckpt_dir = Path(checkpoint_dir)

    try:
        ckpt_dir.mkdir(parents = True, exist_ok = True)
    except FileExistsError:
        pass

    video_path = Path(video_dir)

    if is_main:
        if clear_videos and video_path.exists():
            shutil.rmtree(video_path)
        video_path.mkdir(parents = True, exist_ok = True)

    best_ckpt = ckpt_dir / 'best.pt'
    last_ckpt = ckpt_dir / 'last.pt'
    winner_ckpt = ckpt_dir / 'winner.pt'

    env_module = MicroduckEnvironment(
        horizon = horizon,
        forced_swing = forced_swing,
        kp_mult = kp_mult,
        command = command,
        anchor_command = eval_command,
        repeats = repeats,
        seed = seed,
    )

    policy = MicroduckPolicy(
        obs_dim = env_module.obs_dim,
        action_dim = env_module.action_dim,
        hidden_dim = hidden_dim,
    )

    # auto-resume from last or best checkpoint if available

    if not exists(init_policy) and resume:
        if last_ckpt.exists():
            init_policy = str(last_ckpt)
        elif best_ckpt.exists():
            init_policy = str(best_ckpt)

    best_eval_steps = 0.0
    best_eval_ret = -float('inf')
    best_eval_x = -float('inf')

    if exists(init_policy) and Path(init_policy).exists():
        policy.load_state_dict(torch.load(init_policy, map_location = 'cpu'))

        if is_main:
            print(f'[microduck] warm-started policy from {init_policy}', flush = True)
            eval_init = evaluate_gait(
                policy,
                seeds = (0, 1, 2),
                horizon = eval_horizon,
                forced_swing = forced_swing,
                command = eval_command,
                kp_mult = kp_mult,
            )
            best_eval_steps, best_eval_x, best_eval_ret = eval_init[0], eval_init[1], eval_init[2]
            torch.save(policy.state_dict(), best_ckpt)
            print(f'[microduck] baseline: steps: {best_eval_steps:.1f}/{eval_horizon} | x: {best_eval_x:+.2f}m | ret: {best_eval_ret:.1f}', flush = True)

    consecutive_full_walks = 0

    def on_generation_callback(generation, fitnesses, current_model):
        nonlocal best_eval_steps, best_eval_ret, best_eval_x, consecutive_full_walks

        mean_fit = float(fitnesses.mean().item())
        max_fit = float(fitnesses.max().item())

        is_eval_gen = (generation % eval_every == 0) or (generation == 1) or (generation == num_generations)

        if not is_eval_gen:
            return

        eval_out = evaluate_gait(
            current_model,
            seeds = (0, 1, 2),
            horizon = eval_horizon,
            forced_swing = forced_swing,
            command = eval_command,
            kp_mult = kp_mult,
        )
        eval_steps, eval_x, eval_ret = eval_out[0], eval_out[1], eval_out[2]

        improved = (eval_steps > best_eval_steps) or (eval_steps == best_eval_steps and eval_ret > best_eval_ret)
        if improved:
            best_eval_steps = eval_steps
            best_eval_ret = eval_ret
            best_eval_x = eval_x
            torch.save(current_model.state_dict(), best_ckpt)

        # always save latest checkpoint

        torch.save(current_model.state_dict(), last_ckpt)

        # periodic snapshot checkpoint

        if exists(checkpoint_every) and (generation % checkpoint_every == 0):
            torch.save(current_model.state_dict(), ckpt_dir / f'model_{generation:05d}.pt')

        # render video on evaluation

        vid_path_str = ''
        if eval_video:
            vid_path = Path(video_dir) / f'eval_gen{generation:05d}_steps{eval_steps:03.0f}.mp4'
            try:
                render_gait(
                    current_model,
                    horizon = eval_horizon,
                    out_path = str(vid_path),
                    forced_swing = forced_swing,
                    command = eval_command,
                    kp_mult = kp_mult,
                )
                shutil.copyfile(vid_path, Path(video_dir) / 'microduck_best.mp4')
                vid_path_str = f' | video: {vid_path.name}'
            except Exception as e:
                print(f'[video render error] {e}', flush = True)

        print(
            f'\n[eval gen {generation:5d}/{num_generations}] '
            f'steps: {eval_steps:5.1f}/{eval_horizon} '
            f'| x: {eval_x:+.2f}m | ret: {eval_ret:8.1f} '
            f'| pop fit (mean/max): {mean_fit:7.1f}/{max_fit:7.1f} '
            f'| best: {best_eval_steps:5.1f} steps{vid_path_str}'
            f'{" *" if improved else ""}',
            flush = True,
        )

        # check convergence condition: full horizon walk with forward progress

        if eval_steps >= (eval_horizon - 5) and eval_x >= 0.20:
            consecutive_full_walks += 1
            if consecutive_full_walks >= 3:
                torch.save(current_model.state_dict(), winner_ckpt)
                print(
                    f'\n*** [convergence achieved] ***\n'
                    f'microduck walked full {eval_horizon} steps ({eval_x:+.2f}m) for 3 consecutive evaluations!\n'
                    f'saved winner checkpoint to {winner_ckpt}',
                    flush = True
                )
                if eval_video:
                    try:
                        render_gait(
                            current_model,
                            horizon = eval_horizon,
                            out_path = str(Path(video_dir) / 'microduck_winner.mp4'),
                            forced_swing = forced_swing,
                            command = eval_command,
                            kp_mult = kp_mult,
                        )
                    except Exception:
                        pass
        else:
            consecutive_full_walks = 0

    evo_strat = EvoStrategy(
        policy,
        environment = env_module,
        num_generations = num_generations,
        noise_population_size = noise_population_size,
        noise_scale = noise_scale,
        learning_rate = learning_rate,
        mirror_sampling = mirror_sampling,
        fitness_to_weighted_factor = 'centered_rank',
        use_optimizer = True,
        optimizer_klass = torch.optim.Adam,
        on_generation = on_generation_callback,
        checkpoint_every = checkpoint_every,
        checkpoint_path = str(ckpt_dir),
        cpu = cpu,
        verbose = True,
    )

    if is_main:
        print(
            f'[microduck] starting evolution ({num_generations} gens, pop_size = {noise_population_size}, '
            f'mirror = {mirror_sampling}, lr = {learning_rate}, noise_scale = {noise_scale})'
        )

    evo_strat()

    if is_main:
        torch.save(policy.state_dict(), last_ckpt)
        print(
            f'[microduck] training complete. best eval: {best_eval_steps:.1f} steps, '
            f'x: {best_eval_x:+.2f}m, ret: {best_eval_ret:.1f} | saved to {last_ckpt}'
        )

if __name__ == '__main__':
    fire.Fire(train)
