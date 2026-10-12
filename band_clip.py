# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Adapted from GE-Sim-V2 src/gesim/conditioning/band.py (_render_traj_fast), (c) AgiBot, CC BY-NC-SA 4.0,
# https://github.com/AgibotTech/GE-Sim-V2 at commit 4a34498. Changes: the offframe="clip" mode and
# gripper_uv(); offframe="skip" keeps the original behaviour. This file is therefore CC BY-NC-SA 4.0,
# not MIT like the rest of this repository.
"""Trajectory band renderer with an option to keep grippers that project outside the image.

GE-Sim's renderer (gesim.conditioning.band._render_traj_fast) skips a gripper's circle and axis
sticks whenever its projected origin (0.15 m tool offset) lies outside the 512x384 canvas, so an arm
working near the bottom edge (e.g. task 573 drawer opening) gets no action conditioning at all.

offframe="skip" reproduces the original renderer exactly (verified bit-identical, see
eval/check_band_clip.py). offframe="clip" draws the glyph at its true (off-canvas) position and lets
the canvas clip it, so it slides out of view continuously; frames whose origins are inside the canvas
are rendered exactly as before. Points behind the camera are always skipped.

Also provides gripper_uv() (projected flange / tool-tip pixels) for the gripper-weighted loss.
"""
import matplotlib.cm as cm
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gesim.conditioning.band import _quat_pose_to_transform_matrix, _rpy_pose_to_quat_pose

H_DEF, W_DEF = 384, 512
GRIPPER_OFFSET = 0.15


def render_traj(eef_rpy, c2w, intrinsic, offframe="skip", sample_size=(H_DEF, W_DEF), gripper_offset=GRIPPER_OFFSET):
    """Same contract as gesim.conditioning.band.render_traj: (T,14), (V,T,4,4), (V,3,3) -> (3,V,T,H,W) in [0,1]."""
    assert offframe in ("skip", "clip"), offframe
    H, W = int(sample_size[0]), int(sample_size[1])
    eef_rpy = np.asarray(eef_rpy, dtype=np.float32)
    c2w_np = np.asarray(c2w, dtype=np.float32)
    intr_np = np.asarray(intrinsic, dtype=np.float32)
    V, T = c2w_np.shape[0], c2w_np.shape[1]
    assert eef_rpy.shape == (T, 14) and intr_np.shape == (V, 3, 3), (eef_rpy.shape, intr_np.shape)

    pose_t = _rpy_pose_to_quat_pose(eef_rpy).numpy()  # (T, 16)
    pose_l_mat = _quat_pose_to_transform_matrix(torch.from_numpy(pose_t[:, 0:7])).numpy()
    pose_r_mat = _quat_pose_to_transform_matrix(torch.from_numpy(pose_t[:, 8:15])).numpy()
    correct = np.eye(4, dtype=np.float32)
    correct[2, 3] = gripper_offset
    ee_key_pts_T = np.array([[0, 0, 0, 1], [0.1, 0, 0, 1], [0, 0.1, 0, 1], [0, 0, 0.1, 1]], dtype=np.float32).T

    norm_l = ((pose_t[:, 7] * (120 - 35)) + 35) / 120.0
    norm_r = ((pose_t[:, 15] * (120 - 35)) + 35) / 120.0
    col_l_f = np.array(cm.Greens(norm_l)[:, :3], dtype=np.float32)
    col_r_f = np.array(cm.Reds(norm_r)[:, :3], dtype=np.float32)
    sc_l_f = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.float32)
    sc_r_f = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

    out = np.full((3, V, T, H, W), 50.0 / 255.0, dtype=np.float32)
    half_w = 4
    ys = np.arange(H, dtype=np.int32)
    xs = np.arange(W, dtype=np.int32)
    offsets = np.arange(-half_w, half_w + 1, dtype=np.int32)

    for icam in range(V):
        w2c = np.linalg.inv(c2w_np[icam])
        K = intr_np[icam]
        cam = out[:, icam]
        cpos = c2w_np[icam, :, :3, 3]
        l_dist = np.clip(1.0 - np.sqrt(((pose_t[:, 0:3] - cpos) ** 2).sum(-1)) - 0.07 / (0.8 - 0.07), 0.0, 1.0) * 100
        r_dist = np.clip(1.0 - np.sqrt(((pose_t[:, 8:11] - cpos) ** 2).sum(-1)) - 0.07 / (0.8 - 0.07), 0.0, 1.0) * 100

        def _proj(pose_mat):
            pts_cam = (w2c @ pose_mat @ correct) @ ee_key_pts_T  # (T, 4, 4) column vectors
            pp = K @ pts_cam[:, :3, :]
            uv = (pp[:, :2, :] / pp[:, 2:3, :]).transpose(0, 2, 1)
            return uv.astype(np.int32), pts_cam[:, 2, 0] > 0  # (T, 4, 2), origin in front of camera

        uvs_l, front_l = _proj(pose_l_mat)
        uvs_r, front_r = _proj(pose_r_mat)

        def _draw(u, v, front):
            if offframe == "skip":
                return 0 <= u < W and 0 <= v < H
            return bool(front)

        # circles
        for t in range(T):
            for (u, v), color_f, rad, front in ((uvs_l[t, 0], col_l_f[t], int(l_dist[t]), front_l[t]),
                                                (uvs_r[t, 0], col_r_f[t], int(r_dist[t]), front_r[t])):
                u, v = int(u), int(v)
                if not (_draw(u, v, front) and rad > 0):
                    continue
                y0, y1 = max(0, v - rad), min(H, v + rad + 1)
                x0, x1 = max(0, u - rad), min(W, u + rad + 1)
                if y0 >= y1 or x0 >= x1:
                    continue
                dy = ys[y0:y1] - v
                dx = xs[x0:x1] - u
                mask = (dy[:, None] ** 2 + dx[None, :] ** 2) <= rad * rad
                cam[:, t, y0:y1, x0:x1][:, mask] = color_f[:, None]

        # sticks (lines); the original's per-pixel clipping already handles off-canvas endpoints
        for stick_idx in range(1, 4):
            for uvs, colors_f, front in ((uvs_l, sc_l_f, front_l), (uvs_r, sc_r_f, front_r)):
                color_f = colors_f[(stick_idx - 1) % 3]
                for t in range(T):
                    u0, v0 = int(uvs[t, 0, 0]), int(uvs[t, 0, 1])
                    u1, v1 = int(uvs[t, stick_idx, 0]), int(uvs[t, stick_idx, 1])
                    if not _draw(u0, v0, front[t]):
                        continue
                    ddu, ddv = u1 - u0, v1 - v0
                    if ddu == 0 and ddv == 0:
                        continue
                    if abs(ddu) >= abs(ddv) and ddu != 0:
                        step = 1 if ddu > 0 else -1
                        xs_ = np.arange(u0, u1 + step, step, dtype=np.int32)
                        xs_ = xs_[(xs_ >= 0) & (xs_ < W)]
                        if xs_.size == 0:
                            continue
                        vs_ = np.round(v0 + ddv * (xs_ - u0) / ddu).astype(np.int32)
                        vs_h = vs_[None, :] + offsets[:, None]
                        xs_h = np.broadcast_to(xs_[None, :], vs_h.shape)
                        m = (vs_h >= 0) & (vs_h < H)
                        cam[:, t, vs_h[m], xs_h[m]] = color_f[:, None]
                    elif ddv != 0:
                        step = 1 if ddv > 0 else -1
                        ys_ = np.arange(v0, v1 + step, step, dtype=np.int32)
                        ys_ = ys_[(ys_ >= 0) & (ys_ < H)]
                        if ys_.size == 0:
                            continue
                        us_ = np.round(u0 + ddu * (ys_ - v0) / ddv).astype(np.int32)
                        us_h = us_[None, :] + offsets[:, None]
                        ys_h = np.broadcast_to(ys_[None, :], us_h.shape)
                        m = (us_h >= 0) & (us_h < W)
                        cam[:, t, ys_h[m], us_h[m]] = color_f[:, None]
    return torch.from_numpy(out)


def gripper_uv(end_pos, end_quat, K, c2w, gripper_offset=GRIPPER_OFFSET):
    """Projected flange and tool-tip pixels per arm: (T, 2 arms, 2 points [flange, tip], 3 [u, v, depth])."""
    w2c = np.linalg.inv(c2w)
    out = np.zeros((end_pos.shape[0], 2, 2, 3), np.float32)
    for a in range(2):
        z_axis = Rotation.from_quat(end_quat[:, a]).as_matrix()[:, :, 2]
        for k, p in enumerate((end_pos[:, a], end_pos[:, a] + gripper_offset * z_axis)):
            pc = p @ w2c[:3, :3].T + w2c[:3, 3]
            uv = pc @ K.T
            out[:, a, k, :2] = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
            out[:, a, k, 2] = pc[:, 2]
    return out
