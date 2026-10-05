#!/usr/bin/env python3
"""CORRECTED teacher-forcing test for the 12k G1 Dex3 checkpoint.

Fixes vs /home/astrazzeri/teacher_forced_test.py (2026-09-26 version):
  1. GEOMETRY: uses the TRAINING transform (center_crop 0.95 -> resize to 320x176
     SQUISHED, no bars).  The old script used a 235px pillarbox inset by 42px,
     so the GT latents were structurally OOD vs training -> that run is invalid.
  2. PAIRED: for every anchor it runs NORMAL and TEACHER-FORCED inference on the
     SAME inputs in the same process, so the comparison is exactly controlled.

Saves raw data for later hypothesis work: preds_normal, preds_tf, gt,
state_anchors, per-chunk/per-joint errors, and a metrics.json.

Usage (inside the dreamzero container):
  python /docker_data/tf_paired.py --model-path ./checkpoints/run_20260925_123935/checkpoint-12000 \
      --dataset /datasets/G1_Dex3_GraspSquare_1ep --episode 0 --start 0 --num-chunks 8
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

CAM_KEYS = ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
JOINT_KEYS = ("left_arm_pos", "right_arm_pos", "left_hand_pos", "right_hand_pos")
SL = {"left_arm_pos": (0, 7), "right_arm_pos": (7, 14),
      "left_hand_pos": (14, 21), "right_hand_pos": (21, 28)}
CHUNK = 48
VIEW_W, VIEW_H = 320, 176          # TRAINING geometry (VideoResize 176x320)
N_FRAMES = 9                       # video.delta_indices = [0..8]
GROUPS = [("l_arm", 0, 7), ("r_arm", 7, 14), ("l_hand", 14, 21),
          ("r_hand", 21, 28), ("ARMS", 0, 14), ("HANDS", 14, 28), ("ALL", 0, 28)]


def center_crop_095(img):
    h, w = img.shape[:2]
    ch, cw = int(h * 0.95), int(w * 0.95)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def resize_np(img, w, h):
    from PIL import Image
    return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((w, h), Image.BILINEAR),
                      dtype=np.uint8)


def squished_view(frame):
    """TRAINING geometry: 4:3 frame cropped 0.95 then squished to fill 320x176."""
    return resize_np(center_crop_095(frame), VIEW_W, VIEW_H)


def build_model_grid(frames, cam_keymap):
    """(T,640,352,3) uint8. Model layout: TL=head, TR=right wrist, BL=left wrist, BR=black."""
    T = next(iter(frames.values())).shape[0]
    grid = np.zeros((T, 2 * VIEW_H, 2 * VIEW_W, 3), dtype=np.uint8)
    quads = {"cam_left_high": (0, 0), "cam_right_wrist": (0, VIEW_W),
             "cam_left_wrist": (VIEW_H, 0)}
    for cam, (y0, x0) in quads.items():
        for t in range(T):
            grid[t, y0:y0 + VIEW_H, x0:x0 + VIEW_W, :] = squished_view(frames[cam_keymap[cam]][t])
    return grid


def to_grid_tensor(grid):
    x = torch.from_numpy(grid.astype(np.float32) / 255.0)
    x = x.permute(3, 0, 1, 2).unsqueeze(0)
    return (x - 0.5) / 0.5


def load_episode_frames(root, video_keys, cam_keymap, episode, start, end):
    import av
    out = {cam: [] for cam in CAM_KEYS}
    for cam in CAM_KEYS:
        rel = video_keys[cam]
        p = root / rel.replace("video.", "videos/").replace(".", "/") if False else None
        vf = root / "videos" / Path(rel.replace("video.", "")).parent
        cand = list((root / "videos").rglob(f"episode_{episode:06d}.mp4"))
        c = av.open(str([q for q in cand if Path(rel).name in str(q)][0]))
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        c.seek(int(start / float(c.streams.video[0].average_rate or 30)), any_frame=False)
        got = 0
        for i, fr in enumerate(c.decode(video=0)):
            if i < start:
                continue
            if got >= (end - start):
                break
            out[cam].append(fr.to_ndarray(format="rgb24"))
            got += 1
        c.close()
    return {cam: np.stack(v) for cam, v in out.items()}


def corr(x, y):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-chunks", type=int, default=8)
    ap.add_argument("--out", default="/docker_data/logs/tf_experiment")
    ap.add_argument("--tag", default="tf_paired")
    args = ap.parse_args()

    sys.path.insert(0, "/workspace")
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    from groot.vla.data.schema import EmbodimentTag

    os.environ.setdefault("ENABLE_DIT_CACHE", "false")
    os.environ.setdefault("ATTENTION_BACKEND", "TE")
    torch._dynamo.config.recompile_limit = 100
    if "RANK" not in os.environ:
        import socket as _sock
        _s = _sock.socket(); _s.bind(("127.0.0.1", 0))
        _free_port = str(_s.getsockname()[1]); _s.close()
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=_free_port,
                          RANK="0", WORLD_SIZE="1")
        print(f"[init] using free MASTER_PORT={_free_port}", flush=True)
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("unitree_g1_upper_body_dex3"),
        model_path=args.model_path, device="cuda", device_mesh=mesh,
        skip_assert_delta_indices=True)
    head = policy.trained_model.action_head

    root = Path(args.dataset)
    info = json.loads((root / "meta" / "info.json").read_text())
    video_keys = {}
    for cam in CAM_KEYS:
        for k, v in info["features"].items():
            if isinstance(v, dict) and v.get("dtype") == "video" and cam in k:
                video_keys[cam] = k; break
    import pandas as pd
    df = pd.read_parquet(sorted((root / "data" / "chunk-000").glob("*.parquet"))[0])
    state_col = np.asarray(np.stack(df["observation.state"].to_list()), dtype=np.float64)
    action_col = np.asarray(np.stack(df["action"].to_list()), dtype=np.float64)

    cam_keymap = {c: c for c in CAM_KEYS}
    n = args.num_chunks
    anchors = [args.start + CHUNK * j for j in range(n)]

    P_n, P_t, GT, SA = [], [], [], []
    meta_rows = []
    for j, a in enumerate(anchors):
        g_end = min(a + N_FRAMES, len(state_col))
        frames9 = load_episode_frames(root, video_keys, cam_keymap, args.episode, a, g_end)
        grid9 = build_model_grid(frames9, cam_keymap)
        with torch.no_grad():
            gt_lat_all = head.encode_video(to_grid_tensor(grid9).to(torch.bfloat16).cuda())
        nfpb = head.num_frame_per_block
        gt_lat = gt_lat_all[:, :, -nfpb:].to(torch.bfloat16).cuda()

        obs = {}
        for cam in CAM_KEYS:
            obs[f"video.{cam}"] = frames9[cam_keymap[cam]][0].astype(np.uint8)
        for sk in JOINT_KEYS:
            lo, hi = SL[sk]
            obs[f"state.{sk}"] = state_col[a][lo:hi].astype(np.float64).reshape(1, -1)
        obs["annotation.language.action_text"] = \
            "Stack the three cubic blocks on the desktop from bottom to top in the order of red, yellow, and blue on the black tape affixed to the desktop."

        def run(**kw):
            from tianshou.data import Batch
            policy._reset_inference_state() if hasattr(policy, "_reset_inference_state") else None
            with torch.no_grad():
                r, _ = policy.lazy_joint_forward_causal(Batch(obs=dict(obs)), **kw)
            act = r.act
            return np.concatenate([np.asarray(
                act[f"action.{sk}"].cpu().numpy() if torch.is_tensor(act[f"action.{sk}"])
                else act[f"action.{sk}"]) for sk in JOINT_KEYS], axis=1).reshape(CHUNK, 28)

        t0 = time.time(); pn = run(); dtn = time.time() - t0
        # route GT latents through the repo's OWN parameter (latent_video)
        t0 = time.time(); pt = run(latent_video=gt_lat); dtt = time.time() - t0

        gt = np.stack([action_col[a + t] for t in range(CHUNK)]).astype(np.float64)
        P_n.append(pn); P_t.append(pt); GT.append(gt); SA.append(state_col[a].astype(np.float64))
        meta_rows.append(dict(chunk=j, anchor=a, t_normal_s=round(dtn, 2), t_tf_s=round(dtt, 2)))
        print(f"chunk {j} anchor {a}: normal {dtn:.1f}s  tf {dtt:.1f}s  "
              f"|pn-pt|={np.abs(pn - pt).mean():.4f} rad", flush=True)

    P_n, P_t, GT, SA = np.stack(P_n), np.stack(P_t), np.stack(GT), np.stack(SA)

    m = {"tag": args.tag, "geometry": "TRAINING squished 320x176 (center_crop 0.95)",
         "n_frames": N_FRAMES, "chunk": CHUNK, "anchors": anchors,
         "episode": args.episode, "start": args.start, "timing": meta_rows,
         "groups": {}}
    print(f"\n===== PAIRED normal vs teacher-forced (geometry = TRAINING) =====")
    for name, s, e in GROUPS:
        gn = gt_rel = None
        pn_rel = P_n[:, :, s:e] - SA[:, None, s:e]
        pt_rel = P_t[:, :, s:e] - SA[:, None, s:e]
        gt_rel = GT[:, :, s:e] - SA[:, None, s:e]
        row = {
            "mae_normal": float(np.abs(pn_rel - gt_rel).mean()),
            "mae_tf": float(np.abs(pt_rel - gt_rel).mean()),
            "corr_normal": corr(pn_rel, gt_rel),
            "corr_tf": corr(pt_rel, gt_rel),
            "mean_abs_delta_normal": float(np.abs(pn_rel).mean()),
            "mean_abs_delta_tf": float(np.abs(pt_rel).mean()),
            "mean_abs_delta_gt": float(np.abs(gt_rel).mean()),
            "pred_change_from_tf": float(np.abs(pn_rel - pt_rel).mean()),
        }
        m["groups"][name] = row
        print(f"  {name:<7} MAE normal={row['mae_normal']:.4f} tf={row['mae_tf']:.4f} | "
              f"rel-corr normal={row['corr_normal']:.4f} tf={row['corr_tf']:.4f} | "
              f"|delta| normal={row['mean_abs_delta_normal']:.4f} tf={row['mean_abs_delta_tf']:.4f} "
              f"gt={row['mean_abs_delta_gt']:.4f}")

    dmn = P_n - SA[:, None, :]; dmt = P_t - SA[:, None, :]
    m["sign_consistency_normal"] = int(np.all(np.sign(dmn.mean(1)) == np.sign(dmn.mean(1))[0], axis=0).sum())
    m["sign_consistency_tf"] = int(np.all(np.sign(dmt.mean(1)) == np.sign(dmt.mean(1))[0], axis=0).sum())
    m["per_chunk_mae_normal"] = [float(np.abs((P_n[i] - SA[i]) - (GT[i] - SA[i])).mean()) for i in range(n)]
    m["per_chunk_mae_tf"] = [float(np.abs((P_t[i] - SA[i]) - (GT[i] - SA[i])).mean()) for i in range(n)]

    out = Path(args.out) / f"{args.tag}_ep{args.episode}_s{args.start}_n{n}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "preds_normal_tf_gt_state.npz",
                        preds_normal=P_n, preds_tf=P_t, gt=GT, state_anchors=SA,
                        anchors=np.array(anchors),
                        joint_names=np.array([k for k in JOINT_KEYS]),
                        group_bounds=np.array([[0, 7], [7, 14], [14, 21], [21, 28]]))
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"\nsaved -> {out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()