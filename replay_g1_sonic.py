"""g1_sonic replay client: sends ego_view video + 43-D state to the .240 ZMQ
server (simulating a sim/robot), collects the model's 78-D actions + generated
video, and produces VISUAL verification artifacts:

  1. montage.mp4      : model-generated video | GT ego_view video (side by side)
  2. joints.png       : whole-body joint trajectories (decoded motion_token vs GT)
  3. hands.png        : 14-D hand joint trajectories (model vs GT)
  4. token.png        : 64-D motion_token trajectories (model vs GT)
  5. metrics.json     : summary numbers (MAE, rel-corr per modality)

Usage (in the dreamzero container on .240, server already running):
  python3 replay_g1_sonic.py --episode 0 --start 0 --nchunks 5
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zmq
from openpi_client import msgpack_numpy

DS = Path("/datasets/mncai_G1_Dex3_Trash_LocoManipulation_GEAR")
SONIC_DEC = os.environ.get("SONIC_DECODER", "/workspace/checkpoints/sonic_v1_1/model_decoder.onnx")


def load_episode(ep):
    pq = DS / f"data/chunk-000/episode_{ep:06d}.parquet"
    df = pd.read_parquet(pq)
    state = np.array(df["state_qpos"].tolist(), float)  # (N,29)
    hand = np.array(df["state_hand"].tolist(), float)    # (N,14)
    state43 = np.concatenate([state, hand], axis=1)       # (N,43) grouped
    ahand = np.array(df["action_hand"].tolist(), float)   # (N,14) GT hands
    atok = np.array(df["action_token"].tolist(), float)   # (N,64) GT token
    tasks = [json.loads(l) for l in (DS / "meta/tasks.jsonl").read_text().splitlines()]
    task_idx = int(df["task_index"].iloc[0])
    prompt = tasks[task_idx]["task"] if task_idx < len(tasks) else "pick up the trash bag"
    return df, state43, ahand, atok, prompt


def read_video_frames(ep, start, end):
    vp = DS / f"videos/chunk-000/observation.images.ego_view/episode_{ep:06d}.mp4"
    reader = imageio.get_reader(vp)
    frames = []
    for i, f in enumerate(reader):
        if i < start:
            continue
        if i >= end:
            break
        frames.append(np.asarray(f))
    reader.close()
    return np.stack(frames, axis=0)  # (T,H,W,3)


def decode_token(token, onnx_path):
    """Decode 64-D GEAR-SONIC v1.1 motion token -> 43-D joints via onnx decoder.
    Returns (N,43) or None if decoder unavailable."""
    if not os.path.exists(onnx_path):
        return None
    try:
        import onnxruntime as ort
        so = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        in_name = so.get_inputs()[0].name
        out = so.run(None, {in_name: token.astype(np.float32)})[0]
        return np.asarray(out, float)
    except Exception as e:
        print(f"  [warn] SONIC decoder failed: {e}")
        return None


def rel_corr(a, b):
    a, b = a.flatten(), b.flatten()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5550)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=5)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(f"/docker_data/logs/replay/replay_g1_sonic_ep{args.episode}_s{args.start}_n{args.nchunks}")
    outdir.mkdir(parents=True, exist_ok=True)

    df, state43, ahand_gt, atok_gt, prompt = load_episode(args.episode)
    N = len(df)
    print(f"episode {args.episode}: {N} frames, prompt={prompt!r}")
    print(f"  state43 {state43.shape}, GT hand {ahand_gt.shape}, GT token {atok_gt.shape}")

    # ZMQ connection
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 300_000)
    sock.setsockopt(zmq.SNDTIMEO, 120_000)
    sock.connect(f"tcp://{args.host}:{args.port}")
    packer = msgpack_numpy.Packer()

    # metadata
    sock.send(packer.pack({"endpoint": "metadata"}))
    meta = msgpack_numpy.unpackb(sock.recv())
    print(f"server metadata: {meta}")
    napb = int(meta.get("num_action_per_block", 24))

    # GT video segment (from start, nchunks*napb frames)
    seg_end = min(N, args.start + args.nchunks * napb)
    gt_frames = read_video_frames(args.episode, args.start, seg_end)
    print(f"GT segment: {gt_frames.shape}")

    # replay: send frame + state at each anchor
    preds_hand = []
    preds_tok = []
    for c in range(args.nchunks):
        a = args.start + c * napb
        if a >= N:
            break
        frame = gt_frames[c * napb] if (c * napb) < gt_frames.shape[0] else gt_frames[-1]
        obs = {
            "endpoint": "infer",
            "observation/ego_view": frame,          # (H,W,3)
            "observation/state": state43[a],        # (43,)
            "prompt": prompt,
        }
        sock.send(packer.pack(obs))
        resp = msgpack_numpy.unpackb(sock.recv())
        if isinstance(resp, dict) and "error" in resp:
            print(f"  [error] {resp['error'][:400]}")
            break
        h = np.asarray(resp.get("action.hand_joints", np.zeros(14)))
        t = np.asarray(resp.get("action.token", np.zeros(64)))
        if h.ndim == 2:
            h = h[0]  # first step of the 24-step chunk (action at the anchor)
        if t.ndim == 2:
            t = t[0]
        h = h.flatten()
        t = t.flatten()
        preds_hand.append(h)
        preds_tok.append(t)
        print(f"  chunk {c}: anchor={a} hand={np.round(h[:4],2).tolist()} tok[:4]={np.round(t[:4],2).tolist()}")

    # save model video
    sock.send(packer.pack({"endpoint": "save_video"}))
    sv = msgpack_numpy.unpackb(sock.recv())
    model_video_path = sv.get("path") if isinstance(sv, dict) else None
    print(f"save_video: {sv}")
    sock.send(packer.pack({"endpoint": "reset"}))
    try:
        sock.recv()
    except Exception:
        pass
    sock.close()

    preds_hand = np.stack(preds_hand, axis=0) if preds_hand else np.zeros((0, 14))
    preds_tok = np.stack(preds_tok, axis=0) if preds_tok else np.zeros((0, 64))

    # ---- metrics (per anchor, vs GT at anchor) ----
    metrics = {"episode": args.episode, "start": args.start, "nchunks": args.nchunks, "prompt": prompt}
    anchors = [args.start + c * napb for c in range(len(preds_hand))]
    if len(preds_hand):
        gt_h = ahand_gt[anchors]
        gt_t = atok_gt[anchors]
        metrics["hand"] = {"mae": float(np.abs(preds_hand - gt_h).mean()),
                           "rel_corr": [rel_corr(preds_hand[i], gt_h[i]) for i in range(len(preds_hand))]}
        metrics["token"] = {"mae": float(np.abs(preds_tok - gt_t).mean()),
                            "rel_corr": [rel_corr(preds_tok[i], gt_t[i]) for i in range(len(preds_tok))],
                            "cos_sim": [float((preds_tok[i] @ gt_t[i]) / (np.linalg.norm(preds_tok[i]) * np.linalg.norm(gt_t[i]) + 1e-8)) for i in range(len(preds_tok))]}
    # decoded whole-body joints
    dec = decode_token(preds_tok, SONIC_DEC) if len(preds_tok) else None
    metrics["sonic_decoded"] = None
    if dec is not None:
        dec43 = dec[:, :43]
        gt_wbc = np.array(df["action.wbc"].tolist(), float)[anchors]
        metrics["sonic_decoded"] = {
            "shape": list(dec43.shape),
            "mae_vs_wbc": float(np.abs(dec43 - gt_wbc).mean()),
            "rel_corr": [rel_corr(dec43[i], gt_wbc[i]) for i in range(len(dec43))],
        }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))

    # save raw predictions + per-joint std to diagnose unresponsive hands
    if len(preds_hand):
        np.savez(outdir / "preds.npz",
                 preds_hand=preds_hand, preds_tok=preds_tok,
                 gt_hand=gt_h, gt_tok=gt_t, anchors=np.array(anchors))
        hp = preds_hand.std(axis=0); hg = gt_h.std(axis=0)
        tp = preds_tok.std(axis=0); tg = gt_t.std(axis=0)
        print("  hand per-joint std  model: " + " ".join(f"{x:.2f}" for x in hp))
        print("  hand per-joint std  GT   : " + " ".join(f"{x:.2f}" for x in hg))
        print(f"  hand std overall: model={hp.mean():.4f} GT={hg.mean():.4f} | "
              f"token std overall: model={tp.mean():.4f} GT={tg.mean():.4f}")

    # ---- montage video (model | GT) ----
    if model_video_path and os.path.exists(model_video_path):
        model_frames = imageio.mimread(model_video_path)
        model_frames = np.stack(model_frames, axis=0)
        Tm = model_frames.shape[0]
        # resample GT to the model's frame count so both cover the full episode at the same speed
        gt_resampled = gt_frames[np.linspace(0, gt_frames.shape[0] - 1, Tm).astype(int)]
        T = min(Tm, gt_resampled.shape[0])
        if T > 0:
            mh, mw = model_frames[0].shape[:2]
            gh, gw = gt_resampled[0].shape[:2]
            H = max(mh, gh)
            rows = []
            for i in range(T):
                mf = model_frames[i][:H, :mw]
                gf = gt_resampled[i][:H, :gw]
                row = np.zeros((H, mw + gw + 4, 3), dtype=np.uint8)
                row[:mf.shape[0], :mf.shape[1]] = mf
                row[:gf.shape[0], mw + 4:mw + 4 + gf.shape[1]] = gf
                rows.append(row)
            imageio.mimsave(outdir / "montage.mp4", rows, fps=5, codec="libx264")
            print(f"saved montage.mp4 ({T} frames)")
    else:
        print("[warn] no model video to montage")

    # ---- joint plots ----
    if len(preds_hand):
        # hands (14-D)
        fig, axes = plt.subplots(14, 1, figsize=(10, 24), sharex=True)
        for d in range(14):
            axes[d].plot(anchors, preds_hand[:, d], "r-", label="model")
            axes[d].plot(anchors, ahand_gt[anchors, d], "b-", label="GT")
            axes[d].set_title(f"hand[{d}]", fontsize=8)
            if d == 0:
                axes[d].legend(fontsize=7)
        plt.xlabel("frame (anchor)"); plt.tight_layout(); plt.savefig(outdir / "hands.png", dpi=90); plt.close()
        # token (64-D)
        fig, axes = plt.subplots(16, 4, figsize=(16, 16), sharex=True)
        for d in range(64):
            ax = axes[d // 4, d % 4]
            ax.plot(anchors, preds_tok[:, d], "r-", label="model")
            ax.plot(anchors, atok_gt[anchors, d], "b-", label="GT")
            ax.set_title(f"tok[{d}]", fontsize=7)
        plt.xlabel("frame (anchor)"); plt.tight_layout(); plt.savefig(outdir / "token.png", dpi=90); plt.close()
        # decoded whole-body (43-D) vs GT wbc
        if dec is not None:
            dec43 = dec[:, :43]
            gt_wbc = np.array(df["action.wbc"].tolist(), float)[anchors]
            fig, axes = plt.subplots(43, 1, figsize=(10, 40), sharex=True)
            for d in range(43):
                axes[d].plot(anchors, dec43[:, d], "r-", label="model(decode)")
                axes[d].plot(anchors, gt_wbc[:, d], "b-", label="GT wbc")
                axes[d].set_title(f"joint[{d}]", fontsize=7)
                if d == 0:
                    axes[d].legend(fontsize=7)
            plt.xlabel("frame (anchor)"); plt.tight_layout(); plt.savefig(outdir / "joints.png", dpi=90); plt.close()
            print("saved joints.png (decoded 43-D vs GT)")
        print("saved hands.png + token.png")

    print(f"\nDONE. artifacts in {outdir}")


if __name__ == "__main__":
    main()
