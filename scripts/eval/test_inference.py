"""Replay training frames to the DreamZero ZMQ inference server and score the
returned 24-step action chunks against the dataset's ground-truth actions.

Server behavior (verified against socket_test_g1_dex3.py + WANPolicyHead):
every infer call is a cold-start I2V from the FIRST frame sent, conditioned
on the sent state vector; the returned actions are absolute (the server
denormalizes and adds the sent state back). So for anchor a_j = start + 24*j
we send frame a_j + state[a_j] and expect actions ~= action[a_j : a_j+24].

Every run writes ALL of its artifacts into its own directory:
    <out>/replay_<YYYYmmdd_HHMMSS>_ep<E>_s<S>_n<N>/
containing: metrics.json (all derived metrics + auto verdict),
preds_gt_state.npz (preds/gt/err, FULL state+action window, anchor states,
joint group slices, fps), per-anchor input frames (anchor_XX_views.png,
anchor_XX_model_input_grid.png), per_joint_error.png, per_joint_bias.png,
per_chunk_mae.png, pred_vs_gt_timeseries.png, server_grid.jpg,
model_video.mp4, video_mad.csv, montage.png.
The npz is self-sufficient for regenerating any plot (e.g. SVG) offline.

Run inside the dreamzero container:
    python /workspace/test_inference.py --episode 0 --start 0 --num-chunks 5
    python /workspace/test_inference.py --smoke   # legacy zero-frame check
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zmq
from openpi_client import msgpack_numpy

packer = msgpack_numpy.Packer()

CAM_KEYS = ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist")
JOINT_KEYS = ("left_arm_pos", "right_arm_pos", "left_hand_pos", "right_hand_pos")
SHORT = {"left_arm_pos": "l_arm", "right_arm_pos": "r_arm", "left_hand_pos": "l_hand", "right_hand_pos": "r_hand"}
CHUNK = 24
VIEW_W, VIEW_H = 320, 176
BAR = 32
PB_W, PB_BAR = 235, 42  # pillarbox: 4:3 content at 176px height -> 235px wide, (320-235)//2 = 42px bars L/R
SERVER_DEBUG_GRID = "/workspace/debug_quadrant_padded.jpg"


def parse_args():
    p = argparse.ArgumentParser(
        description="Replay training frames to the DreamZero ZMQ server and score actions vs GT"
    )
    p.add_argument("--host", default=os.environ.get("DREAMZERO_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("DREAMZERO_PORT", "5550")))
    p.add_argument("--dataset", default="/datasets/G1_Dex3_AllMerged_GEAR")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num-chunks", type=int, default=5)
    p.add_argument("--prompt", default=None, help="language prompt override (default: episode task text)")
    p.add_argument("--out", default="/docker_data/logs/replay", help="base dir; each run lands in <out>/<run_id>/")
    p.add_argument("--no-save-video", action="store_true", help="skip save_video at the end")
    p.add_argument("--server-layout", default="nobar", choices=["nobar", "letterbox", "pillarbox"],
                   help="layout the server currently applies to the 4 views (for the saved anchor input-grid image)")
    p.add_argument("--no-video-compare", action="store_true", help="skip model-video vs GT video analysis")
    p.add_argument("--smoke", action="store_true", help="legacy zero-frame smoke test (no dataset)")
    return p.parse_args()


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _flatten_names(names):
    if isinstance(names, list) and names and isinstance(names[0], list):
        return [n for sub in names for n in sub]
    return names


def load_episode_data(dataset, episode):
    root = Path(dataset)
    meta = root / "meta"
    info = _read_json(meta / "info.json")
    modality = _read_json(meta / "modality.json")
    episodes = {e["episode_index"]: e for e in _read_jsonl(meta / "episodes.jsonl")}
    tasks = {t["task_index"]: t["task"] for t in _read_jsonl(meta / "tasks.jsonl")}
    if episode not in episodes:
        raise SystemExit(f"episode {episode} not found in {meta / 'episodes.jsonl'}")
    ep = episodes[episode]

    slices = {}
    for key in JOINT_KEYS:
        m = modality["state"][key]
        slices[key] = (int(m["start"]), int(m["end"]))

    import pyarrow.parquet as pq

    chunk_id = episode // int(info.get("chunks_size", 1000))
    parquet_path = root / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(parquet_path, columns=["observation.state", "action"])
    state = np.asarray(np.stack(table.column("observation.state").to_pylist()), dtype=np.float64)
    action = np.asarray(np.stack(table.column("action").to_pylist()), dtype=np.float64)
    if len(state) != ep["length"]:
        raise SystemExit(f"parquet has {len(state)} rows but episode length is {ep['length']}")

    features = info.get("features", {})
    candidate_keys = list(info.get("video_keys") or [])
    candidate_keys += [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    video_keys = {}
    for vk in candidate_keys:
        for cam in CAM_KEYS:
            if vk.endswith("." + cam):
                video_keys[cam] = vk
    missing = [c for c in CAM_KEYS if c not in video_keys]
    if missing:
        raise SystemExit(f"video keys missing for {missing}; info.json has {sorted(features)}")

    fps = None
    for vk in video_keys.values():
        f = features.get(vk, {}).get("fps")
        if f is not None:
            fps = f
            break

    return {
        "slices": slices,
        "state": state,
        "action": action,
        "length": len(state),
        "prompt": tasks.get(ep["task_index"], (ep.get("tasks") or [""])[0]),
        "joint_names": _flatten_names(features.get("observation.state", {}).get("names"))
        or _flatten_names(features.get("action", {}).get("names")),
        "chunk_id": chunk_id,
        "video_keys": video_keys,
        "root": root,
        "parquet": str(parquet_path),
        "fps": fps,
    }


def load_segment_frames(data, start, end):
    """Decode frames start..end-1 from every camera (lockstep), native res uint8."""
    import av

    paths = {
        cam: data["root"] / "videos" / f"chunk-{data['chunk_id']:03d}" / data["video_keys"][cam]
            / f"episode_{data['episode']:06d}.mp4"
        for cam in CAM_KEYS
    }
    for cam, p in paths.items():
        if not p.exists():
            raise SystemExit(f"missing video file: {p}")
    S = end - start
    out = {cam: np.zeros((S, 480, 640, 3), dtype=np.uint8) for cam in CAM_KEYS}
    containers = [av.open(str(p)) for p in paths.values()]
    try:
        iters = {cam: c.decode(c.streams.video[0]) for cam, c in zip(CAM_KEYS, containers)}
        counts = {cam: 0 for cam in CAM_KEYS}
        for i in range(start, end):
            for cam in CAM_KEYS:
                while counts[cam] < i:
                    next(iters[cam])
                    counts[cam] += 1
                fr = next(iters[cam])
                counts[cam] += 1
                arr = fr.to_ndarray(format="rgb24")
                h, w = arr.shape[:2]
                out[cam][i - start, :h, :w] = arr
            print(f"  segment decoded frame {i - start}/{S}")
    finally:
        for c in containers:
            c.close()
    return out


def center_crop_095(img):
    h, w = img.shape[:2]
    ch, cw = int(h * 0.95), int(w * 0.95)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def resize_np(img, w, h):
    from PIL import Image
    return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float64)


def squished_view(frame):
    return resize_np(center_crop_095(frame), VIEW_W, VIEW_H)


def pillarbox_view(frame):
    return resize_np(center_crop_095(frame), PB_W, VIEW_H)


def build_grid(frames4, layout):
    """2x2 grid in CAM_KEYS order [TL, TR, BL, BR].
    layout: "nobar"     training: squished 320x176 views, 640x352 grid
            "letterbox" pre-fix server: squished views + 32px T/B bars, 640x480
            "pillarbox" A/B: aspect-preserving 235x176 views + 42px L/R bars, 640x352
    bools accepted for back-compat: True=letterbox, False=nobar."""
    if isinstance(layout, bool):
        layout = "letterbox" if layout else "nobar"
    if layout == "letterbox":
        qh, off_y, off_x, cw, view = VIEW_H + 2 * BAR, BAR, 0, VIEW_W, squished_view
    elif layout == "pillarbox":
        qh, off_y, off_x, cw, view = VIEW_H, 0, PB_BAR, PB_W, pillarbox_view
    else:
        qh, off_y, off_x, cw, view = VIEW_H, 0, 0, VIEW_W, squished_view
    g = np.zeros((2 * qh, 2 * VIEW_W, 3), dtype=np.float64)
    quads = [(0, 0), (0, VIEW_W), (qh, 0), (qh, VIEW_W)]
    for cam, (y0, x0) in zip(CAM_KEYS, quads):
        y = y0 + off_y
        g[y:y + VIEW_H, x0 + off_x:x0 + off_x + cw] = view(frames4[cam])
    return g


def detect_dark_spans(grid, thresh=10.0):
    def spans(mask):
        out, s = [], None
        for i, m in enumerate(mask):
            if m and s is None:
                s = i
            elif not m and s is not None:
                out.append((s, i - 1))
                s = None
        if s is not None:
            out.append((s, len(mask) - 1))
        return out
    rows = spans(grid.mean(axis=(1, 2)) < thresh)
    cols = spans(grid.mean(axis=(0, 2)) < thresh)
    return rows, cols


def probe_server_grid(rid, outdir):
    info = {"source": SERVER_DEBUG_GRID, "found": False}
    src = Path(SERVER_DEBUG_GRID)
    if src.exists():
        from PIL import Image
        arr = np.asarray(Image.open(src), dtype=np.float64)
        h, w = arr.shape[:2]
        rows, cols = detect_dark_spans(arr)
        info.update(
            found=True, width=w, height=h, dark_row_spans=rows, dark_col_spans=cols,
            mtime_utc=datetime.fromtimestamp(src.stat().st_mtime, tz=timezone.utc).isoformat(),
            tokens_per_frame=(h // 8 // 2) * (w // 8 // 2),
        )
        dst = outdir / "server_grid.jpg"
        shutil.copyfile(src, dst)
        info["copied_to"] = str(dst)
    info["training_geometry"] = {
        "per_view": f"{VIEW_W}x{VIEW_H} (squished 4:3, no bars)",
        "grid": f"{2 * VIEW_W}x{2 * VIEW_H}",
        "tokens_per_frame": 880,
        "note": "frame_seqlen=880 in conf.yaml implies 640x352 grid -> VAE latent 44x80",
    }
    return info


def compute_metrics(anchors, preds, gt, state, data):
    slices = data["slices"]
    names = data["joint_names"] or [f"j{i}" for i in range(preds.shape[2])]
    base = np.stack([state[int(a)] for a in anchors])
    per_chunk = []
    for j, a in enumerate(anchors):
        d = preds[j] - gt[j]
        per_chunk.append({
            "chunk": j, "anchor": int(a),
            "mae": float(np.abs(d).mean()), "mse": float((d * d).mean()), "max_ae": float(np.abs(d).max()),
            "bias": float(d.mean()),
            "groups": {k: float(np.abs(preds[j, :, slices[k][0]:slices[k][1]] - gt[j, :, slices[k][0]:slices[k][1]]).mean()) for k in JOINT_KEYS},
            "gt_motion": float(np.abs(gt[j] - base[j][None, :]).mean()),
            "pred_motion": float(np.abs(preds[j] - base[j][None, :]).mean()),
        })
    per_joint = []
    for j in range(preds.shape[2]):
        d = preds[:, :, j] - gt[:, :, j]
        c = float(np.corrcoef(gt[:, :, j].ravel(), preds[:, :, j].ravel())[0, 1])
        gstd = float(gt[:, :, j].std())
        pstd = float(preds[:, :, j].std())
        per_joint.append({
            "joint": names[j], "mae": float(np.abs(d).mean()), "mse": float((d * d).mean()),
            "bias": float(d.mean()),
            "gt_mean": float(gt[:, :, j].mean()), "pred_mean": float(preds[:, :, j].mean()),
            "gt_std": gstd, "pred_std": pstd,
            "std_ratio": (pstd / gstd) if gstd > 1e-9 else None,
            "corr": c if np.isfinite(c) else None,
            "max_abs_gt": float(np.abs(gt[:, :, j]).max()),
            "max_abs_pred": float(np.abs(preds[:, :, j]).max()),
        })
    # endpoint accuracy: last predicted frame of chunk j vs ACTUAL state at a+24
    end_err = []
    for j, a in enumerate(anchors[:-1]):
        end_err.append(np.abs(preds[j, -1] - state[int(a) + CHUNK]))
    endpoint_per_joint = (
        [float(x) for x in np.mean(np.stack(end_err), axis=0)] if end_err else None
    )
    # relative-action space: model was trained on (action - state[chunk_start])
    gt_rel = gt - base[:, None, :]
    pr_rel = preds - base[:, None, :]
    rel_joint = []
    for j in range(preds.shape[2]):
        d = pr_rel[:, :, j] - gt_rel[:, :, j]
        c = float(np.corrcoef(gt_rel[:, :, j].ravel(), pr_rel[:, :, j].ravel())[0, 1])
        rel_joint.append({"joint": names[j], "mae": float(np.abs(d).mean()),
                          "bias": float(d.mean()), "corr": c if np.isfinite(c) else None})
    alignment = {}
    for s in range(-2, 3):
        errs = []
        for j in range(len(anchors)):
            if s == 0:
                errs.append(np.abs(preds[j] - gt[j]))
            elif s > 0:
                errs.append(np.abs(preds[j, :CHUNK - s] - gt[j, s:]))
            else:
                errs.append(np.abs(preds[j, -s:] - gt[j, :CHUNK + s]))
        alignment[s] = float(np.mean(np.concatenate([x.ravel() for x in errs])))
    model_mae = float(np.abs(preds - gt).mean())
    base_err = float(np.abs(gt - base[:, None, :]).mean())
    return {
        "per_chunk": per_chunk,
        "per_joint": per_joint,
        "chunk_joint_mae": np.abs(preds - gt).mean(axis=1).tolist(),
        "alignment_scan": alignment,
        "best_shift": min(alignment, key=alignment.get),
        "model_mae": model_mae,
        "no_motion_baseline": base_err,
        "model_vs_baseline_ratio": model_mae / base_err if base_err > 0 else None,
        "corr_all": float(np.corrcoef(gt.ravel(), preds.ravel())[0, 1]),
        "endpoint_mae": (float(np.mean(np.stack(end_err))) if end_err else None),
        "endpoint_mae_per_joint": endpoint_per_joint,
        "relative_space": {
            "note": "preds and gt both expressed as (value - anchor state); this is the training action regime",
            "mae": float(np.abs(pr_rel - gt_rel).mean()),
            "bias": float((pr_rel - gt_rel).mean()),
            "corr": float(np.corrcoef(gt_rel.ravel(), pr_rel.ravel())[0, 1]),
            "per_joint": rel_joint,
        },
    }


def build_verdict(m, grid_info, video_info):
    v = {}
    if grid_info.get("found"):
        t = grid_info["tokens_per_frame"]
        row_bars = bool(grid_info["dark_row_spans"])
        col_bars = bool(grid_info["dark_col_spans"])
        if t == 880 and not row_bars and not col_bars:
            v["input_grid"] = (
                f"OK - matches training: {grid_info['width']}x{grid_info['height']} grid, "
                f"{t} tokens/frame, no bars (squished 320x176 views)")
        elif t == 880 and col_bars and not row_bars:
            v["input_grid"] = (
                f"PILLARBOX A/B: {grid_info['width']}x{grid_info['height']} grid, {t} tokens/frame, "
                f"side bars (aspect-preserving 235x176 views; NOT the training layout)")
        else:
            v["input_grid"] = (
                f"OTHER: {grid_info['width']}x{grid_info['height']} grid, {t} tokens/frame "
                f"(training: 880, no bars); row bars: {row_bars}, col bars: {col_bars}")
    else:
        v["input_grid"] = "unknown (server debug grid image not found)"
    ratio = m["model_vs_baseline_ratio"]
    v["abs_error_vs_baseline"] = (
        f"MAE {m['model_mae']:.4f} vs no-motion baseline {m['no_motion_baseline']:.4f} "
        f"(ratio {ratio:.2f}; <1 would beat 'stay at anchor' in absolute error)")
    c = m["corr_all"]
    v["tracking"] = (
        f"corr(pred,gt) = {c:.3f} -> "
        + ("strong: motion is tracked" if c > 0.9 else "moderate" if c > 0.7 else "weak: output not following GT"))
    rs = m["relative_space"]
    rel_ratio = rs["mae"] / m["model_mae"] if m["model_mae"] > 1e-9 else None
    v["relative_vs_absolute"] = (
        f"relative-space MAE {rs['mae']:.4f} (corr {rs['corr']:.3f}) vs absolute MAE {m['model_mae']:.4f}; "
        + ("error shrinks in relative space -> anchoring/offset problem"
           if rel_ratio is not None and rel_ratio < 0.8
           else "error persists in relative space -> the motion prediction itself is off"))
    if m["endpoint_mae"] is not None:
        v["endpoint"] = f"|pred[last frame] - actual state at a+{CHUNK}| = {m['endpoint_mae']:.4f} rad"
    worst = sorted(m["per_joint"], key=lambda r: -r["mae"])[:5]
    v["worst_joints"] = ", ".join(f"{r['joint']} ({r['mae']:.4f})" for r in worst)
    if video_info:
        v["video_layout"] = (
            f"model video {video_info['model_video_size'][0]}x{video_info['model_video_size'][1]} (HxW) -> "
            f"best match: {video_info['layout_best']} "
            f"(MAD letterbox {video_info['mad_mean_letterbox']:.1f} / no-bar {video_info['mad_mean_nobar']:.1f} / "
            f"pillarbox {video_info.get('mad_mean_pillarbox', float('nan')):.1f})")
    return v


def print_report(rid, args, metadata, data, anchors, latencies, m, grid_info, video_info, verdict):
    line = "=" * 100
    print(f"\n{line}")
    print(f" DreamZero GT replay  run={rid}")
    print(f" server  : tcp://{args.host}:{args.port}   model={metadata.get('model_path')}   embodiment={metadata.get('embodiment')}")
    print(f" dataset : {args.dataset}   episode={args.episode} (len {data['length']})")
    print(f" prompt  : {data['prompt']!r}")
    print(f" protocol: per call send frame[a] + state[a] + prompt (a = anchor) -> expect absolute action[a:a+{CHUNK}]")
    print(f" server view layout (per --server-layout): {args.server_layout}")
    print(f" anchors : {anchors}")
    print(line)

    print("\n[1] per-chunk action error (pred vs GT, rad)")
    print(f"{'chunk':>5} {'anchor':>6} {'lat(s)':>7} | {'MAE':>8} {'MSE':>8} {'maxAE':>8} | "
          + " ".join(f"{SHORT[k]:>7}" for k in JOINT_KEYS) + f" | {'gt_mot':>7} {'prd_mot':>7}")
    for c in m["per_chunk"]:
        print(f"{c['chunk']:>5} {c['anchor']:>6} {latencies[c['chunk']]:>7.2f} | "
              f"{c['mae']:8.5f} {c['mse']:8.5f} {c['max_ae']:8.5f} | "
              + " ".join(f"{c['groups'][k]:7.4f}" for k in JOINT_KEYS)
              + f" | {c['gt_motion']:7.4f} {c['pred_motion']:7.4f}")

    print("\n[2] per-joint error (over all chunks/timesteps)")
    print(f"{'joint':<16} {'MAE':>8} {'bias':>8} {'gt_mean':>9} {'pred_mean':>9} {'gt_std':>7} {'pr_std':>7} {'std*':>6} {'corr':>6}")
    for r in m["per_joint"]:
        sr = f"{r['std_ratio']:6.2f}" if r["std_ratio"] is not None else "   n/a"
        cr = f"{r['corr']:6.3f}" if r["corr"] is not None else "   n/a"
        print(f"{r['joint']:<16} {r['mae']:8.4f} {r['bias']:8.4f} {r['gt_mean']:9.4f} {r['pred_mean']:9.4f} "
              f"{r['gt_std']:7.4f} {r['pred_std']:7.4f} {sr} {cr}")
    print("  bias = mean(pred-gt, signed)   std* = pred_std/gt_std (>1 = model overshoots motion amplitude)")

    rs = m["relative_space"]
    print("\n[2b] relative-action space (value - anchor state; the training action regime)")
    print(f"  MAE {rs['mae']:.5f}   bias {rs['bias']:+.5f}   corr {rs['corr']:.4f}   "
          f"(absolute space: MAE {m['model_mae']:.5f}, corr {m['corr_all']:.4f})")
    worst_rel = sorted(rs["per_joint"], key=lambda r: -r["mae"])[:5]
    print("  worst joints: " + ", ".join(f"{r['joint']} {r['mae']:.4f}" for r in worst_rel))

    print("\n[3] overall")
    print(f"  MAE model {m['model_mae']:.5f} | no-motion baseline {m['no_motion_baseline']:.5f} | "
          f"ratio {m['model_vs_baseline_ratio']:.3f} | corr(pred,gt) {m['corr_all']:.4f}")
    if m["endpoint_mae"] is not None:
        print(f"  endpoint error |pred[last] - state[a+{CHUNK}]| = {m['endpoint_mae']:.5f} rad "
              f"(mean over {len(anchors) - 1} chunks; 0 = predicted trajectory lands on the real future state)")
    lat = np.asarray(latencies)
    print(f"  latency: mean {lat.mean():.2f}s | min {lat.min():.2f}s | max {lat.max():.2f}s | "
          f"throughput {CHUNK / lat.mean():.1f} pred-steps/s")

    print("\n[4] alignment scan (pred[t] vs gt[t+s]) - a clear dip = off-by-s")
    for s in sorted(m["alignment_scan"]):
        mark = "  <== best" if s == m["best_shift"] else ""
        print(f"  shift {s:+d}: MAE {m['alignment_scan'][s]:.5f}{mark}")

    if grid_info.get("found"):
        print(f"\n[5] server input grid (from {grid_info['source']}, mtime {grid_info['mtime_utc']})")
        print(f"  grid {grid_info['width']}x{grid_info['height']} (WxH) -> {grid_info['tokens_per_frame']} tokens/frame "
              f"(training: 640x352 -> 880 tokens/frame)")
        print(f"  dark row spans : {grid_info['dark_row_spans']}")
        print(f"  dark col spans : {grid_info['dark_col_spans']}")
        if grid_info["dark_row_spans"] and not grid_info["dark_col_spans"]:
            print("  -> letterboxed (black bars top/bottom): differs from training (no padding)")
        elif grid_info["dark_col_spans"] and not grid_info["dark_row_spans"]:
            print("  -> pillarboxed (black bars left/right): aspect-preserving A/B layout, NOT training")
        elif not grid_info["dark_row_spans"] and not grid_info["dark_col_spans"]:
            print("  -> no bars: full-bleed views (training layout if grid is 640x352)")
    else:
        print("\n[5] server input grid: not found (server may not have run _prepare_video yet)")

    if video_info:
        print("\n[6] model video vs GT video (same 4-cam grid layout)")
        print(f"  model video frames: {video_info['model_frames']}  size {video_info['model_video_size'][0]}x"
              f"{video_info['model_video_size'][1]} (HxW)  GT segment frames: {video_info['gt_frames']}")
        print(f"  layout match: {video_info['layout_best']}")
        print(f"  MAD vs GT grid: letterbox {video_info['mad_mean_letterbox']:.2f} | no-bar {video_info['mad_mean_nobar']:.2f} | "
              f"pillarbox {video_info.get('mad_mean_pillarbox', float('nan')):.2f} "
              f"(best: {video_info['mad_mean']:.2f})   mean PSNR: {video_info['psnr_mean']:.2f} dB")
        print("  video alignment scan (model frame i vs GT frame start+i+s):")
        for s in sorted(video_info["mad_by_shift"]):
            mark = "  <== best" if s == video_info["video_best_shift"] else ""
            print(f"    shift {s:+d}: MAD {video_info['mad_by_shift'][s]:.2f}{mark}")
        print(f"  per-quadrant content MAD (view region of layout '{video_info['layout_best']}'):")
        for cam, v in video_info["quadrant_mad"].items():
            print(f"    {cam:<16} {v:8.2f}")

    if verdict:
        print("\n[7] verdict (auto-derived)")
        for k in verdict:
            print(f"  {k:<24} {verdict[k]}")
    print()


def save_plots(rid, outdir, data, anchors, preds, gt, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["joint"] for r in m["per_joint"]]
    joint_mae = [r["mae"] for r in m["per_joint"]]
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.barh(np.arange(len(names)), joint_mae)
    ax.set_yticks(np.arange(len(names)), names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("mean |pred - GT| (rad)")
    ax.set_title(f"per-joint error  {rid}")
    fig.tight_layout()
    fig.savefig(outdir / "per_joint_error.png", dpi=150)
    plt.close(fig)

    biases = [r["bias"] for r in m["per_joint"]]
    fig, ax = plt.subplots(figsize=(6, 9))
    colors = ["tab:red" if abs(b) > 0.5 * ma else "tab:blue" for b, ma in zip(biases, joint_mae)]
    ax.barh(np.arange(len(names)), biases, color=colors)
    ax.set_yticks(np.arange(len(names)), names, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("signed bias mean(pred-GT), rad (red: |bias| > 0.5*MAE = systematic offset)")
    ax.set_title(f"per-joint bias  {rid}")
    fig.tight_layout()
    fig.savefig(outdir / "per_joint_bias.png", dpi=150)
    plt.close(fig)

    chunk_mae = [c["mae"] for c in m["per_chunk"]]
    chunk_gt = [c["gt_motion"] for c in m["per_chunk"]]
    chunk_pr = [c["pred_motion"] for c in m["per_chunk"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(anchors)), chunk_mae, "o-", label="MAE pred vs GT")
    ax.plot(range(len(anchors)), chunk_gt, "s--", label="GT motion |gt-state|")
    ax.plot(range(len(anchors)), chunk_pr, "^-", label="pred motion |pred-state|")
    ax.set_xlabel("chunk (anchor = start + 24*chunk)")
    ax.set_ylabel("rad")
    ax.set_title(f"per-chunk error vs motion magnitude  {rid}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "per_chunk_mae.png", dpi=150)
    plt.close(fig)

    active = sorted(range(len(names)), key=lambda j: -m["per_joint"][j]["gt_std"])[:12]
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    for axi, j in zip(axes.ravel(), active):
        t = np.arange(preds.shape[1])
        for cj in range(preds.shape[0]):
            axi.plot(t, gt[cj, :, j], color="tab:blue", alpha=0.35, lw=1)
            axi.plot(t, preds[cj, :, j], color="tab:red", alpha=0.55, lw=1)
        axi.set_title(f"{names[j]}  MAE={m['per_joint'][j]['mae']:.3f}", fontsize=8)
    fig.suptitle(f"pred (red) vs GT (blue), one line per chunk  {rid}")
    fig.tight_layout()
    fig.savefig(outdir / "pred_vs_gt_timeseries.png", dpi=130)
    plt.close(fig)


def compare_video(rid, outdir, model_video_rel, data, start, n_chunks, seg_frames, server_layout="nobar"):
    import av
    import csv as _csv

    src = Path("/workspace") / model_video_rel
    if not src.exists():
        print(f"  model video not found at {src}; skipping video compare")
        return None
    dst = outdir / "model_video.mp4"
    shutil.copyfile(src, dst)

    model_frames = []
    with av.open(str(dst)) as c:
        for fr in c.decode(c.streams.video[0]):
            model_frames.append(fr.to_ndarray(format="rgb24"))
    model_frames = np.stack(model_frames)
    S = seg_frames[CAM_KEYS[0]].shape[0]
    MH, MW = int(model_frames.shape[1]), int(model_frames.shape[2])
    n_cmp = min(len(model_frames), S)
    print(f"  model video: {len(model_frames)} frames, {MH}x{MW} (HxW)")

    def grid_series(layout):
        return [build_grid({cam: seg_frames[cam][i] for cam in CAM_KEYS}, layout) for i in range(n_cmp)]

    def fit(g):
        if g.shape[0] == MH and g.shape[1] == MW:
            return g
        return np.stack([resize_np(g[:, :, k], MW, MH) for k in range(3)], axis=-1)

    # score the model video against all three candidate GT layouts; the lower-MAD
    # one is the layout the server actually used
    LAYOUTS = ("letterbox", "nobar", "pillarbox")
    cands = {}
    for lay in LAYOUTS:
        gs = grid_series(lay)
        cands[lay] = (gs, float(np.mean([float(np.abs(model_frames[i] - fit(gs[i])).mean()) for i in range(n_cmp)])))
    layout_best = min(cands, key=lambda k: cands[k][1])
    g_act = cands[layout_best][0]
    mad_lb, mad_nb, mad_pb = cands["letterbox"][1], cands["nobar"][1], cands["pillarbox"][1]

    mad_by_shift = {}
    for s in range(-3, 4):
        vals = []
        for i in range(len(model_frames)):
            gi = i + s
            if gi < 0 or gi >= n_cmp:
                continue
            vals.append(float(np.abs(model_frames[i].astype(np.float64) - fit(g_act[gi])).mean()))
        mad_by_shift[s] = float(np.mean(vals)) if vals else None
    valid = {k: v for k, v in mad_by_shift.items() if v is not None}
    best = min(valid, key=valid.get)

    # per-quadrant content MAD, sampling the view region of the active layout
    # in model-video coordinates (scales if the model video size != native grid)
    off_y, off_x, cw, view_fn = 0, 0, VIEW_W, squished_view
    if layout_best == "letterbox":
        off_y, off_x, cw, view_fn = BAR, 0, VIEW_W, squished_view
    elif layout_best == "pillarbox":
        off_y, off_x, cw, view_fn = 0, PB_BAR, PB_W, pillarbox_view
    native_qh = VIEW_H + (2 * BAR if layout_best == "letterbox" else 0)
    native_h = 2 * native_qh
    quads = [(0, 0), (0, VIEW_W), (native_qh, 0), (native_qh, VIEW_W)]
    sc = MH / native_h
    quad_mad = {}
    for cam, (y0, x0) in zip(CAM_KEYS, quads):
        acc = []
        for i in range(0, n_cmp, max(1, n_cmp // 20)):
            yy, xx = int(round((y0 + off_y) * sc)), int(round((x0 + off_x) * sc))
            hh, ww = int(round(VIEW_H * sc)), int(round(cw * sc))
            mc = model_frames[i, yy:yy + hh, xx:xx + ww].astype(np.float64)
            gc = view_fn(seg_frames[cam][i])
            if (hh, ww) != (VIEW_H, cw):
                gc = resize_np(gc, ww, hh)
            acc.append(float(np.abs(mc - gc).mean()))
        quad_mad[cam] = float(np.mean(acc))

    psnrs = []
    for i in range(n_cmp):
        mse = float(((model_frames[i].astype(np.float64) - fit(g_act[i])) ** 2).mean())
        psnrs.append(10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf"))

    with open(outdir / "video_mad.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model_frame_idx", "gt_frame_idx", "layout", "MAD_0_255"])
        for i in range(n_cmp):
            w.writerow([i, start + i, layout_best,
                        f"{float(np.abs(model_frames[i].astype(np.float64) - fit(g_act[i])).mean()):.3f}"])

    make_montage(rid, outdir, model_frames, seg_frames, start, n_chunks, layout=server_layout)

    return {
        "model_video": str(dst),
        "model_frames": int(len(model_frames)),
        "model_video_size": [MH, MW],
        "gt_frames": int(S),
        "layout_best": layout_best,
        "mad_mean_letterbox": mad_lb,
        "mad_mean_nobar": mad_nb,
        "mad_mean_pillarbox": mad_pb,
        "mad_mean": cands[layout_best][1],
        "psnr_mean": float(np.mean(psnrs)),
        "mad_by_shift": mad_by_shift,
        "video_best_shift": best,
        "quadrant_mad": quad_mad,
    }


def make_montage(rid, outdir, model_frames, seg_frames, start, n_chunks, layout="nobar"):
    """3 rows x 5 sampled frames, all panels native 640x352 (no stretching):
    A: model input (GT frames in the server's current view layout)
    B: model video (generated frames)
    C: GT in the training layout (no-bar squish) - the layout the model was trained on"""
    from PIL import Image, ImageDraw

    sample = [min(i, len(model_frames) - 1) for i in range(0, len(model_frames), max(1, len(model_frames) // 5))]
    sample = sample[:5]
    cw, ch = 640, 352
    gut = 150
    img = Image.new("RGB", (cw * len(sample) + gut, ch * 3), (48, 48, 48))
    d = ImageDraw.Draw(img)

    def put(x, y, arr):
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if im.size != (cw, ch):
            im = im.resize((cw, ch))
        img.paste(im, (x, y))

    row_labels = [
        f"A: model input\n(server layout: {layout})",
        "B: model video\n(generated frames)",
        "C: GT, training layout\n(no-bar squish)",
    ]
    for r in range(3):
        d.text((8, r * ch + 8), row_labels[r], fill=(220, 220, 120))
    for ci, i in enumerate(sample):
        x = gut + ci * cw
        fr = {cam: seg_frames[cam][i] for cam in CAM_KEYS}
        put(x, 0, build_grid(fr, layout))
        if i < len(model_frames):
            put(x, ch, model_frames[i])
        put(x, 2 * ch, build_grid(fr, "nobar"))
    img.save(outdir / "montage.png")


def run_smoke(call):
    obs = {f"observation/{cam}": np.zeros((480, 640, 3), dtype=np.uint8) for cam in CAM_KEYS}
    for key in JOINT_KEYS:
        obs[f"observation/{key}"] = np.zeros(7, dtype=np.float64)
    obs["prompt"] = "grasp object"
    obs["endpoint"] = "infer"
    t0 = time.perf_counter()
    resp = call(obs)
    dt = time.perf_counter() - t0
    if isinstance(resp, dict) and "error" in resp:
        raise SystemExit(f"server error: {resp['error']}")
    action_keys = sorted(k for k in resp if k.startswith("action."))
    assert action_keys, f"no action.* keys in response: {sorted(resp)}"
    for k in action_keys:
        v = np.asarray(resp[k])
        assert v.ndim == 2 and v.shape[0] == CHUNK, f"{k}: expected ({CHUNK}, ...), got {v.shape}"
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
    print(f"Smoke inference OK in {dt:.2f}s")
    print("save_video:", call({"endpoint": "save_video"}))
    print("reset:", call({"endpoint": "reset"}))


def run_gt(args, call, rid, outdir, metadata):
    data = load_episode_data(args.dataset, args.episode)
    data["episode"] = args.episode
    state, action = data["state"], data["action"]
    end = args.start + CHUNK * args.num_chunks
    if end > data["length"]:
        raise SystemExit(f"start {args.start} + {args.num_chunks} chunks x {CHUNK} = {end} > episode length {data['length']}")
    prompt = args.prompt or data["prompt"]
    data["prompt"] = prompt
    print(f"episode {args.episode}: length={data['length']} prompt={prompt!r}")

    anchors = [args.start + CHUNK * j for j in range(args.num_chunks)]
    print(f"Decoding segment frames {args.start}..{end - 1} (4 cameras)...")
    seg = load_segment_frames(data, args.start, end)
    h0, w0 = seg[CAM_KEYS[0]][0].shape[:2]
    print(f"  native frame size: {h0}x{w0} (HxW)")

    from PIL import Image
    for j, a in enumerate(anchors):
        i = a - args.start
        fr = {cam: seg[cam][i] for cam in CAM_KEYS}
        grid_sent = build_grid(fr, args.server_layout)
        Image.fromarray(np.clip(grid_sent, 0, 255).astype(np.uint8)).save(
            outdir / f"anchor_{j:02d}_model_input_grid.png")
        nat = np.concatenate([
            np.concatenate([fr[CAM_KEYS[0]], fr[CAM_KEYS[1]]], axis=1),
            np.concatenate([fr[CAM_KEYS[2]], fr[CAM_KEYS[3]]], axis=1),
        ], axis=0)
        Image.fromarray(nat).save(outdir / f"anchor_{j:02d}_views.png")

    print("Reset...")
    print(" ", call({"endpoint": "reset"}))
    preds = np.zeros((args.num_chunks, CHUNK, state.shape[1]), dtype=np.float64)
    latencies = []
    for j, a in enumerate(anchors):
        i = a - args.start
        obs = {f"observation/{cam}": seg[cam][i] for cam in CAM_KEYS}
        for key in JOINT_KEYS:
            s, e = data["slices"][key]
            obs[f"observation/{key}"] = state[a, s:e].copy()
        obs["prompt"] = prompt
        obs["endpoint"] = "infer"
        t0 = time.perf_counter()
        resp = call(obs)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        if isinstance(resp, dict) and "error" in resp:
            raise SystemExit(f"chunk {j}: server error: {resp['error']}")
        for key in JOINT_KEYS:
            s, e = data["slices"][key]
            v = np.asarray(resp.get(f"action.{key}"), dtype=np.float64)
            if v.shape != (CHUNK, e - s):
                raise SystemExit(f"chunk {j}: action.{key} shape {v.shape}, expected ({CHUNK}, {e - s})")
            preds[j, :, s:e] = v
        print(f"chunk {j}: anchor={a:<6d} latency={dt:6.2f}s")

    model_video_rel = None
    if not args.no_save_video:
        rv = call({"endpoint": "save_video"})
        print("save_video:", rv)
        if isinstance(rv, dict) and rv.get("status") == "saved":
            model_video_rel = rv.get("path")
    print("reset:", call({"endpoint": "reset"}))

    gt = np.stack([action[a:a + CHUNK] for a in anchors])
    m = compute_metrics(anchors, preds, gt, state, data)
    grid_info = probe_server_grid(rid, outdir)
    video_info = None
    if not args.no_video_compare and model_video_rel:
        video_info = compare_video(rid, outdir, model_video_rel, data, args.start, args.num_chunks, seg,
                                   server_layout=args.server_layout)

    base = np.stack([state[int(a)] for a in anchors])
    npz = dict(
        preds=preds,                       # (n_chunks, 24, n_joints) absolute
        gt=gt,                             # (n_chunks, 24, n_joints) absolute GT action[a:a+24]
        err=preds - gt,                    # per-frame signed error
        anchors=np.array(anchors, dtype=np.int64),
        state_anchors=base,                # state at each anchor (chunk_start)
        state_full=state[args.start:end],  # FULL state trajectory over the window
        action_full=action[args.start:end],  # FULL GT action trajectory over the window
        joint_names=np.array(data["joint_names"]),
        group_keys=np.array(JOINT_KEYS),
        group_slices=np.array([data["slices"][k] for k in JOINT_KEYS], dtype=np.int64),
        episode=args.episode, start=args.start, prompt=np.array(prompt),
    )
    if data["fps"] is not None:
        npz["fps"] = np.array(data["fps"])
    np.savez_compressed(outdir / "preds_gt_state.npz", **npz)
    save_plots(rid, outdir, data, anchors, preds, gt, m)
    verdict = build_verdict(m, grid_info, video_info)
    metrics = {
        "run_id": rid,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "server": metadata,
        "args": vars(args),
        "protocol": "send frame[a]+state[a]+prompt per call; expect absolute action[a:a+24]",
        "anchors": anchors,
        "latencies_s": latencies,
        "latency_s": {"mean": float(np.mean(latencies)), "min": float(np.min(latencies)),
                      "max": float(np.max(latencies))},
        "dataset": {"path": args.dataset, "episode": args.episode, "length": data["length"],
                    "parquet": data["parquet"], "fps": data["fps"], "prompt": prompt},
        "action": m,
        "server_grid": grid_info,
        "video": video_info,
        "verdict": verdict,
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print_report(rid, args, metadata, data, anchors, latencies, m, grid_info, video_info, verdict)
    print(f"ARTIFACTS in {outdir}:")
    for p in sorted(outdir.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.2f} MB)")


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rid = f"replay_{ts}_ep{args.episode}_s{args.start}_n{args.num_chunks}"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rundir = outdir / rid
    rundir.mkdir(parents=True, exist_ok=True)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 120_000)
    sock.connect(f"tcp://{args.host}:{args.port}")

    def call(payload):
        sock.send(packer.pack(payload))
        return msgpack_numpy.unpackb(sock.recv())

    try:
        metadata = call({"endpoint": "metadata"})
        if isinstance(metadata, dict) and "error" in metadata:
            raise SystemExit(f"server error: {metadata['error']}")
        print(f"run id: {rid}")
        print(f"Server metadata: {metadata}")
        if args.smoke:
            run_smoke(call)
            return
        run_gt(args, call, rid, rundir, metadata)
    finally:
        sock.close(linger=0)


if __name__ == "__main__":
    main()
