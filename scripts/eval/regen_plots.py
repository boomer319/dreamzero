"""Offline regeneration of replay plots + montage (no GPU, no inference).

Reads each replay dir's preds_gt_state.npz + metrics.json, decodes the GT
segment frames from the dataset, and rewrites:
  per_joint_error.png, per_joint_bias.png, per_chunk_mae.png,
  pred_vs_gt_timeseries.png, montage.png
with self-explanatory labels, proper scaling, fully-visible text, and a
consistent view layout + a GT-ghost overlay in the montage.

Run inside the dreamzero container:
    python /workspace/regen_plots.py /docker_data/logs/replay
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image, ImageDraw

JOINT_KEYS = ("left_arm_pos", "right_arm_pos", "left_hand_pos", "right_hand_pos")
GROUP_LABEL = {"left_arm_pos": "Left arm", "right_arm_pos": "Right arm",
               "left_hand_pos": "Left hand", "right_hand_pos": "Right hand"}
# 3-view grid layout (matches test_inference_3view.py): [TL, TR, BL, BR]
CAM3 = ("cam_left_high", "cam_left_wrist", "cam_right_wrist")
CAM_LABEL = {"cam_left_high": "head", "cam_left_wrist": "L wrist", "cam_right_wrist": "R wrist"}
VIEW_W, VIEW_H = 320, 176


def short_name(n):
    return (n.replace("kLeft", "L").replace("kRight", "R")
              .replace("Shoulder", "Sh").replace("Elbow", "Elb")
              .replace("Wrist", "Wri").replace("Hand", "Hand"))


def load_npz(d):
    z = np.load(d / "preds_gt_state.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def decode_gt_frames(dataset, episode, start, end, chunk_id):
    import av
    root = Path(dataset)
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    features = info.get("features", {})
    cands = list(info.get("video_keys") or []) + \
        [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    vkeys = {}
    for vk in cands:
        for cam in CAM3:
            if vk.endswith("." + cam):
                vkeys[cam] = vk
    paths = {cam: root / "videos" / f"chunk-{chunk_id:03d}" / vkeys[cam] / f"episode_{episode:06d}.mp4"
             for cam in CAM3}
    S = end - start
    out = {cam: np.zeros((S, 480, 640, 3), dtype=np.uint8) for cam in CAM3}
    containers = [av.open(str(paths[cam])) for cam in CAM3]
    try:
        iters = {cam: c.decode(c.streams.video[0]) for cam, c in zip(CAM3, containers)}
        counts = {cam: 0 for cam in CAM3}
        for i in range(start, end):
            for cam in CAM3:
                while counts[cam] < i:
                    next(iters[cam]); counts[cam] += 1
                fr = next(iters[cam]); counts[cam] += 1
                a = fr.to_ndarray(format="rgb24")
                h, w = a.shape[:2]
                out[cam][i - start, :h, :w] = a
    finally:
        for c in containers:
            c.close()
    return out


def resize(arr, w, h):
    return np.asarray(Image.fromarray(arr.astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float64)


def squish(frame):
    h, w = frame.shape[:2]
    ch, cw = int(h * 0.95), int(w * 0.95)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return resize(frame[y0:y0 + ch, x0:x0 + cw], VIEW_W, VIEW_H)


def grid3(fr):
    """3-view 2x2 grid matching the SERVER layout (verified by quadrant
    matching): [head, R_wrist; L_wrist, blank]. 640x352."""
    g = np.zeros((2 * VIEW_H, 2 * VIEW_W, 3), dtype=np.float64)
    g[0:VIEW_H, 0:VIEW_W] = squish(fr[CAM3[0]])          # TL: head
    g[0:VIEW_H, VIEW_W:2 * VIEW_W] = squish(fr[CAM3[2]]) # TR: R wrist
    g[VIEW_H:2 * VIEW_H, 0:VIEW_W] = squish(fr[CAM3[1]]) # BL: L wrist
    return g


def plot_per_joint(rid, outdir, m, names):
    rows = m["per_joint"]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 2))
    mae = [r["mae"] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, mae, color="#d62728", alpha=0.85)
    ax.set_yticks(y, [short_name(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error, rad  (|pred - GT|, over all chunks & timesteps)")
    ax.set_title(f"Per-joint error — {rid}\n(pred = model action output, GT = dataset ground truth; lower = better)")
    ax.set_xlim(0, max(mae) * 1.25)
    for i, v in enumerate(mae):
        ax.text(v + max(mae) * 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "per_joint_error.png", dpi=140)
    plt.close(fig)


def plot_bias(rid, outdir, m, names):
    rows = m["per_joint"]
    if not any("bias" in r for r in rows):
        return
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 2))
    bias = [r.get("bias", 0.0) for r in rows]
    mae = [r.get("mae", 0.0) for r in rows]
    y = np.arange(len(rows))
    colors = ["#d62728" if abs(b) > 0.5 * ma else "#1f77b4" for b, ma in zip(bias, mae)]
    ax.barh(y, bias, color=colors, alpha=0.85)
    ax.set_yticks(y, [short_name(n) for n in names])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    lim = max(1e-6, max(abs(b) for b in bias)) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Signed bias, rad  (mean(pred - GT); red = |bias| > 0.5·MAE, a systematic offset)")
    ax.set_title(f"Per-joint bias — {rid}\n(positive = model predicts higher than GT)")
    fig.tight_layout()
    fig.savefig(outdir / "per_joint_bias.png", dpi=140)
    plt.close(fig)


def plot_chunk(rid, outdir, m, anchors):
    pc = m["per_chunk"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(pc)))
    ax.plot(x, [c["mae"] for c in pc], "o-", color="#d62728", lw=2, label="MAE (|pred - GT|, rad)")
    ax.plot(x, [c["gt_motion"] for c in pc], "s--", color="#1f77b4", lw=1.5, label="GT motion |GT - anchor| (rad)")
    ax.plot(x, [c["pred_motion"] for c in pc], "^-", color="#2ca02c", lw=1.5, label="Pred motion |pred - anchor| (rad)")
    ax.set_xticks(x, [f"c{j}\na={a}" for j, a in enumerate(anchors)])
    ax.set_ylabel("radians")
    ax.set_xlabel("Chunk index (anchor = start + 48·chunk)")
    ax.set_title(f"Per-chunk error vs. motion magnitude — {rid}\n(pred moves ~2.5x more than GT => not tracking GT)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "per_chunk_mae.png", dpi=140)
    plt.close(fig)


def plot_timeseries(rid, outdir, data, names):
    preds, gt = data["preds"], data["gt"]
    active = sorted(range(len(names)), key=lambda j: -gt[:, :, j].std())[:12]
    fig, axes = plt.subplots(4, 3, figsize=(16, 11))
    t = np.arange(preds.shape[1])
    for axi, j in zip(axes.ravel(), active):
        for cj in range(preds.shape[0]):
            axi.plot(t, gt[cj, :, j], color="#1f77b4", alpha=0.5, lw=1.2, label="GT" if cj == 0 else None)
            axi.plot(t, preds[cj, :, j], color="#d62728", alpha=0.6, lw=1.2, label="pred" if cj == 0 else None)
        axi.set_title(f"{short_name(names[j])}", fontsize=9)
        axi.set_xlabel("timestep in chunk (0..47)", fontsize=8)
        axi.set_ylabel("rad", fontsize=8)
        if j == active[0]:
            axi.legend(fontsize=8)
    fig.suptitle(f"Pred (red) vs GT (blue) per joint, one line per chunk — {rid}\n"
                 f"(each chunk is a cold-start from its anchor; GT is near-static, pred drifts => no memorization)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "pred_vs_gt_timeseries.png", dpi=130)
    plt.close(fig)


def make_montage(rid, outdir, model_frames, seg_frames, start, n_chunks):
    """3 rows x 5 sampled frames, consistent 3-view layout, clear labels,
    plus a 4th row = GT ghost (GT overlaid 50% on the model output)."""
    max_i = min(len(model_frames), seg_frames[CAM3[0]].shape[0]) - 1
    n = min(max_i + 1, 5)
    sample = [int(i) for i in np.linspace(0, max_i, n).astype(int)]
    cw, ch = 640, 352
    gut = 190
    nrows = 4
    img = Image.new("RGB", (cw * n + gut, ch * nrows + 40), (40, 40, 40))
    d = ImageDraw.Draw(img)
    d.text((10, 8), f"{rid}   (columns = 5 sampled frames; rows: A input, B model output, C GT, D GT-ghost on model)",
           fill=(255, 255, 255), )

    def put(x, y, arr, alpha=1.0, base=None):
        a = np.clip(arr, 0, 255).astype(np.uint8)
        if a.shape[0] != ch or a.shape[1] != cw:
            a = np.asarray(Image.fromarray(a).resize((cw, ch)))
        im = Image.fromarray(a)
        if base is not None and alpha < 1.0:
            b = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
            if b.size != (cw, ch):
                b = b.resize((cw, ch))
            im = Image.blend(b, im, alpha)
        img.paste(im, (x, y))

    row_titles = [
        "A. MODEL INPUT  (GT frames we feed the model)",
        "B. MODEL OUTPUT  (video the model generates)",
        "C. GROUND TRUTH  (what should be displayed)",
        "D. GT-GHOST  (GT at 50% over model output => where it should be)",
    ]
    for r in range(nrows):
        d.text((8, r * ch + 12), row_titles[r], fill=(255, 230, 90))
    for ci, i in enumerate(sample):
        x = gut + ci * cw
        fr = {cam: seg_frames[cam][i] for cam in CAM3}
        g = grid3(fr)
        put(x, 0, g)
        if i < len(model_frames):
            put(x, ch, model_frames[i])
        put(x, 2 * ch, g)
        if i < len(model_frames):
            put(x, 3 * ch, g, alpha=0.5, base=model_frames[i])
    img.save(outdir / "montage.png")


def process(d):
    data = load_npz(d)
    with open(d / "metrics.json") as f:
        m = json.load(f)
    rid = m.get("run_id", d.name)
    args = m.get("args", {})
    dataset = (m.get("dataset") or {}).get("path") or args.get("dataset")
    episode = int(data.get("episode", args.get("episode", 0)))
    start = int(data.get("start", args.get("start", 0)))
    n_chunks = data["preds"].shape[0]; CHUNK = data["preds"].shape[1]
    end = start + n_chunks * CHUNK
    chunk_id = episode // 1000
    names = [str(x) for x in data["joint_names"]]
    anchors = [int(a) for a in data["anchors"]]
    am = m.get("action") or {}
    if "per_joint" not in am:
        print(f"  [skip {d.name}: no action metrics]"); return
    # plots
    plot_per_joint(rid, d, am, names)
    plot_bias(rid, d, am, names)
    plot_chunk(rid, d, am, anchors)
    plot_timeseries(rid, d, data, names)
    # montage (needs GT frames + model video)
    if dataset and (d / "model_video.mp4").exists():
        try:
            seg = decode_gt_frames(dataset, episode, start, end, chunk_id)
            import av
            mfr = []
            with av.open(str(d / "model_video.mp4")) as c:
                for fr in c.decode(c.streams.video[0]):
                    mfr.append(fr.to_ndarray(format="rgb24"))
            mfr = np.stack(mfr)
            make_montage(rid, d, mfr, seg, start, n_chunks)
        except Exception as e:
            print(f"  [montage skipped: {e}]")
    print(f"  regenerated {d.name}")


def main():
    base = Path(sys.argv[1])
    dirs = sorted(base.glob("replay_*"))
    print(f"Found {len(dirs)} replay dirs under {base}")
    for d in dirs:
        if not (d / "preds_gt_state.npz").exists() or not (d / "metrics.json").exists():
            continue
        process(d)
    print("done")


if __name__ == "__main__":
    main()
