"""Convert mncai G1_Dex3_Trash_LocoManipulation (LeRobot, INTERLEAVED 43-D)
into a GEAR dataset (GROUPED layout) for the liang12121 g1_sonic DreamZero ckpt.

mncai interleaved 43-D (per README):
  0:6 L_leg  6:12 R_leg  12:15 waist  15:22 L_arm  22:29 L_hand  29:36 R_arm  36:43 R_hand

model grouped layout (inferred from ckpt stats):
  state_qpos(29) = [0:22] + [29:36]      (L_leg+R_leg+waist+L_arm+R_arm)
  state_hand(14) = [22:29] + [36:43]     (L_hand+R_hand)
  action_hand(14)= wbc[22:29] + wbc[36:43]
  action_token(64)= motion_token[0:64]
"""
import sys
import pandas as pd, numpy as np, json, glob, os, shutil
from pathlib import Path

# Usage (inside the dreamzero container on .240):
#   python3 convert_mncai_g1_sonic_to_gear.py [SRC] [DST] [CK]
# Defaults point at the .240 container paths.
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/datasets/mncai_G1_Dex3_Trash_LocoManipulation")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "/datasets/mncai_G1_Dex3_Trash_LocoManipulation_GEAR")
CK  = Path(sys.argv[3] if len(sys.argv) > 3 else "/workspace/checkpoints/liang12121-g1-sonic-0422/checkpoint-5000")

I = {  # mncai interleaved slice boundaries
    "L_leg": (0, 6), "R_leg": (6, 12), "waist": (12, 15),
    "L_arm": (15, 22), "L_hand": (22, 29), "R_arm": (29, 36), "R_hand": (36, 43),
}
def sl(v, k):
    a, b = I[k]
    return v[:, a:b]

print(f"SRC={SRC}\nDST={DST}\nCK={CK}")
assert SRC.exists(), SRC
assert CK.exists(), CK

# --- 1. create output tree ---
if DST.exists():
    print("DST exists, removing for a clean re-run")
    shutil.rmtree(DST)
(DST / "meta").mkdir(parents=True)
for f in sorted(glob.glob(str(SRC / "data" / "chunk-*"))):
    (DST / f.split(str(SRC) + "/")[-1]).mkdir(parents=True, exist_ok=True)

# --- 2. convert each parquet: add reordered columns ---
src_info = json.load(open(SRC / "meta" / "info.json"))
files = sorted(glob.glob(str(SRC / "data" / "chunk-*" / "*.parquet")))
print(f"converting {len(files)} parquet files")
total_frames = 0
for i, fp in enumerate(files):
    rel = fp.split(str(SRC) + "/")[-1]
    df = pd.read_parquet(fp)
    s = np.array(df["observation.state"].tolist(), dtype=np.float32)
    w = np.array(df["action.wbc"].tolist(), dtype=np.float32)
    t = np.array(df["action.motion_token"].tolist(), dtype=np.float32)
    # grouped reorder
    state_qpos = np.concatenate([sl(s, "L_leg"), sl(s, "R_leg"), sl(s, "waist"), sl(s, "L_arm"), sl(s, "R_arm")], axis=1)  # 29
    state_hand = np.concatenate([sl(s, "L_hand"), sl(s, "R_hand")], axis=1)  # 14
    action_hand = np.concatenate([sl(w, "L_hand"), sl(w, "R_hand")], axis=1)  # 14
    action_token = t[:, :64].copy()  # 64
    assert state_qpos.shape[1] == 29 and state_hand.shape[1] == 14
    assert action_hand.shape[1] == 14 and action_token.shape[1] == 64
    df["state_qpos"] = [list(map(float, r)) for r in state_qpos]
    df["state_hand"] = [list(map(float, r)) for r in state_hand]
    df["action_hand"] = [list(map(float, r)) for r in action_hand]
    df["action_token"] = [list(map(float, r)) for r in action_token]
    out = DST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    total_frames += len(df)
    if i % 20 == 0:
        print(f"  [{i}/{len(files)}] {rel} rows={len(df)}")
print(f"total frames converted: {total_frames}")

# --- 3. meta files ---
ck_stats = json.load(open(CK / "experiment_cfg" / "metadata.json"))["g1_sonic"]["statistics"]
def stat(col, sub, name):
    d = ck_stats[col][sub]
    return {k: [float(x) for x in d[k]] for k in ["mean", "std", "min", "max", "q01", "q99"]}

stats = {
    "state_qpos": stat("state", "qpos", "qpos"),          # 29
    "state_hand": stat("state", "hand_joints", "hand"),   # 14
    "action_hand": stat("action", "hand_joints", "hand"), # 14
    "action_token": stat("action", "token", "token"),     # 64
}
# sanity: dims
for k, v in stats.items():
    print(f"  stats {k}: dim={len(v['mean'])}")

modality = {
    "state": {
        "qpos": {"original_key": "state_qpos", "start": 0, "end": 29, "rotation_type": None, "absolute": True, "dtype": "float32"},
        "hand_joints": {"original_key": "state_hand", "start": 0, "end": 14, "rotation_type": None, "absolute": True, "dtype": "float32"},
    },
    "action": {
        "hand_joints": {"original_key": "action_hand", "start": 0, "end": 14, "rotation_type": None, "absolute": True, "dtype": "float32"},
        "token": {"original_key": "action_token", "start": 0, "end": 64, "rotation_type": None, "absolute": True, "dtype": "float32"},
    },
    "video": {"head": {"original_key": "observation.images.ego_view"}},
    "annotation": {"task": {"original_key": "task_index"}},
}
json.dump(modality, open(DST / "meta" / "modality.json", "w"), indent=1)
json.dump(stats, open(DST / "meta" / "stats.json", "w"), indent=1)
json.dump({"robot_type": src_info.get("robot_type", "unitree_g1_dex3"), "embodiment_tag": "g1_sonic"},
          open(DST / "meta" / "embodiment.json", "w"), indent=4)

# info.json: copy src, add new features
info = json.loads(json.dumps(src_info))  # deep copy
feat = info["features"]
feat["state_qpos"] = {"dtype": "float32", "shape": [29], "names": [["qpos"] * 29]}
feat["state_hand"] = {"dtype": "float32", "shape": [14], "names": [["hand"] * 14]}
feat["action_hand"] = {"dtype": "float32", "shape": [14], "names": [["hand"] * 14]}
feat["action_token"] = {"dtype": "float32", "shape": [64], "names": [["token"] * 64]}
info["robot_type"] = "unitree_g1_dex3"
json.dump(info, open(DST / "meta" / "info.json", "w"), indent=1)

# tasks.jsonl + episodes.jsonl: copy from src
shutil.copy(SRC / "meta" / "tasks.jsonl", DST / "meta" / "tasks.jsonl")
shutil.copy(SRC / "meta" / "episodes.jsonl", DST / "meta" / "episodes.jsonl")

# --- 4. videos: symlink ---
src_videos = SRC / "videos"
dst_videos = DST / "videos"
if dst_videos.exists() or dst_videos.is_symlink():
    if dst_videos.is_symlink():
        dst_videos.unlink()
    else:
        shutil.rmtree(dst_videos)
os.symlink(src_videos.resolve(), dst_videos)
print("symlinked videos ->", src_videos.resolve())

print("\nDONE. DST tree:")
for p in sorted(DST.rglob("*")):
    if p.is_dir():
        continue
    print(" ", str(p.relative_to(DST)))
