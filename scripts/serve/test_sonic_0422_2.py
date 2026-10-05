"""Smoke test: real mncai ep0 frame + 43-D state -> 0422_2 server on :5551."""
import time
import functools
import subprocess

import cv2
import msgpack
import numpy as np
import pandas as pd
import zmq

DS = "/datasets/mncai_G1_Dex3_Trash_LocoManipulation_GEAR"
EP = 0


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    return obj


packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

# real state from GEAR parquet
parquet = f"{DS}/data/chunk-000/episode_{EP:06d}.parquet"
df = pd.read_parquet(parquet)
state = np.concatenate([
    np.asarray(df["state_qpos"].iloc[0], dtype=np.float64),
    np.asarray(df["state_hand"].iloc[0], dtype=np.float64),
])
gt_token = np.asarray(df["action_token"].iloc[0], dtype=np.float64)
gt_hand = np.asarray(df["action_hand"].iloc[0], dtype=np.float64)
print("state43 ok:", state.shape, "gt_token:", gt_token.shape)

# real first frame from the episode video (head/ego cam)
video = f"{DS}/videos/chunk-000/observation.images.ego_view/episode_{EP:06d}.mp4"
cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()
assert ok, f"failed to read {video}"
frame = cv2.resize(frame, (640, 480))
print("frame ok:", frame.shape, frame.dtype)

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.setsockopt(zmq.RCVTIMEO, 300000)
sock.connect("tcp://127.0.0.1:5551")

# metadata
t0 = time.time()
sock.send(packer().pack({"endpoint": "metadata"}))
meta = unpackb(sock.recv())
print(f"metadata ({time.time()-t0:.2f}s): {str(meta)[:300]}")

# infer
obs = {
    "observation/ego_view": frame,
    "observation/state": state,
    "prompt": "pick up the trash bag",
    "endpoint": "infer",
}
t0 = time.time()
sock.send(packer().pack(obs))
resp = unpackb(sock.recv())
dt = time.time() - t0
print(f"infer took {dt:.2f}s")
if isinstance(resp, dict) and "error" in resp:
    print("ERROR:", resp["error"][:2000])
    raise SystemExit(1)

for k, v in resp.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: shape={v.shape} dtype={v.dtype} std={v.std():.4f}")
    else:
        print(f"  {k}: {str(v)[:120]}")

if "action.token" in resp and "action.hand_joints" in resp:
    tok = np.asarray(resp["action.token"], dtype=np.float64).reshape(-1, 64)
    hand = np.asarray(resp["action.hand_joints"], dtype=np.float64).reshape(-1, 14)
    print(f"chunk: token {tok.shape[0]} steps, hand {hand.shape[0]} steps")
    cos = float(np.dot(tok[0], gt_token) / (np.linalg.norm(tok[0]) * np.linalg.norm(gt_token) + 1e-9))
    print(f"token[0] vs GT token cos_sim: {cos:+.4f}")
    print(f"hand[0] vs GT hand  diff: {np.abs(hand[0] - gt_hand).max():.4f}")
    print(f"token[0] first 8: {np.round(tok[0][:8], 3).tolist()}")
    print(f"GT    token first 8: {np.round(gt_token[:8], 3).tolist()}")

sock.send(packer().pack({"endpoint": "save_video"}))
print("save_video:", unpackb(sock.recv()))
sock.send(packer().pack({"endpoint": "reset"}))
print("reset:", unpackb(sock.recv()))
sock.close(linger=0)
print("SMOKE TEST DONE")
