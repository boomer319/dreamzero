import argparse
import os
import time

import numpy as np
import zmq
from openpi_client import msgpack_numpy

packer = msgpack_numpy.Packer()

parser = argparse.ArgumentParser(description="ZMQ REQ smoke client for socket_test_g1_dex3.py")
parser.add_argument("--host", default=os.environ.get("DREAMZERO_HOST", "127.0.0.1"))
parser.add_argument("--port", type=int, default=int(os.environ.get("DREAMZERO_PORT", "5550")))
parser.add_argument("--prompt", default="grasp object")
args = parser.parse_args()

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 120_000)
sock.connect(f"tcp://{args.host}:{args.port}")

sock.send(packer.pack({"endpoint": "metadata"}))
metadata = msgpack_numpy.unpackb(sock.recv())
print("Server metadata:", metadata)

# Minimal observation
obs = {
    "observation/cam_left_high": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/cam_right_high": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/cam_left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/cam_right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    "observation/left_arm_pos": np.zeros(7, dtype=np.float64),
    "observation/right_arm_pos": np.zeros(7, dtype=np.float64),
    "observation/left_hand_pos": np.zeros(7, dtype=np.float64),
    "observation/right_hand_pos": np.zeros(7, dtype=np.float64),
    "prompt": args.prompt,
    "endpoint": "infer",
}

start = time.perf_counter()
sock.send(packer.pack(obs))
response = sock.recv()
latency = time.perf_counter() - start

actions = msgpack_numpy.unpackb(response)
if isinstance(actions, dict) and "error" in actions:
    print("Error:", actions["error"])
    raise SystemExit(1)

action_keys = sorted(k for k in actions if k.startswith("action."))
assert action_keys, f"no action.* keys in response: {sorted(actions)}"
for k in action_keys:
    v = np.asarray(actions[k])
    assert v.ndim == 2 and v.shape[0] == 24, f"{k}: expected (24, ...), got {v.shape}"
    print(f"  {k}: shape={v.shape} dtype={v.dtype}")

print(f"Inference OK in {latency:.2f}s; {len(action_keys)} action key(s), horizon 24")
sock.close(linger=0)
