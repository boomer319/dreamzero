"""Manual save_video + reset for the g1_sonic server (no inference).

Run from .240:
  cd /data/docker-services/world_action_models/dreamzero/dreamzero_docker_env
  docker compose run --rm dreamzero-debug python /docker_data/save_sonic_video.py
"""
import functools
import msgpack
import numpy as np
import zmq


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    return obj


packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_pack_array)

HOST, PORT = "127.0.0.1", 5551

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.setsockopt(zmq.RCVTIMEO, 120000)
sock.connect(f"tcp://{HOST}:{PORT}")

sock.send(packer().pack({"endpoint": "save_video"}))
print("save_video:", unpackb(sock.recv()))
sock.send(packer().pack({"endpoint": "reset"}))
print("reset:", unpackb(sock.recv()))
sock.close(linger=0)
print("DONE")
