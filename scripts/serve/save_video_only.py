"""save_video ONLY (no reset) for the 3-view G1 Dex3 server, so the running
buffer is written to disk without destroying an in-flight sim session."""
import functools, msgpack, numpy as np, zmq

def _pack(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    return obj

packer = functools.partial(msgpack.Packer, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_pack)

ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.setsockopt(zmq.LINGER, 0)
s.setsockopt(zmq.RCVTIMEO, 180000)
s.connect("tcp://127.0.0.1:5551")
s.send(packer().pack({"endpoint": "save_video"}))
print("save_video ->", unpackb(s.recv()))
s.close(linger=0)
print("DONE (buffer preserved)")
