"""Send ONLY a ZMQ reset to the DreamZero server (clears its video buffer)."""
import functools, msgpack, zmq
p = functools.partial(msgpack.Packer)
u = functools.partial(msgpack.unpackb)
c = zmq.Context(); s = c.socket(zmq.REQ)
s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, 60000)
s.connect("tcp://127.0.0.1:5551")
s.send(p().pack({"endpoint": "reset"}))
print("reset ->", u(s.recv()))
s.close(linger=0)
