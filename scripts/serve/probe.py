"""Probe the DreamZero ZMQ server: is it responsive, and on what port?"""
import functools, socket, sys, time
import msgpack, zmq

for name, mod in (("msgpack", msgpack),):
    pass

def probe(port):
    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, 20000)
    s.connect("tcp://127.0.0.1:%d" % port)
    t0 = time.time()
    try:
        s.send(msgpack.packb({"endpoint": "metadata"}))
        raw = s.recv()
        print("port %d: OK in %.2fs, %d bytes" % (port, time.time() - t0, len(raw)))
        d = msgpack.unpackb(raw, raw=False)
        print("  metadata:", {k: d.get(k) for k in
              ("embodiment", "num_cameras", "num_frames", "action_horizon",
               "num_frame_per_block", "action_chunk_per_call", "model_path")})
    except Exception as e:
        print("port %d: FAILED after %.2fs -> %s" % (port, time.time() - t0, type(e).__name__))
    finally:
        s.close(linger=0)
        ctx.term()

for p in (5551, 5550):
    probe(p)