"""Generic ZMQ save_video / reset client for ALL DreamZero policy servers.

The save_video / reset endpoints are identical across every server script
(socket_test_g1_dex3_3view.py, socket_test_g1_sonic.py,
socket_test_g1_sonic_neck.py, ...): the client just sends
{"endpoint": "save_video"} (and optionally {"endpoint": "reset"}) as msgpack
over a ZMQ REQ socket; the server decodes its own buffered video latents.
So ONE client script serves every model / fine-tune.

Usage:
  python scripts/serve/save_video.py                    # save_video on :5551
  python scripts/serve/save_video.py --port 5552        # different port
  python scripts/serve/save_video.py --reset            # save, then reset
  python scripts/serve/save_video.py --reset-only       # only clear the buffer
  python scripts/serve/save_video.py --host 141.19.87.240   # remote server

Run from the .240 docker_env for the in-container copy:
  cd .../dreamzero_docker_env && docker compose run --rm dreamzero-debug \
      python /docker_data/save_video.py --port 5551
"""
import argparse
import functools
import msgpack
import numpy as np
import zmq


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    return obj


packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_pack_array)


def main():
    ap = argparse.ArgumentParser(
        description="Generic save_video/reset client for all DreamZero servers.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5551)
    ap.add_argument("--reset", action="store_true",
                    help="send reset after save_video")
    ap.add_argument("--reset-only", action="store_true",
                    help="send only reset (clear the buffer, no save)")
    args = ap.parse_args()

    ctx = zmq.Context()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, 180000)
    s.connect(f"tcp://{args.host}:{args.port}")

    if not args.reset_only:
        s.send(packer().pack({"endpoint": "save_video"}))
        print("save_video ->", unpackb(s.recv()))
    if args.reset or args.reset_only:
        s.send(packer().pack({"endpoint": "reset"}))
        print("reset ->", unpackb(s.recv()))

    s.close(linger=0)
    cleared = args.reset or args.reset_only
    print("DONE" + (" (buffer cleared)" if cleared else " (buffer preserved)"))


if __name__ == "__main__":
    main()
