import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy

packer = msgpack_numpy.Packer()
uri = "ws://172.22.0.3:8000"

with websockets.sync.client.connect(uri, compression=None, max_size=None) as ws:
    metadata = msgpack_numpy.unpackb(ws.recv())
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
        "prompt": "grasp object",
    }
    ws.send(packer.pack(obs))
    response = ws.recv()
    response = ws.recv()
    if isinstance(response, str):
        print("Error:", response)
    else:
        actions = msgpack_numpy.unpackb(response)
        print("Actions:", actions)