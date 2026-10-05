import dataclasses
import logging
import socket
import asyncio
import os
import http
import time
import traceback
import torch
import tyro
from einops import rearrange
import datetime

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
import imageio
import cv2
import numpy as np

from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Args:
    port: int = 8000
    timeout_seconds: int = 50000
    model_path: str = "./checkpoints/liang12121-g1-sonic-0422/checkpoint-5000"
    enable_dit_cache: bool = False
    index: int = 0
    max_chunk_size: int | None = None
    transport: str = "zmq"
    zmq_port: int = 5550
    skip_assert_delta_indices: bool = False


class G1Dex3Server:
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        output_dir: str | None = None,
    ) -> None:
        self._policy = groot_policy
        self._output_dir = output_dir

        self._call_count = 0
        self._is_first_call = True

        self.video_across_time = []
        self._msg_index = 0

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def _build_observation(self, obs: dict) -> dict:
        converted = {}

        # Single ego camera. The g1_sonic_neck modality expects 672x384 (WxH);
        # resize the sent frame to that before the transform.
        target_h, target_w = 384, 672
        if "observation/ego_view" in obs:
            data = obs["observation/ego_view"]
            if isinstance(data, np.ndarray):
                if data.ndim == 4:
                    data = data[0]  # take the first frame
                if data.ndim == 3:
                    h, w = data.shape[:2]
                    if (h, w) != (target_h, target_w):
                        data = cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    converted["video.head"] = data[np.newaxis, ...]
                else:
                    converted["video.head"] = data
        else:
            converted["video.head"] = np.zeros((1, target_h, target_w, 3), dtype=np.uint8)

        # 43-D state (grouped layout: qpos 29 + hand 14). The client sends the
        # full 43-D vector; we split it into the two modality keys.
        if "observation/state" in obs:
            s = obs["observation/state"]
            if isinstance(s, np.ndarray):
                if s.ndim == 1:
                    s = s.reshape(1, -1)
                if s.shape[1] >= 45:
                    converted["state.qpos"] = s[:, :29].astype(np.float64)
                    converted["state.hand_joints"] = s[:, 29:43].astype(np.float64)
                    # 45-D: real 2-D neck (arthead gimbal) -> pass through
                    converted["state.neck"] = s[:, 43:45].astype(np.float64)
                elif s.shape[1] >= 43:
                    converted["state.qpos"] = s[:, :29].astype(np.float64)
                    converted["state.hand_joints"] = s[:, 29:43].astype(np.float64)
                    # 43-D (no neck, e.g. mncai G1) -> zero-pad
                    converted["state.neck"] = np.zeros((1, 2), dtype=np.float64)
                else:
                    raise ValueError(f"g1_sonic_neck expects >=43-D state, got {s.shape[1]}")
            else:
                raise ValueError("observation/state must be an ndarray")
        else:
            converted["state.qpos"] = np.zeros((1, 29), dtype=np.float64)
            converted["state.hand_joints"] = np.zeros((1, 14), dtype=np.float64)
            converted["state.neck"] = np.zeros((1, 2), dtype=np.float64)

        # language
        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""

        return converted

    def _extract_action_chunk(self, result_batch) -> dict:
        action_chunk_dict = result_batch.act
        out = {}
        if isinstance(action_chunk_dict, Batch):
            for k in dir(action_chunk_dict):
                if k.startswith("action."):
                    val = getattr(action_chunk_dict, k)
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    out[k] = val
        elif isinstance(action_chunk_dict, dict):
            for k, val in action_chunk_dict.items():
                if k.startswith("action."):
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    out[k] = val
        return out

    def infer(self, obs: dict) -> dict:
        self._call_count += 1

        converted_obs = self._build_observation(obs)

        batch = Batch(obs=converted_obs)

        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        self.video_across_time.append(video_pred)

        action_chunk = self._extract_action_chunk(result_batch)

        if self._is_first_call:
            self._is_first_call = False

        return action_chunk

    def reset(self):
        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []
        self._msg_index = 0
        action_head = getattr(getattr(self._policy, "trained_model", None), "action_head", None)
        if action_head is not None:
            if hasattr(action_head, "language"):
                action_head.language = None
            if hasattr(action_head, "current_start_frame"):
                action_head.current_start_frame = 0

    def _save_video(self):
        if len(self.video_across_time) <= 0:
            return None
        output_path = None
        try:
            frame_list = []
            video_across_time_cat = torch.cat(self.video_across_time, dim=2)
            frames = self._policy.trained_model.action_head.vae.decode(
                video_across_time_cat,
                tiled=self._policy.trained_model.action_head.tiled,
                tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
            )
            frames = rearrange(frames, "B C T H W -> B T H W C")
            frames = frames[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            for frame in frames:
                frame_list.append(frame)

            if len(frame_list) > 0:
                sample_frame = frame_list[0]
                if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                    save_dir = self._output_dir if self._output_dir else "."
                    os.makedirs(save_dir, exist_ok=True)
                    all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                    output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}.mp4')
                    imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                    logger.info(f"Saved video to: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save video: {e}")
            return None
        return output_path


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: G1Dex3Server,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        try:
            while True:
                try:
                    start_time = time.perf_counter()
                    data = await websocket.recv()
                    recv_done = time.perf_counter()
                    obs = msgpack_numpy.unpackb(data)
                    logger.info(f"Wait Time: {recv_done - start_time:.2f}s")
                    self._policy._msg_index += 1

                    obs.pop("endpoint", None)

                    action_chunk = self._policy.infer(obs)

                    await websocket.send(packer.pack(action_chunk))

                except websockets.ConnectionClosed:
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    self._policy._save_video()
                    self._policy.reset()
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error",
                    )
                    raise
        finally:
            logger.info("Client session ended")


class ZmqPolicyServer:
    def __init__(
        self,
        policy: G1Dex3Server,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
    ) -> None:
        import zmq

        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._output_dir = output_dir
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{port}")
        self._packer = msgpack_numpy.Packer()

    def serve_forever(self):
        logger.info(f"ZMQ REP server listening on tcp://{self._host}:{self._port}")
        while True:
            try:
                start_time = time.perf_counter()
                data = self._sock.recv()
                msg = msgpack_numpy.unpackb(data)
                logger.info(f"Request received after {time.perf_counter() - start_time:.2f}s wait")

                endpoint = msg.get("endpoint", "infer") if isinstance(msg, dict) else "infer"
                msg.pop("endpoint", None)

                if endpoint == "metadata":
                    self._sock.send(self._packer.pack(self._metadata))
                    continue

                if endpoint == "save_video":
                    video_path = self._policy._save_video()
                    self._sock.send(self._packer.pack({
                        "status": "saved" if video_path else "empty",
                        "path": video_path,
                    }))
                    continue

                if endpoint == "reset":
                    self._policy.reset()
                    self._sock.send(self._packer.pack({"status": "reset"}))
                    continue

                self._policy._msg_index += 1
                action_chunk = self._policy.infer(msg)
                self._sock.send(self._packer.pack(action_chunk))

            except Exception:
                error = traceback.format_exc()
                logger.error(error)
                try:
                    self._sock.send(self._packer.pack({"error": error}))
                except Exception:
                    logger.warning("Failed to send error reply; dropping request and waiting for next")
                    continue


def init_mesh() -> DeviceMesh:
    if "RANK" not in os.environ:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    logger.info(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = "g1_sonic_neck"
    model_path = args.model_path
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
        "num_cameras": 1,
        "cameras": ["ego_view"],
        "state_dim": 45,
        "action_dim": 80,
        "num_frames": 33,
        "action_horizon": 24,
        "num_frame_per_block": 2,
        "num_action_per_block": 24,
        "action_chunk_per_call": 24,
    }

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        skip_assert_delta_indices=args.skip_assert_delta_indices,
    )

    hostname = socket.gethostname()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip_str = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip_str = socket.gethostbyname(hostname)

    if rank == 0:
        logger.info("Creating server (host: %s, ip: %s)", hostname, local_ip_str)
        replay_base = "/docker_data/logs/replay"
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(replay_base, f"g1_sonic_eval_{date_suffix}_{args.index}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        logger.info(f"Rank {rank} starting as worker for distributed inference...")

    wrapper_policy = G1Dex3Server(
        groot_policy=policy,
        output_dir=output_dir,
    )

    if rank == 0:
        if args.transport == "zmq":
            server = ZmqPolicyServer(
                policy=wrapper_policy,
                host="0.0.0.0",
                port=args.zmq_port,
                metadata=policy_metadata,
                output_dir=output_dir,
            )
        else:
            server = WebsocketPolicyServer(
                policy=wrapper_policy,
                host="0.0.0.0",
                port=args.port,
                metadata=policy_metadata,
                output_dir=output_dir,
            )
        server.serve_forever()
    else:
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                dist.broadcast(signal_tensor, src=0, group=signal_group)
                signal = signal_tensor.item()
                if signal == 1:
                    logger.info(f"Rank {rank} received shutdown signal")
                    break
                elif signal == 2:
                    continue

                import pickle
                size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
                dist.broadcast(size_tensor, src=0)
                data_size = size_tensor.item()
                data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
                dist.broadcast(data_tensor, src=0)
                obs = pickle.loads(data_tensor.cpu().numpy().tobytes())

                batch = Batch(obs=obs)
                dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
