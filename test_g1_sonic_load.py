import os, sys
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag

def init_mesh():
    if "RANK" not in os.environ:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return init_device_mesh(device_type="cuda", mesh_shape=(dist.get_world_size(),), mesh_dim_names=("ip",))

def main(model_path):
    mesh = init_mesh()
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.G1_SONIC,
        model_path=model_path,
        device="cuda",
        device_mesh=mesh,
        skip_assert_delta_indices=True,
    )
    print("=== MODEL LOADED SUCCESSFULLY ===", flush=True)
    ah = policy.trained_model.action_head.config
    print(f"action_dim={ah.action_dim} max_state_dim={ah.max_state_dim} num_frames={ah.num_frames} action_horizon={ah.action_horizon}", flush=True)
    print(f"train_architecture={policy.trained_model.action_head.train_architecture}", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/liang12121-g1-sonic-0422/checkpoint-5000")
