import os
import sys

import numpy as np

os.chdir("/workspace")
sys.path.insert(0, "/workspace")

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

OVERRIDES = [
    "report_to=none",
    "data=dreamzero/g1_sonic",
    "train_architecture=full",
    "num_frames=33",
    "action_horizon=24",
    "num_views=1",
    "model=dreamzero/vla",
    "model/dreamzero/action_head=wan_flow_matching_action_tf",
    "model/dreamzero/transform=dreamzero_cotrain",
    "num_frame_per_block=2",
    "num_action_per_block=24",
    "num_state_per_block=1",
    "seed=42",
    "image_resolution_width=320",
    "image_resolution_height=176",
    "frame_seqlen=220",
    "max_chunk_size=4",
    "g1_sonic_data_root=/datasets/mncai_G1_Dex3_Trash_LocoManipulation_GEAR",
    "dit_version=./checkpoints/Wan2.1-I2V-14B-480P",
    "text_encoder_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
    "image_encoder_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "vae_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    "tokenizer_path=./checkpoints/umt5-xxl",
    "pretrained_model_path=./checkpoints/liang12121-g1-sonic-0422/checkpoint-5000",
    "++action_head_cfg.config.skip_component_loading=true",
    "++action_head_cfg.config.defer_lora_injection=true",
]

CFG_DIR = "/workspace/groot/vla/configs"

with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
    cfg = compose(config_name="conf", overrides=OVERRIDES)

chain = cfg.train_dataset.all_transforms.g1_sonic
print("=== g1_sonic transform chain ===")
for i, t in enumerate(chain.transforms):
    name = t.get("_target_", "?").split(".")[-1]
    extra = ""
    if name == "VideoResize":
        extra = f" height={t.height} width={t.width}"
    if name == "DreamTransform":
        extra = (f" max_state_dim={t.get('max_state_dim')} max_action_dim={t.get('max_action_dim')} "
                 f"action_horizon={t.get('action_horizon')}")
    if name == "StateActionTransform":
        extra = f" modes={dict(t.get('normalization_modes', {}))}"
    print(f"  chain[{i}] {name}{extra}")

print("\n=== instantiating train_dataset (g1_sonic, GEAR converted) ===")
ds = instantiate(cfg.train_dataset)
print("dataset:", type(ds).__name__)

print("fetching first sample via iter ...")
item = next(iter(ds))
print("\n=== item keys + shapes ===")
for k, v in item.items():
    try:
        arr = np.asarray(v)
        print(f"  {k:28s} shape={arr.shape} dtype={arr.dtype}")
    except Exception as e:
        print(f"  {k:28s} (non-array) {type(v).__name__}: {str(v)[:80]}")

# ---- verify the reorder: compare model-input state to raw mncai columns ----
print("\n=== reorder sanity (first sample) ===")
# item['state'] should be [qpos(29)+hand(14)] = 43, in GROUPED layout
st = np.asarray(item.get("state"), dtype=np.float64).flatten()
print("state shape:", st.shape)
print("  state[0:22] (legs+waist+L_arm) first 5:", np.round(st[0:5], 3))
print("  state[22:29] (R_arm) first 5:", np.round(st[22:27], 3))
print("  state[29:43] (L_hand+R_hand) first 14:", np.round(st[29:43], 3))
