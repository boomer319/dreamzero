import os
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.chdir("/workspace")
sys.path.insert(0, "/workspace")

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

OVERRIDES = [
    "report_to=wandb",
    "data=dreamzero/g1_dex3_relative",
    "wandb_project=dreamzero",
    "train_architecture=lora",
    "num_frames=33",
    "action_horizon=48",
    "state_horizon=4",
    "num_views=3",
    "model=dreamzero/vla",
    "model/dreamzero/action_head=wan_flow_matching_action_tf",
    "model/dreamzero/transform=dreamzero_cotrain",
    "num_frame_per_block=2",
    "num_action_per_block=12",
    "num_state_per_block=1",
    "seed=42",
    "training_args.learning_rate=1e-5",
    "image_resolution_width=320",
    "image_resolution_height=176",
    "save_lora_only=true",
    "max_chunk_size=4",
    "frame_seqlen=880",
    "g1_dex3_data_root=/datasets/G1_Dex3_GraspSquare_Dataset_GEAR/",
    "modality_config_g1_dex3.state.delta_indices=[0,1,2,3]",
    "dit_version=./checkpoints/Wan2.1-I2V-14B-480P",
    "text_encoder_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
    "image_encoder_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "vae_pretrained_path=./checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    "tokenizer_path=./checkpoints/umt5-xxl",
    "pretrained_model_path=./checkpoints/DreamZero-AgiBot",
    "++action_head_cfg.config.skip_component_loading=true",
    "++action_head_cfg.config.defer_lora_injection=true",
]

CFG_DIR = "/workspace/groot/vla/configs"

with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
    cfg = compose(config_name="conf", overrides=OVERRIDES)

print("num_frames", cfg.num_frames)
print("action_horizon", cfg.action_horizon)
print("state_horizon", cfg.state_horizon)
print("num_views", cfg.num_views)
print("num_frame_per_block", cfg.num_frame_per_block)
print("num_action_per_block", cfg.num_action_per_block)
print("num_state_per_block", cfg.num_state_per_block)
print("image_resolution_width", cfg.image_resolution_width)
print("image_resolution_height", cfg.image_resolution_height)
print("frame_seqlen", cfg.frame_seqlen)
print("max_chunk_size", cfg.max_chunk_size)
print("max_state_dim", cfg.max_state_dim)
print("max_action_dim", cfg.max_action_dim)
print("state delta_indices", list(cfg.modality_config_g1_dex3.state.delta_indices))

chain = cfg.train_dataset.all_transforms.unitree_g1_upper_body_dex3
print("\nchain:")
for i, t in enumerate(chain.transforms):
    name = t.get("_target_", "?").split(".")[-1]
    extra = ""
    if name == "VideoResize":
        extra = f" height={t.height} width={t.width}"
    if name == "DreamTransform":
        extra = (
            f" image_resolution_width={t.get('image_resolution_width', 'MISSING')} "
            f"image_resolution_height={t.get('image_resolution_height', 'MISSING')}"
        )
    if name == "VideoCrop":
        extra = f" scale={t.get('scale')}"
    print(f"chain[{i}] {name}{extra}")

print("\ninstantiating train_dataset ...")
ds = instantiate(cfg.train_dataset)
print("dataset:", type(ds).__name__)

print("fetching first sample via iter ...")
item = next(iter(ds))
print("\nitem keys:")
for k, v in item.items():
    try:
        arr = np.asarray(v)
        print(f"  {k}: {arr.shape} {arr.dtype}")
    except Exception:
        print(f"  {k}: {type(v)} {v}")

images = np.asarray(item["images"])
print(f"\nimages: shape={images.shape} dtype={images.dtype} min={images.min()} max={images.max()}")

if images.ndim == 4:
    t_, h, w, c = images.shape
    frame = images[t_ // 2]
elif images.ndim == 5:
    v_, t_, c, h, w = images.shape
    frame = images[0, t_ // 2].transpose(1, 2, 0)
else:
    raise SystemExit(f"UNEXPECTED images ndim={images.ndim}")

frame = np.asarray(frame)
if frame.shape[0] == 3:
    frame = frame.transpose(1, 2, 0)

if frame.dtype != np.uint8:
    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

print(f"grid: T={t_} H={h} W={w}")
tok_h, tok_w = h // 16, w // 16
print(f"VAE divisibility: H%16={h % 16} W%16={w % 16} tokens/frame={tok_h * tok_w} (expect 880)")

ok = True
if (h, w) != (352, 640):
    ok = False
    print(f"FAIL: expected grid 640x352 (WxH), got {w}x{h}")
if tok_h * tok_w != 880:
    ok = False
    print("FAIL: tokens/frame != 880")

black = frame[176:, 320:]
print(f"bottom-right black quadrant mean={black.mean():.4f} max={black.max()}")
if black.mean() >= 1.0 or black.max() > 0:
    ok = False
    print("FAIL: bottom-right quadrant is not black")

top_left = frame[:176, :320]
top_right = frame[:176, 320:]
bottom_left = frame[176:, :320]
print(f"top-left mean={top_left.mean():.2f}")
print(f"top-right mean={top_right.mean():.2f}")
print(f"bottom-left mean={bottom_left.mean():.2f}")

state = np.asarray(item.get("state"))
action = np.asarray(item.get("action"))
text = item.get("text")
print(f"state: shape={state.shape} dtype={state.dtype}")
print(f"action: shape={action.shape} dtype={action.dtype}")
print(f"text: {text!r}")
if state.shape != (4, 64):
    ok = False
    print(f"FAIL: expected state (4, 64), got {state.shape}")
if action.shape != (48, 32):
    ok = False
    print(f"FAIL: expected action (48, 32), got {action.shape}")

from PIL import Image

out = "/workspace/dryrun_grid_check_graspsquare.png"
Image.fromarray(frame).save(out)
print(f"\nsaved frame -> {out}")
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
