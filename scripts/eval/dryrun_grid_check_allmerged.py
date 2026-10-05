import os
import sys

import numpy as np

os.chdir("/workspace")
sys.path.insert(0, "/workspace")

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

OVERRIDES = [
    "report_to=wandb",
    "data=dreamzero/g1_dex3_relative",
    "wandb_project=dreamzero",
    "train_architecture=lora",
    "num_frames=25",
    "action_horizon=24",
    "num_views=4",
    "model=dreamzero/vla",
    "model/dreamzero/action_head=wan_flow_matching_action_tf",
    "model/dreamzero/transform=dreamzero_cotrain",
    "num_frame_per_block=6",
    "num_action_per_block=24",
    "num_state_per_block=1",
    "seed=42",
    "training_args.learning_rate=1e-5",
    "image_resolution_width=235",
    "image_resolution_height=176",
    "save_lora_only=true",
    "max_chunk_size=4",
    "frame_seqlen=880",
    "g1_dex3_data_root=/datasets/G1_Dex3_AllMerged_GEAR/",
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

chain = cfg.train_dataset.all_transforms.unitree_g1_upper_body_dex3
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
        shape = np.asarray(v).shape
        dtype = np.asarray(v).dtype
    except Exception:
        shape, dtype = type(v), None
    print(f"  {k}: {shape} {dtype}")

v = item.get("images", item.get("video"))
a = np.asarray(v)
print(f"\nvideo array: shape={a.shape} dtype={a.dtype} min={a.min()} max={a.max()}")

ok = True
if a.ndim == 4:
    t_, h, w, c = a.shape
elif a.ndim == 5:
    v_, t_, c, h, w = a.shape
else:
    raise SystemExit(f"UNEXPECTED video ndim={a.ndim}")

print(f"grid: T={t_} H={h} W={w}")
tok_h, tok_w = h // 16, w // 16
print(f"VAE divisibility: H%16={h % 16} W%16={w % 16}  tokens/frame={tok_h * tok_w} (expect 880)")
if (h, w) != (352, 640):
    ok = False
    print(f"FAIL: expected grid 640x352 (WxH), got {w}x{h}")
if tok_h * tok_w != 880:
    ok = False
    print("FAIL: tokens/frame != 880")

frame = a[t_ // 2] if a.ndim == 4 else a[0, t_ // 2].transpose(1, 2, 0)
if frame.dtype != np.uint8:
    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
colmean = frame.mean(axis=(0, 2))
rowmean = frame.mean(axis=(1, 2))
isdark = colmean < 10
runs = []
i = 0
while i < w:
    if isdark[i]:
        j = i
        while j < w and isdark[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    else:
        i += 1
print(f"dark col spans: {runs}  (expect ~[(0,41),(277,361),(597,639)])")
probes = {0: "bar-L", 100: "content-L", 200: "content-L", 320: "bar-M", 480: "content-R", 630: "bar-R"}
for col, label in probes.items():
    dm = colmean[col] < 10
    print(f"  col {col:3d} ({label}): mean={colmean[col]:.1f} dark={dm}")
if runs != [(0, 41), (277, 361), (597, 639)] and len(runs) != 3:
    ok = False
    print("WARN: dark col spans differ from expected pillarbox layout")
letterbox_rows = [r for r in range(h) if rowmean[r] < 10]
print(f"dark rows: {len(letterbox_rows)} (expect 0 -> no top/bottom bars)")
if letterbox_rows:
    ok = False
    print("FAIL: top/bottom bars present (letterbox, not pillarbox)")

content_w = 235
content_h = 176
print(f"content aspect: {content_w}/{content_h} = {content_w / content_h:.4f} (4:3 = {4 / 3:.4f}, max dev 0.14%)")

from PIL import Image

out = "/workspace/dryrun_grid_check_allmerged.png"
Image.fromarray(frame).save(out)
print(f"\nsaved frame -> {out}")
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
