#!/bin/bash
# DreamZero G1 Dex3 Training Script (default: G1_Dex3_AllMerged_GEAR)
#
# Views: 3 views (cam_left_high / cam_left_wrist / cam_right_wrist) at 320x176
# (full-bleed) -> 2x2 quadrant grid 640x352, 880 tokens per latent frame
# (frame_seqlen=880). num_frames=9, action_horizon=48, num_frame_per_block=2,
# num_action_per_block=48 (matches the DreamZero-AgiBot base geometry).
#
# Launch:  cd dreamzero_docker_env && ./run.sh training
#
# Env overrides (optional):
#   G1_DEX3_DATA_ROOT              dataset root (default /datasets/G1_Dex3_AllMerged_GEAR/)
#   OUTPUT_DIR                     checkpoint dir (default ./checkpoints/${RUN_NAME:-dreamzero_g1_dex3_allmerged})
#   NUM_GPUS                       processes (default 7)
#   MAX_STEPS                      stop after N steps (default 500; use 3 for a smoke test)
#   SAVE_STEPS                     checkpoint interval (default 500)
#   LEARNING_RATE                  (default 1e-5)
#   DATASET_SHARD_SAMPLING_RATE    (default 0.1)
#   RESUME_FROM_CHECKPOINT         checkpoint dir to resume from (default: empty = fresh run)
#   WAN_CKPT_DIR / TOKENIZER_DIR
export HYDRA_FULL_ERROR=1

# ============ CHANGE THESE VARIABLES ============
G1_DEX3_DATA_ROOT=${G1_DEX3_DATA_ROOT:-"/datasets/G1_Dex3_AllMerged_GEAR/"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/${RUN_NAME:-dreamzero_g1_dex3_allmerged}"}

if [ -z "${NUM_GPUS}" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-7}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

if [ ! -d "$G1_DEX3_DATA_ROOT" ]; then
    echo "ERROR: G1 Dex3 dataset not found at $G1_DEX3_DATA_ROOT"
    echo "Set G1_DEX3_DATA_ROOT to your converted DreamZero/GEAR dataset root"
    exit 1
fi

echo "G1_DEX3_DATA_ROOT=$G1_DEX3_DATA_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_GPUS=$NUM_GPUS MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS"

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/g1_dex3_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=9 \
    action_horizon=48 \
    state_horizon=1 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=48 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=${LEARNING_RATE:-1e-5} \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=${SAVE_STEPS:-500} \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=${MAX_STEPS:-500} \
    +resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-} \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    dataset_shard_sampling_rate=${DATASET_SHARD_SAMPLING_RATE:-0.1} \
    g1_dex3_data_root=$G1_DEX3_DATA_ROOT \
    modality_config_g1_dex3.video.delta_indices=[0,1,2,3,4,5,6,7,8] \
    modality_config_g1_dex3.state.delta_indices=[0] \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true