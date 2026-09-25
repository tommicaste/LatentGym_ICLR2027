#!/usr/bin/env bash
# Multi-GPU training with FSDP (default: 4 GPUs).
# GRPO on number_guessing with Qwen2.5-1.5B-Instruct.
# Run `python -m latentgym.cli.generate_data train ...` first to create the parquets.
set -e

# ============================================================
# Config (edit these)
# ============================================================
ENV=number_guessing
LATENT=set_of_3
MODEL=Qwen/Qwen2.5-1.5B-Instruct
SEED=42
NUM_EPISODES=10
MAX_TURNS_PER_EPISODE=10

# GPUs and batch sizes
NUM_GPUS=4
TRAIN_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
MINI_BATCH_SIZE=8

# Optional W&B logging (uncomment and fill in)
# WANDB_PROJECT=LatentGym
# WANDB_GROUP=training-runs

TRAIN_PARQUET=latentgym/data/train/${ENV}/${LATENT}/parquets/full_info_standard_cumulative.parquet
VAL_PARQUET=latentgym/data/train/${ENV}/${LATENT}/val/parquets/full_info_standard.parquet

# ============================================================
# Setup
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../project_config.sh"
source "$VENV_DIR/bin/activate"

export SKYRL_REGISTER_MODULES=latentgym.register_skyrl
ray stop --force 2>/dev/null || true

# ============================================================
# Training
# Multi-GPU strategy:
#   - FSDP2 (trainer.strategy=fsdp2, default)
#   - colocate_all=true puts policy+ref on the same GPUs
#   - cpu_offload=true reduces memory pressure (params + optimizer to CPU during fwd)
#   - one vLLM inference engine per GPU (tensor_parallel_size=1)
# ============================================================
MAX_TURNS=$((MAX_TURNS_PER_EPISODE * NUM_EPISODES))

python -m skyrl_train.entrypoints.main_base \
    data.train_data="['$TRAIN_PARQUET']" \
    data.val_data="['$VAL_PARQUET']" \
    environment.env_class=benchmark_${ENV} \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=0.1 \
    trainer.policy.model.path=$MODEL \
    trainer.policy.optimizer_config.lr=5e-6 \
    trainer.policy.fsdp_config.cpu_offload=true \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.colocate_all=true \
    trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
    trainer.strategy=fsdp2 \
    trainer.epochs=10 \
    trainer.train_batch_size=$TRAIN_BATCH_SIZE \
    trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
    trainer.eval_batch_size=64 \
    trainer.eval_before_train=false \
    trainer.eval_interval=5 \
    trainer.max_prompt_length=4096 \
    trainer.logger=console \
    trainer.project_name="LatentGym" \
    trainer.run_name="fsdp-${ENV}-${LATENT}" \
    trainer.seed=$SEED \
    generator.backend=vllm \
    generator.num_inference_engines=$NUM_GPUS \
    generator.inference_engine_tensor_parallel_size=1 \
    generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
    generator.gpu_memory_utilization=0.7 \
    generator.sampling_params.temperature=0.7 \
    generator.sampling_params.top_p=0.95 \
    generator.sampling_params.max_generate_length=1024 \
    generator.max_turns=$MAX_TURNS

# To enable W&B logging, change trainer.logger to "wandb" and add:
#    +trainer.wandb_entity=${WANDB_PROJECT} \
#    +trainer.wandb_group=${WANDB_GROUP} \
