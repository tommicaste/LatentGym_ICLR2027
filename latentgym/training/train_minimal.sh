#!/usr/bin/env bash
# Minimal single-GPU training example.
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
# Training (single GPU, GRPO, no critic)
# ============================================================
MAX_TURNS=$((MAX_TURNS_PER_EPISODE * NUM_EPISODES))

python -m skyrl_train.entrypoints.main_base \
    data.train_data="['$TRAIN_PARQUET']" \
    data.val_data="['$VAL_PARQUET']" \
    environment.env_class=benchmark_${ENV} \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.kl_loss_coef=0.1 \
    trainer.policy.model.path=$MODEL \
    trainer.policy.optimizer_config.lr=5e-6 \
    trainer.policy.fsdp_config.cpu_offload=true \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.policy_num_gpus_per_node=1 \
    trainer.placement.ref_num_gpus_per_node=1 \
    trainer.placement.critic_num_gpus_per_node=1 \
    trainer.epochs=5 \
    trainer.train_batch_size=8 \
    trainer.policy_mini_batch_size=4 \
    trainer.eval_batch_size=8 \
    trainer.eval_before_train=false \
    trainer.eval_interval=2 \
    trainer.max_prompt_length=4096 \
    trainer.logger=console \
    trainer.seed=$SEED \
    trainer.run_name="minimal-${ENV}-${LATENT}" \
    generator.num_inference_engines=1 \
    generator.inference_engine_tensor_parallel_size=1 \
    generator.n_samples_per_prompt=4 \
    generator.gpu_memory_utilization=0.5 \
    generator.sampling_params.temperature=0.7 \
    generator.sampling_params.top_p=0.95 \
    generator.sampling_params.max_generate_length=512 \
    generator.max_turns=$MAX_TURNS
