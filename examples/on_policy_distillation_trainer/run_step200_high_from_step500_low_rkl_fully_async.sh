#!/usr/bin/env bash
set -xeuo pipefail

export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REWARD_MANAGER_MODULE="${SCRIPT_DIR}/zero_reward_manager.py"

PYTHON=${PYTHON:-/root/paddlejob/gpfsspace/liuyang_etr_prm/rl_new_verl/miniconda3_verl2/bin/python}
if [ ! -x "${PYTHON}" ]; then
    PYTHON=python3
fi

resolve_model_path() {
    local model_path="$1"
    if [ -d "${model_path}/actor_model_hf" ]; then
        printf '%s\n' "${model_path}/actor_model_hf"
    else
        printf '%s\n' "${model_path}"
    fi
}

STUDENT_MODEL=${STUDENT_MODEL:-/root/paddlejob/gpfsspace/liuyang_etr_orm/reasoning_effort/step200_high}
TEACHER_MODEL=${TEACHER_MODEL:-/root/paddlejob/gpfsspace/liuyang_etr_orm/reasoning_effort/step500_low}
STUDENT_MODEL="$(resolve_model_path "${STUDENT_MODEL}")"
TEACHER_MODEL="$(resolve_model_path "${TEACHER_MODEL}")"
TRAIN_FILE=${TRAIN_FILE:-/root/paddlejob/gpfsspace/liuyang_etr_prm/train_aime25_ifeval.parquet}
VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}

PROJECT_NAME=${PROJECT_NAME:-verl-opd-rkl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-step200_high_from_step500_low_rkl_fully_async}
CKPTS_DIR=${CKPTS_DIR:-${VERL_ROOT}/ckpt}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${VERL_ROOT}/tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-2}
N_GPUS_TEACHER=${N_GPUS_TEACHER:-2}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-$((NGPUS_PER_NODE - N_GPUS_ROLLOUT - N_GPUS_TEACHER))}

ROLLOUT_TP=${ROLLOUT_TP:-2}
TEACHER_TP=${TEACHER_TP:-2}
FSDP_SIZE=${FSDP_SIZE:-${N_GPUS_TRAINING}}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1280}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((1024 * 16))}
MAX_NUM_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}

GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-1}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-$((64 * 250))}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_OFFLOAD=${ACTOR_OFFLOAD:-False}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.80}
ACTOR_MAX_TOKEN_LEN_PER_GPU=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-${MAX_NUM_TOKENS}}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${MAX_NUM_TOKENS}}

SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:--1}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${VERL_ROOT}/rollout}
REWARD_ROLLOUT_DIR=${REWARD_ROLLOUT_DIR:-${ROLLOUT_DATA_DIR}/reward_manager}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-64}

STALENESS_THRESHOLD=${STALENESS_THRESHOLD:-0.1}
TRIGGER_PARAMETER_SYNC_STEP=${TRIGGER_PARAMETER_SYNC_STEP:-4}
PARTIAL_ROLLOUT=${PARTIAL_ROLLOUT:-True}

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "TRAIN_FILE does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi
if [ ! -f "${VAL_FILE}" ]; then
    echo "VAL_FILE does not exist: ${VAL_FILE}" >&2
    exit 1
fi
if [ ! -f "${STUDENT_MODEL}/config.json" ]; then
    echo "Student HF model config not found: ${STUDENT_MODEL}/config.json" >&2
    exit 1
fi
if [ ! -f "${TEACHER_MODEL}/config.json" ]; then
    echo "Teacher HF model config not found: ${TEACHER_MODEL}/config.json" >&2
    exit 1
fi
if [ ! -f "${REWARD_MANAGER_MODULE}" ]; then
    echo "Reward manager module not found: ${REWARD_MANAGER_MODULE}" >&2
    exit 1
fi
if ((N_GPUS_TRAINING <= 0)); then
    echo "N_GPUS_TRAINING must be positive, got ${N_GPUS_TRAINING}" >&2
    exit 1
fi
if ((N_GPUS_ROLLOUT + N_GPUS_TRAINING + N_GPUS_TEACHER > NGPUS_PER_NODE)); then
    echo "GPU split exceeds NGPUS_PER_NODE: rollout=${N_GPUS_ROLLOUT}, training=${N_GPUS_TRAINING}, teacher=${N_GPUS_TEACHER}, total=${NGPUS_PER_NODE}" >&2
    exit 1
fi
if ((N_GPUS_ROLLOUT % ROLLOUT_TP != 0)); then
    echo "N_GPUS_ROLLOUT (${N_GPUS_ROLLOUT}) must be divisible by ROLLOUT_TP (${ROLLOUT_TP})" >&2
    exit 1
fi
if ((N_GPUS_TEACHER % TEACHER_TP != 0)); then
    echo "N_GPUS_TEACHER (${N_GPUS_TEACHER}) must be divisible by TEACHER_TP (${TEACHER_TP})" >&2
    exit 1
fi

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.prompt_key=prompt
    data.truncation=error
    data.filter_overlong_prompts=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.train_batch_size=0
    data.gen_batch_size=${GEN_BATCH_SIZE}
    data.return_raw_chat=True
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
)

ROLLOUT=(
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
)

DISTILLATION=(
    distillation.enabled=True
    distillation.teacher_key=data_source
    distillation.n_gpus_per_node=${N_GPUS_TEACHER}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}
    distillation.teacher_models.teacher_model.inference.temperature=1.0
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_NUM_TOKENS}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${MAX_NUM_TOKENS}
    distillation.teacher_models.teacher_model.inference.max_num_seqs=${MAX_NUM_SEQS}
    distillation.distillation_loss.loss_mode=k1
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    critic.enable=False
    critic.strategy=fsdp2
)

REWARD=(
    reward.reward_model.enable=False
    reward.reward_manager.source=importlib
    reward.reward_manager.name=OPDVerifierRewardManager
    reward.reward_manager.module.path="${REWARD_MANAGER_MODULE}"
    reward.num_workers=${REWARD_NUM_WORKERS}
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console","tensorboard"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.val_before_train=False
    trainer.test_freq=${TEST_FREQ}
    trainer.log_val_generations=0
    trainer.save_freq=${SAVE_FREQ}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=disable
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

ASYNC_TRAINING=(
    rollout.nnodes=${NNODES}
    rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS}
    async_training.staleness_threshold=${STALENESS_THRESHOLD}
    async_training.partial_rollout=${PARTIAL_ROLLOUT}
    async_training.trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP}
    async_training.require_batches=${REQUIRE_BATCHES}
    async_training.use_trainer_do_validate=False
)

mkdir -p "${CKPTS_DIR}" "${ROLLOUT_DATA_DIR}" "${TENSORBOARD_DIR}" "${REWARD_ROLLOUT_DIR}"

cd "${VERL_ROOT}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export TENSORBOARD_DIR
export ROLLOUT_DIR="${REWARD_ROLLOUT_DIR}"

"${PYTHON}" -m verl.experimental.fully_async_policy.fully_async_main \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DISTILLATION[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC_TRAINING[@]}" \
    "$@"
