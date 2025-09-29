
set -x
ray stop
ROOT=/verl
export PYTHONPATH=$ROOT:$PYTHONPATH

# WANDB (optional)
export PROJECT_NAME=verl_train
export WANDB_API_KEY=
export WANDB_PROJECT="critique_grpo_math_4k_qwen3_8b_rollout7_critique_1_simple_gt_online"
export WANDB_NAME="critique_grpo_math_4k_qwen3_8b_rollout7_critique_1_simple_gt_online"
export WANDB_OFFICIAL=1
wandb online

export RAY_BACKEND_LOG_LEVEL=debug
export HYDRA_FULL_ERROR=1
export DATA_DIR=/mnt/workspace/xyzhang/evaluation/batch_eval_results/tokens8192/verl_grpo_openr1_math_4k_qwen3_8b_rollout8_0523/simple_gt_critique
export MODEL_PATH=/mnt/workspace/huggingface_models/Qwen3-8B
export RUN_NAME=critique_grpo_math_4k_qwen3_8b_rollout7_critique_1_simple_gt_online
export HDFS_CHECKPOINT_PATH=/mnt/workspace/xyzhang/rl_tuned_ckpts/luffy
TRAIN_BATCH_SIZE=128
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=6144
LEARNING_RATE=1e-6
PPO_MINI_BATCH_SIZE=64
# per GPU
PPO_MICRO_BATCH_SIZE=64
CLIP_RATIO=0.2
KL_LOSS_COEF=0.000
ENTROPY_COEFFIENT=0.001

KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=4
ROLLOUT_N=8
KL_COEF=0.00
TOTAL_EPOCHS=30
SAVE_FREQ=300
TEST_FREQ=50
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=4
REMOVE_PREVIOUS_CKPT=False
LOG_FILE_PATH=critique_grpo_math_4k_qwen3_8b_rollout7_critique_1_simple_gt_online.log
CRITIQUE_TYPE="simple_gt"
echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE" 
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" 
echo "Max Response Length: $MAX_RESPONSE_LENGTH" 
echo "Learning Rate: $LEARNING_RATE" 
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" 
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" 
echo "Micro Rollout Batch Size: $MICRO_ROLLOUT_BATCH_SIZE"
echo "KL Loss Coefficient: $KL_LOSS_COEF" 
echo "KL Loss Type: $KL_LOSS_TYPE" 
echo "Temperature: $TEMPERATURE" 
echo "Rollout N: $ROLLOUT_N" 
echo "KL Coefficient: $KL_COEF" 
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Dataset Name: $DATA_DIR"
echo "Model Name: $MODEL_PATH"
echo "Remove Clip: $REMOVE_CLIP"
echo "Remove Previous Ckpt: $REMOVE_PREVIOUS_CKPT"
echo "LOG FILE PATH: $LOG_FILE_PATH"
echo "CRITIQUE TYPE: $CRITIQUE_TYPE"

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 1000)
echo -e "Training with the following parameters:\nTrain Batch Size: $TRAIN_BATCH_SIZE\nVal Batch Size: $VAL_BATCH_SIZE\nMax Prompt Length: $MAX_PROMPT_LENGTH\nMax Response Length: $MAX_RESPONSE_LENGTH\nLearning Rate: $LEARNING_RATE\nPPO Mini Batch Size: $PPO_MINI_BATCH_SIZE\nPPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE\nKL Loss Coefficient: $KL_LOSS_COEF\nKL Loss Type: $KL_LOSS_TYPE\nTemperature: $TEMPERATURE\nRollout N: $ROLLOUT_N\nKL Coefficient: $KL_COEF\nTotal Epochs: $TOTAL_EPOCHS\nDataset Name: $DATASET_NAME\nModel Name: $MODEL_NAME"


# Ensure no previous Ray address is set
# unset RAY_ADDRESS

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.critique_grpo.critique_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train_add_question.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.critique_type=$CRITIQUE_TYPE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="p_div_p_0.1" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH
