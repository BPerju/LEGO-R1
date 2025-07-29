PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
PARENT_DIR="$( dirname "$REPO_HOME" )"
echo "PARENT_DIR: $PARENT_DIR"
# Change the data_paths and image_folders to your own data
#data_paths="${REPO_HOME}/grounding_vlm.jsonl:${REPO_HOME}/object_vlm.jsonl:${REPO_HOME}/state_vlm.jsonl" 
#image_folders="${PARENT_DIR}/ARTA_LEGO:${PARENT_DIR}:${PARENT_DIR}/ARTA_LEGO"
image_folders="${PARENT_DIR}"
data_paths="${REPO_HOME}/object_train.jsonl" 
model_path="Qwen/Qwen2.5-VL-3B-Instruct"
is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="Qwen2.5-VL-3B-Instruct-Lego-3" # TODO: change this to your own experiment name
TASK_TYPE="rec"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="8080" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm True \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 100 \
    --num_generations 2 \
    --max_completion_length 128 \
    --reward_funcs accuracy format \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed="local_scripts/zero2.json" \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 8 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true

echo "Training completed for ${EXP_NAME}"