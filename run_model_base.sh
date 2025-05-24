OUTPUT_FOLDER="/workspace/output"

# Base Args
SEQ_LEN=${1}
MODEL_NAME=${2}

# Train Args
ZERO_STAGE=${3}
TRAIN_BATCH_SIZE=${4}
GRAD_ACCU=${5}
EVAL_BATCH_SIZE=${6}
NUM_TRAIN_EPOCHS=${7}
MAIN_LEARNING_RATE=${8}

# CAM Args
MULTI_CLASS=${9}
LORA_DIM=${10}

LEARNING_RATE=${11} #learning_rate
WEIGHT_DECAY=${12}
WARMUP_RATIO=${13}


args=()

args+=("--num_train_epochs" "$NUM_TRAIN_EPOCHS")
args+=("--dtype" "bf16")

args+=("--max_seq_len" "$SEQ_LEN")
args+=("--model_name_or_path" "/workspace/models/$MODEL_NAME/")
args+=("--per_device_train_batch_size" "$TRAIN_BATCH_SIZE")
args+=("--gradient_accumulation_steps" "$GRAD_ACCU")
args+=("--per_device_eval_batch_size" "$EVAL_BATCH_SIZE")
args+=("--main_learning_rate" "$MAIN_LEARNING_RATE")
args+=("--learning_rate" "$LEARNING_RATE") #   --learning_rate 9.65e-6 \
args+=("--zero_stage" "$ZERO_STAGE")
args+=("--weight_decay" "$WEIGHT_DECAY") #      --weight_decay 0. \
args+=("--num_warmup_ratio" "$WARMUP_RATIO") #   --num_warmup_ratio "0.1" \


# Data
args+=("--data_path")
args+=("/workspace/data/Auto-CoT/Auto-CoT_traineval.json")
args+=("/workspace/data/self-iCliniq/iCliniq_traineval.json")
args+=("/workspace/data/dolly/dolly_traineval.json")
args+=("/workspace/data/CodeAlpaca/code_alpaca_traineval.json")
args+=("/workspace/data/webGPT/webGPT_traineval.json")

# Model
args+=("--gradient_checkpointing")
args+=("--use_CAM")
args+=("--lora_dim" "$LORA_DIM")
args+=("--multi_classes" "$MULTI_CLASS")

model_str="HyCAM_c${MULTI_CLASS}r${LORA_DIM}"


current_date=$(date +%d%H%M)

OUTPUT="${OUTPUT_FOLDER}/output_len${SEQ_LEN}_${MODEL_NAME/-instruct}_${model_str}_bs${TRAIN_BATCH_SIZE}ac${GRAD_ACCU}_llr${MAIN_LEARNING_RATE}_lr${LEARNING_RATE}_WD${WEIGHT_DECAY}_WM${WARMUP_RATIO}_${current_date}"

OUTPUT_LOG="${OUTPUT}/training.log"

mkdir -p $OUTPUT
echo $OUTPUT
echo $OUTPUT_LOG

deepspeed main.py \
   --data_split 10,0,0 \
   --seed 2333 \
   --lr_scheduler_type cosine \
   --deepspeed \
   --reload \
   --print_loss \
   --enable_tensorboard \
   "${args[@]}"  --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT \
   | tee -a $OUTPUT_LOG