lorma_r=$1
seed=$2
lorma_mode=$3
rank_inflation=$4

export CUDA_VISIBLE_DEVICES=0
export num_gpus=1
export MODEL_PATH="meta-llama/Meta-Llama-3-8B"
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"


export SAVE_PATH="./Llama3_8B_metamath40k_lorma_${lorma_r}_${lorma_mode}_${rank_inflation}_lr_5e_m4_seed_${seed}"
python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=$num_gpus --use_env train_math_lorma.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "./data/train/MetaMathQA-40K.json" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 2 \
    --learning_rate 5e-4\
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --lorma_r $lorma_r \
    --lorma_alpha $((lorma_r * 2)) \
    --lorma_mode $lorma_mode \
    --rank_inflation $rank_inflation \
    --target_modules up_proj down_proj o_proj gate_proj \




echo "====================="
echo "====================="
echo "====================="
echo "EVAL"
echo "====================="
echo "====================="
echo "====================="


python eval_gsm8k.py --model "./Llama3_8B_metamath40k_lorma_${lorma_r}_${lorma_mode}_${rank_inflation}_lr_5e_m4_seed_${seed}" --data_file ../MetaMath/data/test/GSM8K_test.jsonl


echo "====================="
echo "====================="
echo "====================="


python eval_math.py --model "./Llama3_8B_metamath40k_lorma_${lorma_r}_${lorma_mode}_${rank_inflation}_lr_5e_m4_seed_${seed}" --data_file ../MetaMath/data/test/MATH_test.jsonl



