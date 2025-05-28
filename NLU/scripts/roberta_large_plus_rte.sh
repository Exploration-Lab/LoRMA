lorma_mode=$1
lorma_r=$2
do_pi=$3
seed=$4

export LOCAL_RANK=0
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./rte_roberta_large_rri/rte_${lorma_mode}_rank_${lorma_r}_rri_${do_rri}_lr_4e_m4_scale_1_2_seed_${seed}"
python -m torch.distributed.launch --nproc_per_node=$num_gpus --use_env --master_port=25910 \
../run_glue_lorma_plus.py \
--model_name_or_path roberta-large \
--task_name rte \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 8 \
--learning_rate 4e-4 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy no \
--warmup_ratio 0.06 \
--lorma_r $lorma_r \
--lorma_mode $lorma_mode \
--lorma_alpha $((lorma_r / 2)) \
--do_rri $do_rri \
--seed $seed \
--weight_decay 0.1 \
--fp16 False
