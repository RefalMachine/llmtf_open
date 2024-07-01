current_node=$(hostname)

echo Current Node: $current_node
echo Head Node Name: $head_node
echo Head Node IP: $head_node_ip

pip show pytest
pip install pytest==8.0.0

echo $NNODES
echo $GPUS_PER_NODE
echo $HF_HOME

rdzv_id="512${head_node_ip: -1}"
rdzv_port="2650${head_node_ip: -1}"
echo $rdzv_id
echo $rdzv_port

model_dir=../../data/models/llama3_cluster/ruadapt_llama3_bpe_extended_part1-2_vo_1e4_bs256
torchrun --nnodes=$NNODES --nproc-per-node=$GPUS_PER_NODE --rdzv-id=$rdzv_id --rdzv-backend=c10d --rdzv-endpoint=$head_node_ip:$rdzv_port run_evaluate_multinode_multigpu.py \
--model_dir $model_dir \
--conv_path conversation_configs/non_instruct_simple.json \
--batch_size 1 \
--max_len 4000 \
--few_shot_count 5