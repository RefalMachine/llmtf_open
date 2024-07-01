current_node=$(hostname)

echo Current Node: $current_node

pip show pytest
pip install pytest==8.0.0

echo $GPUS_PER_NODE

torchrun --nnodes=1 --nproc-per-node=$GPUS_PER_NODE run_evaluate_multinode_multigpu.py \
--model_dir ../../data/models/llama3_cluster/ruadapt_llama3_bpe_extended_part1-2_vo_1e4_bs256 \
--conv_path conversation_configs/non_instruct_simple.json \
--batch_size 1 \
--max_len 4000 \
--few_shot_count 5