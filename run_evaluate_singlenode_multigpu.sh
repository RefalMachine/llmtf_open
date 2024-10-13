current_node=$(hostname)

echo Current Node: $current_node

pip show pytest
pip install pytest==8.0.0

echo $GPUS_PER_NODE

torchrun --nnodes=1 --nproc-per-node=6 run_evaluate_multinode_multigpu.py \
--model_dir ../../data/models/saiga_scored_d7_mistral \
--conv_path conversation_configs/openchat_3.5_1210.json \
--output_dir ../../data/models/saiga_scored_d7_mistral/llmtf_eval_k5 \
--batch_size 1 \
--max_len 4000 \
--few_shot_count 5

torchrun --nnodes=1 --nproc-per-node=6 run_evaluate_multinode_multigpu.py \
--model_dir ../../data/models/saiga_scored_d7_mistral_extended_darulm_20_05_24_part1-2_32000_bpe_full_lr1e4_bs256 \
--conv_path conversation_configs/openchat_3.5_1210.json \
--output_dir ../../data/models/saiga_scored_d7_mistral_extended_darulm_20_05_24_part1-2_32000_bpe_full_lr1e4_bs256/llmtf_eval_k5 \
--batch_size 1 \
--max_len 4000 \
--few_shot_count 5

torchrun --nnodes=1 --nproc-per-node=6 run_evaluate_multinode_multigpu.py \
--model_dir ../../data/models/saiga_scored_d7_mistral_darulm_20_05_24_part1-2_32000_unigram_full_lr1e4_bs256 \
--conv_path conversation_configs/openchat_3.5_1210.json \
--output_dir ../../data/models/saiga_scored_d7_mistral_darulm_20_05_24_part1-2_32000_unigram_full_lr1e4_bs256/llmtf_eval_k5 \
--batch_size 1 \
--max_len 4000 \
--few_shot_count 5

torchrun --nnodes=1 --nproc-per-node=6 run_evaluate_multinode_multigpu.py \
--model_dir ../../data/models/saiga_scored_d7_mistral_darulm_20_05_24_part1-2_32000_bpe_full_lr1e4_bs256 \
--conv_path conversation_configs/openchat_3.5_1210.json \
--output_dir ../../data/models/saiga_scored_d7_mistral_darulm_20_05_24_part1-2_32000_bpe_full_lr1e4_bs256/llmtf_eval_k5 \
--batch_size 1 \
--max_len 4000 \
--few_shot_count 5