python benchmark/calculate_benchmark.py \
--model_dir /workdir/data/models/qwen3/Qwen3-4B \
--gen_config_settings benchmark/config_balanced.json \
--conv_path conversation_configs/qwen3-no-think.json \
--output_dir /workdir/projects/devel/llmtf_open/benchmark/balanced/Qwen3-4B \
--tensor_parallel_size 1 \
--force_recalc \
--add_reasoning_tasks