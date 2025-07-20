python benchmark/calculate_benchmark_math.py \
--base_url $API_URL \
--model_dir /workdir/data/models/qwen3/Qwen3-4B \
--gen_config_settings benchmark/config_balanced.json \
--output_dir /workdir/projects/devel/llmtf_open/benchmark/balanced/Qwen3-4B-math \
--force_recalc \
--add_reasoning_tasks