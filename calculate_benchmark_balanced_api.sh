python benchmark/calculate_benchmark_api.py \
--model_dir /workdir/data/models/T-pro-it-2.0 \
--gen_config_settings benchmark/config_balanced.json \
--output_dir /workdir/projects/devel/llmtf_open/benchmark/balanced/T-pro-it-2.0-ifeval \
--tensor_parallel_size 4 \
--force_recalc