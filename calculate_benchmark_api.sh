python benchmark/calculate_benchmark_api.py \
--model_dir /workdir/data/models/ruadapt/qwen3/8b/RuadaptQwen3-8B-Hybrid \
--gen_config_settings benchmark/config_balanced.json \
--output_dir /workdir/projects/devel/llmtf_open/benchmark/balanced/RuadaptQwen3-8B-Hybrid \
--tensor_parallel_size 2 \
--force_recalc

python benchmark/calculate_benchmark_api.py \
--model_dir /workdir/data/models/qwen3/Qwen3-8B \
--gen_config_settings benchmark/config_balanced.json \
--output_dir /workdir/projects/devel/llmtf_open/benchmark/balanced/Qwen3-8B \
--tensor_parallel_size 2 \
--force_recalc

#python benchmark/calculate_benchmark_api.py \
#--model_dir /workdir/data/models/qwen3/avibe \
#--gen_config_settings benchmark/config_balanced.json \
#--output_dir /workdir/projects/devel/llmtf_open/benchmark/balanced/avibe \
#--tensor_parallel_size 2 \
#--force_recalc