python benchmark/calculate_benchmark.py \
--model_dir Qwen/Qwen3-1.7B-Base \
--benchmark_config benchmark/config_balanced_test_foundational.yaml \
--output_dir /workdir/projects/devel/llmtf_open/benchmark/balanced_2026_foundational_test/Qwen3-1.7B-Base_local_hf \
--tensor_parallel_size 1 \
--force_recalc \
--is_foundational \
--backend hf