pip install fasttext-langdetect immutabledict langdetect
pip install plotly
pip install evalica
pip install math_verify
pip install fasttext-langdetect
pip install hf_xet
python benchmark/calculate_benchmark_api.py \
--model_dir Qwen/Qwen3-1.7B \
--benchmark_config benchmark/config_balanced_test.yaml \
--output_dir /workdir/projects/devel/llmtf_open/benchmark/balanced_2026_test/Qwen3-1.7B \
--force_recalc


