pip install fasttext-langdetect immutabledict langdetect
pip install plotly
pip install evalica
pip install math_verify
pip install fasttext-langdetect
pip install hf_xet

python benchmark/calculate_benchmark_api.py \
--model_dir $MODEL_NAME_OR_PATH \
--gen_config_settings benchmark/config_balanced.json \
--output_dir $OUTPUT_DIR \
--tensor_parallel_size $TENSOR_PARALLEL_SIZE
