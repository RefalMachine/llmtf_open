export GPUS_PER_NODE=6
export MODEL_TO_EVAL=/workdir/data/models/qwen3/Qwen3-32B
export MODEL_TO_EVAL_OUTPUT_DIR=outputs/Qwen3-32B-zero-shot
export CONV_PATH=conversation_configs/qwen3.json
export BATCH_SIZE=100 # because vllm
export FS_COUNT=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

./run_evaluate.sh


export GPUS_PER_NODE=6
export MODEL_TO_EVAL=/workdir/data/models/qwen3/Qwen3-32B
export MODEL_TO_EVAL_OUTPUT_DIR=outputs/Qwen3-32B-zero-shot-ppl
export CONV_PATH=conversation_configs/qwen3.json
export BATCH_SIZE=4
export FS_COUNT=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

./run_evaluate_ppl.sh

export GPUS_PER_NODE=6
export MODEL_TO_EVAL=/workdir/data/models/qwen3/Qwen3-32B
export MODEL_TO_EVAL_OUTPUT_DIR=outputs/Qwen3-32B-few-shot
export CONV_PATH=conversation_configs/qwen3.json
export BATCH_SIZE=100 # because vllm
export FS_COUNT=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

./run_evaluate.sh


export GPUS_PER_NODE=6
export MODEL_TO_EVAL=/workdir/data/models/qwen3/Qwen3-32B
export MODEL_TO_EVAL_OUTPUT_DIR=outputs/Qwen3-32B-few-shot-ppl
export CONV_PATH=conversation_configs/qwen3.json
export BATCH_SIZE=2
export FS_COUNT=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

./run_evaluate_ppl.sh