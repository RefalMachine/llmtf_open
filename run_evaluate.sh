echo $GPUS_PER_NODE
echo $HF_HOME
echo $MODEL_DIR_EVAL
echo $MODEL_TO_EVAL_OUTPUT_DIR
echo $CONV_PATH
echo $FS_COUNT

BATCH_SIZE_DEFAULT=${BATCH_SIZE:-2}
echo $BATCH_SIZE_DEFAULT

torchrun --nnodes=1 --nproc-per-node=$GPUS_PER_NODE run_evaluate_multinode_multigpu.py \
--model_dir $MODEL_TO_EVAL \
--output_dir $MODEL_TO_EVAL_OUTPUT_DIR \
--conv_path $CONV_PATH \
--batch_size $BATCH_SIZE_DEFAULT \
--max_len 4000 \
--few_shot_count $FS_COUNT \
--vllm
