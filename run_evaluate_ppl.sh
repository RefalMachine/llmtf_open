echo $GPUS_PER_NODE
echo $HF_HOME
echo $MODEL_DIR_EVAL
echo $MODEL_TO_EVAL_OUTPUT_DIR
echo $CONV_PATH
echo $FS_COUNT
echo $BATCH_SIZE

python run_evaluate_singlenode_multigpu.py \
--model_dir $MODEL_TO_EVAL \
--output_dir $MODEL_TO_EVAL_OUTPUT_DIR \
--conv_path $CONV_PATH \
--batch_size $BATCH_SIZE \
--max_len 8000 \
--few_shot_count $FS_COUNT \
--ppl_scoring \
--num_gpus $GPUS_PER_NODE
