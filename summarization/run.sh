python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file ./dataset/train_csv.csv \
    --validation_file ./dataset/eval_csv.csv \
    --text_column text \
    --summary_column summary \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir checkpoint \
    --per_device_train_batch_size=100 \
    --per_device_eval_batch_size=100 \
    --overwrite_output_dir \
    --predict_with_generate \
	--num_train_epochs 2 \
	