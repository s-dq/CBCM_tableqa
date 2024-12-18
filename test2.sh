mkdir -p ./result/test2/cells_ernie_base
mkdir -p ./result/test2/apply/cells

CUDA_VISIBLE_DEVICES=0 python ./IM-TQA-main/TQA_code/train_RCI_model.py \
--model_type ernie \
--do_lower_case \
--train_dir ./traindata/test2/train_cells.jsonl.gz \
--dev_dir ./traindata/test2/test_cells.jsonl.gz \
--seed 1234 \
--full_train_batch_size 64 \
--gradient_accumulation_steps 8 \
--num_train_epochs 1 \
--learning_rate 1e-5 \
--warmup_fraction 0.1 \
--train_instances 150193 \
--weight_decay 0.01 \
--max_seq_length 512 \
--output_dir ./result/test2/cells_ernie_base


CUDA_VISIBLE_DEVICES=0 python ./IM-TQA-main/TQA_code/apply_RCI_model.py \
--model_type ernie \
--model_name_or_path ./result/test2/cells_ernie_base \
--do_lower_case \
--input_dir ./traindata/test2/test_cells.jsonl.gz \
--max_seq_length 512 \
--output_dir ./result/test2/apply/cells

python compute_CI_exact_match.py "./result/test2/apply/cells"
