mkdir -p ./result/test_rgcnrci/rows_ernie_base
mkdir -p ./result/test_rgcnrci/cols_ernie_base
mkdir -p ./result/test_rgcnrci/apply/rows
mkdir -p ./result/test_rgcnrci/apply/cols

CUDA_VISIBLE_DEVICES=0 python ./IM-TQA-main/TQA_code/train_RCI_model.py \
--model_type bert-base-chinese \
--do_lower_case \
--train_dir ./traindata/test_rgcnrci/train_rows.jsonl.gz \
--dev_dir ./traindata/test_rgcnrci/test_rows.jsonl.gz \
--seed 1234 \
--full_train_batch_size 64 \
--gradient_accumulation_steps 8 \
--num_train_epochs 3 \
--learning_rate 2e-5 \
--warmup_fraction 0.1 \
--train_instances 28438 \
--weight_decay 0.01 \
--max_seq_length 512 \
--output_dir ./result/test_rgcnrci/rows_ernie_base

CUDA_VISIBLE_DEVICES=0 python ./IM-TQA-main/TQA_code/train_RCI_model.py \
--model_type bert-base-chinese \
--do_lower_case \
--train_dir ./traindata/test_rgcnrci/train_cols.jsonl.gz \
--dev_dir ./traindata/test_rgcnrci/test_cols.jsonl.gz \
--seed 5678 \
--full_train_batch_size 64 \
--gradient_accumulation_steps 8 \
--num_train_epochs 3 \
--learning_rate 2e-5 \
--warmup_fraction 0.1 \
--train_instances 19827 \
--weight_decay 0.01 \
--max_seq_length 512 \
--output_dir ./result/test_rgcnrci/cols_ernie_base


CUDA_VISIBLE_DEVICES=0 python ./IM-TQA-main/TQA_code/apply_RCI_model.py \
--model_type bert-base-chinese \
--model_name_or_path ./result/test_rgcnrci/rows_ernie_base \
--do_lower_case \
--input_dir ./traindata/test_rgcnrci/test_rows.jsonl.gz \
--max_seq_length 512 \
--output_dir ./result/test_rgcnrci/apply/rows

CUDA_VISIBLE_DEVICES=0 python ./IM-TQA-main/TQA_code/apply_RCI_model.py \
--model_type bert-base-chinese \
--model_name_or_path ./result/test_rgcnrci/cols_ernie_base \
--do_lower_case \
--input_dir ./traindata/test_rgcnrci/test_cols.jsonl.gz \
--max_seq_length 512 \
--output_dir ./result/test_rgcnrci/apply/cols

python compute_RCI_exact_match.py
