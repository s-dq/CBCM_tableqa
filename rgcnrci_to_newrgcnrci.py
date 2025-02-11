import jsonlines
import gzip
import json
import os


def jsonl_to_gz(jsonl_file, gz_file):
    with jsonlines.open(jsonl_file, 'r') as reader:
        with gzip.open(gz_file, 'wt') as writer:
            for obj in reader:
                json_line = json.dumps(obj)
                writer.write(json_line + '\n')


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


test_cols = read_jsonl('./IM-TQA-main/TQA_code/datasets/IM_TQA/test_cols.jsonl')
test_rows = read_jsonl('./IM-TQA-main/TQA_code/datasets/IM_TQA/test_rows.jsonl')
train_cols = read_jsonl('./IM-TQA-main/TQA_code/datasets/IM_TQA/train_cols.jsonl')
train_rows = read_jsonl('./IM-TQA-main/TQA_code/datasets/IM_TQA/train_rows.jsonl')

with open('./newdata/train_tables.json', 'r', encoding='utf-8') as file:
    train_tables = json.load(file)
with open('./newdata/test_tables.json', 'r', encoding='utf-8') as file:
    test_tables = json.load(file)

train_tables_id = []
for i in range(len(train_tables)):
    train_tables_id.append(train_tables[i]['table_id'])
test_tables_id = []
for i in range(len(test_tables)):
    test_tables_id.append(test_tables[i]['table_id'])

test_cols = test_cols + [t for t in train_cols if t['id'].split('_')[0] in test_tables_id]
test_rows = test_rows + [t for t in train_rows if t['id'].split('_')[0] in test_tables_id]
train_cols = [t for t in train_cols if t['id'].split('_')[0] not in test_tables_id]
train_rows = [t for t in train_rows if t['id'].split('_')[0] not in test_tables_id]

if not os.path.exists('./traindata/test_rgcnrci'):
    os.makedirs('./traindata/test_rgcnrci')
with open('./traindata/test_rgcnrci/train_cols.jsonl', "w", encoding="utf-8") as jsonl_file:
    for item in train_cols:
        json.dump(item, jsonl_file, ensure_ascii=False)
        jsonl_file.write('\n')
with open('./traindata/test_rgcnrci/train_rows.jsonl', "w", encoding="utf-8") as jsonl_file:
    for item in train_rows:
        json.dump(item, jsonl_file, ensure_ascii=False)
        jsonl_file.write('\n')
with open('./traindata/test_rgcnrci/test_cols.jsonl', "w", encoding="utf-8") as jsonl_file:
    for item in test_cols:
        json.dump(item, jsonl_file, ensure_ascii=False)
        jsonl_file.write('\n')
with open('./traindata/test_rgcnrci/test_rows.jsonl', "w", encoding="utf-8") as jsonl_file:
    for item in test_rows:
        json.dump(item, jsonl_file, ensure_ascii=False)
        jsonl_file.write('\n')

jsonl_to_gz('./traindata/test_rgcnrci/train_cols.jsonl', './traindata/test_rgcnrci/train_cols.jsonl.gz')
jsonl_to_gz('./traindata/test_rgcnrci/train_rows.jsonl', './traindata/test_rgcnrci/train_rows.jsonl.gz')
jsonl_to_gz('./traindata/test_rgcnrci/test_cols.jsonl', './traindata/test_rgcnrci/test_cols.jsonl.gz')
jsonl_to_gz('./traindata/test_rgcnrci/test_rows.jsonl', './traindata/test_rgcnrci/test_rows.jsonl.gz')
print('success')
