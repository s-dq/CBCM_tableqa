from util.line_corpus import write_open, jsonl_lines
import ujson as json
import numpy as np
from collections import defaultdict
import sys
import sys

# Redirect standard output to a file
sys.stdout = open("./results/output.txt", "a")

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


def gather_predictions(input_file, intent=0):
    # for the given question, find out row ids or col ids in which the trained RCI model predicts that answer cells exist
    qid_to_related_idx = defaultdict(list)
    for line in jsonl_lines(input_file):
        jobj = json.loads(line)
        pred = jobj['predictions']
        qid, ndx_str = jobj['id'].split(':')
        # if pred[1]>pred[0], the trained RCI model think answer cells exist in thie row or col.
        # set intent = 1 if the pre-defined row id or col id starts from 1 instead of 0
        if pred[1] > pred[0]:
            qid_to_related_idx[qid].append(int(ndx_str) + intent)
    return qid_to_related_idx


def get_answer_cells(cell_ids, layout):
    # get the answer cell ids based on target row ids and target col ids, e.g., row_id = 3, col_id = 2, the answer cell locates at (3,2) in the table cell matrix.
    if len(cell_ids) == 0:
        return set([])
    answer_set = set()

    layout_all = [item for sublist in layout for item in sublist]
    for i in cell_ids:
        answer_set.add(layout_all[i])
    return answer_set


# load pred results
cell_prediction_file = sys.argv[1]

qid_to_related_cell_ids = gather_predictions(cell_prediction_file)

print("len(qid_to_related_cell_ids):", len(qid_to_related_cell_ids))
# load ground truth
test_tables = json.load(open('./newdata/test_tables.json'))
test_questions = json.load(open('./newdata/test_questions.json'))

table_id_to_test_tables = {}
for table in test_tables:
    table_id = table['table_id']
    table_id_to_test_tables[table_id] = table
q_id_to_test_questions = {}
for item in test_questions:
    q_id = item['question_id']
    q_id_to_test_questions[q_id] = item

total_question_num = len(q_id_to_test_questions)  #
total_exact_match = 0
question_num_by_table_types = defaultdict(int)
exact_match_num_by_table_types = defaultdict(int)


def get_answer_type(question, table):
    if len(question['answer_cell_list']) == 1:
        return 'single_cell'
    flag = False
    for j in range(len(table['cell_ID_matrix'])):
        if set(question['answer_cell_list']).issubset(table['cell_ID_matrix'][j]):
            flag = True
    for j in range(len(table['cell_ID_matrix'][0])):
        temp = []
        for k in range(len(table['cell_ID_matrix'])):
            temp.append(table['cell_ID_matrix'][k][j])
        if set(question['answer_cell_list']).issubset(temp):
            flag = True
    if flag:
        return 'single_line'
    else:
        return 'multi_line'


single_cell_right_num = 0
single_cell_wrong_num = 0
single_line_right_num = 0
single_line_wrong_num = 0
multi_line_right_num = 0
multi_line_wrong_num = 0

test_pred_results = []
# compute exact match
for q_id in q_id_to_test_questions:
    item = {}
    question = q_id_to_test_questions[q_id]
    item.update(question)
    gold_answer_cell_list = question['answer_cell_list']
    table_id = question['table_id']
    # question_text = question['chinese_question']
    table = table_id_to_test_tables[table_id]
    # file_name = table['file_name']

    layout = table['cell_ID_matrix']
    table_type = table['table_type']
    # For a question, if either positive row_ids or positive col_ids do not exist, i.e., we cannot find predicted answer cells,
    # then the model is thought to be failed to answer this question
    try:
        related_cell_ids = qid_to_related_cell_ids[q_id]
        pred_answer_set = get_answer_cells(related_cell_ids, layout)
    except:
        pred_answer_set = set()
    item['pred_answer_list'] = list(pred_answer_set)

    if pred_answer_set == set(gold_answer_cell_list):
        is_correct = 1
    else:
        is_correct = 0

    answer_type = get_answer_type(question, table)
    if answer_type == 'single_cell':
        single_cell_right_num += is_correct
        single_cell_wrong_num += 1 - is_correct
    if answer_type == 'single_line':
        single_line_right_num += is_correct
        single_line_wrong_num += 1 - is_correct
    if answer_type == 'multi_line':
        multi_line_right_num += is_correct
        multi_line_wrong_num += 1 - is_correct

    total_exact_match += is_correct
    question_num_by_table_types[table_type] += 1
    exact_match_num_by_table_types[table_type] += is_correct
    item['is_correct'] = is_correct
    test_pred_results.append(item)
# output overall exact match results

print("total exact match score: ", total_exact_match / total_question_num)

# output exact match results on tables of each types
index = 2
for table_type, question_num in question_num_by_table_types.items():
    print(f"exact match score on {table_type} tables:",
          exact_match_num_by_table_types[table_type] / question_num_by_table_types[table_type])
    index += 1
print(f"exact match score on single_cell tables:",
      single_cell_right_num / (single_cell_right_num + single_cell_wrong_num))
print(f"exact match score on single_line tables:",
      single_line_right_num / (single_line_right_num + single_line_wrong_num))
print(f"exact match score on multi_line tables:", multi_line_right_num / (multi_line_right_num + multi_line_wrong_num))
sys.stdout.close()
sys.stdout = sys.__stdout__
