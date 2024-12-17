import jsonlines
import gzip
import json
import os


# 保存json文件为RCI模型训练需要的.gz压缩文件
def jsonl_to_gz(jsonl_file, gz_file):
    with jsonlines.open(jsonl_file, 'r') as reader:
        with gzip.open(gz_file, 'wt') as writer:
            for obj in reader:
                json_line = json.dumps(obj)
                writer.write(json_line + '\n')


def data_to_traindata(testnum):
    # 把表格变成一个文本的列表，列表中的每个元素是每个单元格的行和列全部文本
    def table_to_list_1(table):
        table_text = []
        number = table['cell_ID_matrix']
        value = table['chinese_cell_value_list']
        for i in range(len(number)):
            for j in range(len(number[i])):
                text = ''
                for k in range(len(number[i])):
                    if k == j:
                        pass
                    text = text + value[number[i][k]] + '*'
                for k in range(len(number)):
                    if k == i:
                        pass
                    text = text + value[number[k][j]] + '*'
                text = text + value[number[i][j]]
                table_text.append(text)
        return table_text

    def table_to_list_2(table):
        table_text = []
        number = table['cell_ID_matrix']
        value = table['chinese_cell_value_list']
        for i in range(len(number)):
            for j in range(len(number[i])):
                text = ''
                attribute = table['column_attribute'] + table['row_attribute'] + table['column_index'] + table[
                    'row_index']
                for k in range(len(number[i])):
                    if number[i][k] in attribute:
                        if k == j:
                            pass
                        text = text + value[number[i][k]] + '*'
                for k in range(len(number)):
                    if number[k][j] in attribute:
                        if k == i:
                            pass
                        text = text + value[number[k][j]] + '*'
                text = text + value[number[i][j]]
                table_text.append(text)
        return table_text

    def table_to_list_3(table):
        table_text = []
        number = table['cell_ID_matrix']
        value = table['chinese_cell_value_list']
        for i in range(len(number)):
            for j in range(len(number[i])):
                text = ''
                attribute = table['column_attribute'] + table['row_attribute']
                index = table['column_index'] + table['row_index']
                column_attribute = True
                column_index = True
                row_attribute = True
                row_index = True
                for k in range(i):
                    if number[i - k - 1][j] in attribute and row_attribute:
                        text = value[number[i - k - 1][j]] + '*' + text
                        row_attribute = False
                    if number[i - k - 1][j] in index and row_index:
                        text = value[number[i - k - 1][j]] + '*' + text
                        row_index = False
                for k in range(j):
                    if number[i][j - k - 1] in attribute and column_attribute:
                        text = value[number[i][j - k - 1]] + '*' + text
                        column_attribute = False
                    if number[i][j - k - 1] in index and column_index:
                        text = value[number[i][j - k - 1]] + '*' + text
                        column_index = False

                text = text + value[number[i][j]]
                table_text.append(text)
        return table_text

    def table_to_list_4(table):
        table_text = []
        number = table['cell_ID_matrix']
        value = table['chinese_cell_value_list']
        for i in range(len(number)):
            for j in range(len(number[i])):
                text = ''
                attribute = table['column_attribute'] + table['row_attribute'] + table['column_index'] + table[
                    'row_index']
                for k in range(j):
                    if number[i][j - k - 1] in attribute:
                        text = text + value[number[i][j - k - 1]] + '*'
                        break
                for k in range(i):
                    if number[i - k - 1][j] in attribute:
                        text = text + value[number[i - k - 1][j]] + '*'
                        break
                text = text + value[number[i][j]]
                table_text.append(text)
        return table_text

    # 根据问题中的table_id找到表格
    def found_table(table_id, table_list):
        for i in range(len(table_list)):
            if table_list[i]['table_id'] == table_id:
                return table_list[i]
        return 'null'

    def get_index_list(answer, turth):
        restlu = []
        for i in range(len(turth)):
            if answer == turth[i]:
                restlu.append(i)
        return restlu

    def dataset_to_traindata(dataset):
        with open('./newdata/' + dataset + '_tables.json', 'r', encoding='utf-8') as file:
            table_list = json.load(file)
        with open('./newdata/' + dataset + '_questions.json', 'r', encoding='utf-8') as file:
            qa_list = json.load(file)
        # 单元格结果
        cell_result = []
        # 遍历全部问题的列表
        for i in range(len(qa_list)):
            qa = qa_list[i]
            table = found_table(qa['table_id'], table_list)
            if testnum == 1 or testnum == 5:
                table_text_list = table_to_list_1(table)
            if testnum == 2 or testnum == 6:
                table_text_list = table_to_list_2(table)
            if testnum == 3 or testnum == 7:
                table_text_list = table_to_list_3(table)
            if testnum == 4 or testnum == 8:
                table_text_list = table_to_list_4(table)
            # 将表格标注变成一维数组
            turth = [element for sublist in table['cell_ID_matrix'] for element in sublist]
            # 找到全部答案在一维数组的位置
            answer_list = []
            for j in range(len(qa['answer_cell_list'])):
                answer_list = answer_list + get_index_list(qa['answer_cell_list'][j], turth)
            # 遍历需要处理的每一行
            for j in range(len(table_text_list)):
                # id
                id = qa['question_id'] + ':' + str(j)
                # taxt_a：问题
                text_a = qa['chinese_question']
                # label：检查答案是否在这一行
                label = False
                if j in answer_list:
                    label = True
                # text_b：行和列上全部的文本组成
                text_b = table_text_list[j]
                # question_type:问题的类型
                question_type = 0
                if qa['question_type'] == 'single_cell':
                    question_type = 1
                elif qa['question_type'] == 'one_col':
                    question_type = 2
                elif qa['question_type'] == 'one_row':
                    question_type = 3
                else:
                    question_type = 4
                if testnum == 1 or testnum == 2 or testnum == 3 or testnum == 4:
                    cell_result.append(
                        {'id': id, 'text_a': text_a, 'text_b': text_b, 'label': label, 'question_type': question_type})
                if testnum == 5 or testnum == 6 or testnum == 7 or testnum == 8:
                    cell_result.append(
                        {'id': id, 'text_a': text_b, 'text_b': text_a, 'label': label, 'question_type': question_type})

        if not os.path.exists('./traindata/test' + str(testnum)):
            os.makedirs('./traindata/test' + str(testnum))
        with open('./traindata/test' + str(testnum) + '/' + dataset + '_cells.jsonl', "w",
                  encoding="utf-8") as jsonl_file:
            for item in cell_result:
                json.dump(item, jsonl_file, ensure_ascii=False)
                jsonl_file.write('\n')

        jsonl_to_gz('./traindata/test' + str(testnum) + '/' + dataset + '_cells.jsonl',
                    './traindata/test' + str(testnum) + '/' + dataset + '_cells.jsonl.gz')

    dataset_to_traindata('train')
    dataset_to_traindata('test')


for i in range(8):
    data_to_traindata(i + 1)
print('success')
