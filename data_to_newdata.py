import json
import itertools
import os

with open('./data/train_questions.json', 'r', encoding='utf-8') as file:
    train_questions = json.load(file)
with open('./data/train_tables.json', 'r', encoding='utf-8') as file:
    train_tables = json.load(file)
with open('./data/test_questions.json', 'r', encoding='utf-8') as file:
    test_questions = json.load(file)
with open('./data/test_tables.json', 'r', encoding='utf-8') as file:
    test_tables = json.load(file)

question_list = train_questions + test_questions
table_list = train_tables + test_tables


# 获取全部行列匹配不能解决的问题的id，返回结果idlist是存在不能用行列匹配解决的问题的表格的id的列表，if_calc=False就直接获取，True就重新计算获取
def get_rci_nouse_table_id(if_calc):
    # 根据问题中的tableid找到问题对应的表格
    def found_table(question, table_list):
        for i in range(len(table_list)):
            if question['table_id'] == table_list[i]['table_id']:
                return table_list[i]
        return 0

    # 判断行列匹配的方法能否解决当前的问题
    def if_rci_useful(question, table):
        col_nums = len(table['cell_ID_matrix'][0])
        row_nums = len(table['cell_ID_matrix'])
        for row_temp in range(row_nums):  # 选择行的数量
            for col_temp in range(col_nums):  # 选择列的数量
                row_lines_list = list(itertools.combinations(range(row_nums), row_temp + 1))  # 全部的行的组合
                col_lines_list = list(itertools.combinations(range(col_nums), col_temp + 1))  # 全部的列的组合
                # 遍历每个行列组合
                for i in range(len(row_lines_list)):
                    for j in range(len(col_lines_list)):
                        # 当前行列组合下的答案
                        answer = []
                        for k in range(len(row_lines_list[i])):
                            for l in range(len(col_lines_list[j])):
                                answer.append(table['cell_ID_matrix'][row_lines_list[i][k]][col_lines_list[j][l]])
                        # 判断能否行列匹配得到答案
                        if set(answer) == set(question['answer_cell_list']):
                            return True
        return False

    if not if_calc:
        id_list = ['WV1g6IVe', 'FcURgbrI', 'xgNdc3sc', 'G23Sw0Z6', 'ZuDI3FR9',
                   'Un1AFlCd', 'wEKDDD3K', 'CyYtNfVP', 'MSWUkXii', 'PpsGU2ci',
                   'S4NTWQNH', 'nJlmYVa9', '9IX0QDKS', 'qdUkJJWl', 'xVGEuVT5',
                   'stizF5eP', 'zWgbCKw9', 'h5MXsJLu', 'uwEVxkQM', 'KxuCyCG6',
                   'QFwk9z6B', 'zIzziCAU', '5FM9tfQF', 'EZVUv5qL', 'QzEZbhx4',
                   'N9fwHnJN', 'W5WNRvNO', 'hqZwKnXL', 'XvUdTjjN', 'eFZK12X2',
                   'fzWpQEfb', '0JDfkNrD', 'gpUEd2dF', '75xF1Ggo']
        return id_list
    else:
        # 非常久...........................
        id_list = []
        for i in range(len(question_list)):
            if if_rci_useful(question_list[i], found_table(question_list[i], table_list)):
                pass
            else:
                id_list.append(question_list[i]['table_id'])
        return list(set(id_list))


# 把训练集不能用rci方法的表格找到，然后找到表格上的全部的问题，放到测试集中
table_id_list = get_rci_nouse_table_id(False)

test_tables = test_tables + [t for t in train_tables if t['table_id'] in table_id_list]
train_tables = [t for t in train_tables if t['table_id'] not in table_id_list]
test_questions = test_questions + [q for q in train_questions if q['table_id'] in table_id_list]
train_questions = [q for q in train_questions if q['table_id'] not in table_id_list]

# 保存新的数据集
if not os.path.exists('./newdata'):
    os.makedirs('./newdata')
with open("./newdata/train_tables.json", "w", encoding="utf-8") as file:
    json.dump(train_tables, file, ensure_ascii=False, indent=4)
with open("./newdata/test_tables.json", "w", encoding="utf-8") as file:
    json.dump(test_tables, file, ensure_ascii=False, indent=4)
with open("./newdata/train_questions.json", "w", encoding="utf-8") as file:
    json.dump(train_questions, file, ensure_ascii=False, indent=4)
with open("./newdata/test_questions.json", "w", encoding="utf-8") as file:
    json.dump(test_questions, file, ensure_ascii=False, indent=4)

print('success')
