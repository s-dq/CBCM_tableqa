import json

import sys

# Redirect stdout to a file
sys.stdout = open("./results/output.txt", "a")
                  
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

pred_file=read_jsonl('./result/best/apply/cells/results0.jsonl')
turth_file=read_jsonl('./traindata/besst/test_cells.jsonl')

right=0
wrong=0
for i in range(len(pred_file)):
    for j in range(len(turth_file)):
        if pred_file[i]['id']==turth_file[j]['id']:
            if pred_file[i]['predictions'][0]>pred_file[i]['predictions'][0] and turth_file[j]['lable']:
                right=right+1
            elif pred_file[i]['predictions'][0]<pred_file[i]['predictions'][0] and not turth_file[j]['lable']:
                right=right+1
            else:
                wrong=wrong+1
        else:
            continue
print('文本分类准确率：',right/(right+wrong))
sys.stdout.close()
sys.stdout = sys.__stdout__
