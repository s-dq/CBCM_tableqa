# Experiment Code

## 1. Requirements
jsonlines==4.0.0
ujson==5.10.0
paddlepaddle-gpu==2.6.2
paddlenlp==2.8.1

## 2. Data preprocessing
### (1)download IM-TQA-main.zip:
'''
# wget https://github.com/SpursGoZmy/IM-TQA/archive/refs/heads/main.zip
'''
### (2)unzip main.zip:
unzip main.zip
'''
gunzip ./IM-TQA-main/TQA_code/datasets/IM_TQA/test_cols.jsonl.gz
gunzip ./IM-TQA-main/TQA_code/datasets/IM_TQA/test_rows.jsonl.gz
gunzip ./IM-TQA-main/TQA_code/datasets/IM_TQA/train_cols.jsonl.gz
gunzip ./IM-TQA-main/TQA_code/datasets/IM_TQA/train_rows.jsonl.gz
'''
### (3)process
copy im-tqa dataset:
'''
cp -r ./IM-TQA-main/data  ./data
Building new benchmark data:
python data_to_newdata.py
'''
## 3. Experiment 1
### (1) Python environment setup：
export PYTHONPATH=pwd/IM-TQA-main/TQA_code/
### (2)Generate data for comparative experiments：
python rgcnrci_to_newrgcnrci.py
### (3)Generate data for all cell semantic representation methods:
python newdata_to_traindata.py
### (4)experiments:
'''
sh test1.sh
........
sh test8.sh
'''
### (5)comparative experiment:
'''
sh test_rgcnrci.sh
'''
## 4. Experiment 2
### (1)Copy the best cell semantic representation method in Experiment 1 to Experiment 2:
'''
cp -r ./traindata/test3 ./traindata/best
cp -r ./result/test3 ./result/best
gunzip ./result/best/apply/cells/results0.jsonl.gz
'''
### (2)Encoding text data into semantic vectors:
'''
python test_ml_embedding.py
'''
### (3)Machine learning experiment:
'''
python test_ml_svm.py
python test_ml_randomforest.py
python test_ml_knn.py
'''
### (4)Calculate the text classification accuracy of the method in experiment 1:
'''
python test_ml_compare_best.py
'''
