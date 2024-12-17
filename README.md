一、数据准备
1.1下载
下载IM-TQA-main.zip
wget https://github.com/SpursGoZmy/IM-TQA/archive/refs/heads/main.zip
解压main.zip
unzip main.zip
解压rgcnrci训练数据
gunzip ./IM-TQA-main/TQA_code/datasets/IM_TQA/test_cols.jsonl.gz
gunzip ./IM-TQA-main/TQA_code/datasets/IM_TQA/test_rows.jsonl.gz
gunzip ./IM-TQA-main/TQA_code/datasets/IM_TQA/train_cols.jsonl.gz
gunzip ./IM-TQA-main/TQA_code/datasets/IM_TQA/train_rows.jsonl.gz

1.2预处理
复制im-tqa数据集，运行：cp -r ./IM-TQA-main/data  ./data
构建新的数据集，运行：python data_to_newdata.py
根据新数据集调整rgcnrci训练数据，运行python rgcnrci_to_newrgcnrci.py
新数据集到实验训练数据集，运行：python newdata_to_traindata.py

二、实验一
设置python环境
export PYTHONPATH=/home/jupyter-sdq/AAF/IM-TQA-main/TQA_code/
行列匹配的对比试验：
sh test_rgcnrci.sh
不同单元格匹配8个实验
sh test1.sh
........
sh test8.sh

三、实验二
python ./test_ml.py
