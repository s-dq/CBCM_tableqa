一、数据准备

1.1环境

创建虚拟环境

python3 -m venv venv

激活虚拟环境

source venv/bin/activate

安装

pip install jsonlines==4.0.0

pip install ujson==5.10.0

python -m pip install paddlepaddle-gpu==2.6.2.post116 -i https://www.paddlepaddle.org.cn/packages/stable/cu116/

pip install paddlenlp

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

将实验一最好的结果复制

cp -r ./traindata/test3 ./traindata/best

cp -r ./result/test3 ./result/best

gunzip ./result/best/apply/cells/results0.jsonl.gz

将文本数据编码成为语义向量，为机器学习作为输入

python test_ml_embedding.py

机器学习svm实验

python test_ml_svm.py

机器学习randomforest实验

python test_ml_randomforest.py

机器学习knn实验

python test_ml_knn.py

计算实验一方法的文本分类准确率

python test_ml_compare_best.py
