一、IM-TQA-x benchmark dataset
1、IM-TQA数据集来源及简介：https://github.com/SpursGoZmy/IM-TQA
原数据集中存在的定义
![image](https://github.com/user-attachments/assets/86bde8b2-1662-4791-ae08-81445beb4459)

As shown in Figure 1, we divide tables into 4 types according to their structure characteristics, which is in line with previous works with complex table as an important complement. Exploring and including more table types deserve future investigations.
Vertical Table: Table data is arranged in the vertical direction, with the first row as column headers and other rows as data tuples.
Horizontal Table: Table data is arranged in the horizontal direction, with the first column as row headers and other columns as data tuples.
Hierarchical Table: Table data is arranged in both vertical and horizontal directions, with headers exhibiting a multi-level hierarchical structure.
Complex Table: In tables of above 3 types, header cells only locate on the top or left side of the table. But in complex tables, headers also appear at other positions such as bottom-right region in the table and can be mixed with data cells. Such tabular structures with flexible header locations often appear in professional equipment specifications and record sheets, presenting a great challenge to existing methods.
To promote the understanding of implicit table structures, we categorize table cells into 5 types based on their functional roles, with the concentration on header cells that are useful for TQA models to locate correct answer cells.

Row Attribute and Column Attribute: Row attribute and column attribute are traditional table headers which describes other cells in the same row and in the same column respectively, e.g., yellow cells and red cells in Figure 1. Attribute cells only serve the purpose of describing other cells and they are not meaningful data.
Row Index and Column Index: Row index and column index are individual cells that are used to index data records in the row or column orientation, e.g., blue cells and green cells in Figure 1. Index cells are also meaningful data. For instance, in vertical tables, data cells in the primary key column are unique identifiers of each row.
Pure Data: Pure data cells are the core body of a table. They do not have the function of describing or indexing other cells and their meanings should be understood with the help of above header cells.
新增的定义：答案分布类型
![image](https://github.com/user-attachments/assets/8a331450-86a9-4533-a01c-c5823276b1ce)

（1）
（2）
（3）
2、IM-TQA-x benchmark数据集构建原因
（1）
(2)训练过程中不受答案分布类型影响

3、IM-TQA-x benchmark数据集构建方法
(1)find all complex multi-line cell recall questions by applying the R1 rule
(2)identify all the related tables and all the questions on those tables (one table with about four questions).
(3)Organize all the CQ tables and questions into the test set

4、IM-TQA-x benchmark 统计
![image](https://github.com/user-attachments/assets/57d6aa8f-247b-4012-8a87-5596044aad2b)

二、单元格语义表征实验


三、文本二分类方法实验


