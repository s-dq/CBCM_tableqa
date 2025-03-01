# 1 IM-TQA-x benchmark dataset

## (1) Source and Introduction of IM-TQA Dataset

https://github.com/SpursGoZmy/IM-TQA

## (2) Dataset Description

![image](https://github.com/user-attachments/assets/86bde8b2-1662-4791-ae08-81445beb4459)

### Definition of table types in table question answering

Vertical Table: Table data is arranged in the vertical direction, with the first row as column headers and other rows as data tuples.

Horizontal Table: Table data is arranged in the horizontal direction, with the first column as row headers and other columns as data tuples.

Hierarchical Table: Table data is arranged in both vertical and horizontal directions, with headers exhibiting a multi-level hierarchical structure.

Complex Table: In tables of above 3 types, header cells only locate on the top or left side of the table. But in complex tables, headers also appear at other positions such as bottom-right region in the table and can be mixed with data cells. Such tabular structures with flexible header locations often appear in professional equipment specifications and record sheets, presenting a great challenge to existing methods.

### Cell type classification in table question answering

Row Attribute and Column Attribute(yellow and red): Row attribute and column attribute are traditional table headers which describes other cells in the same row and in the same column respectively, e.g., yellow cells and red cells in Figure 1. Attribute cells only serve the purpose of describing other cells and they are not meaningful data.

Row Index and Column Index(blue and green): Row index and column index are individual cells that are used to index data records in the row or column orientation, e.g., blue cells and green cells in Figure 1. Index cells are also meaningful data. For instance, in vertical tables, data cells in the primary key column are unique identifiers of each row.

Pure Data: Pure data cells are the core body of a table. They do not have the function of describing or indexing other cells and their meanings should be understood with the help of above header cells.

### Question classification in table question answering

Based on the original data set, we defined the distribution type of question answers，all examples are given in the figure.

Single Cell Query: The answer to the question consists of a single cell.

Single Line Query: The answer to the question consists of multiple cells in a single row or column.

Multi-Line Query: The answer to the question appears in positions of more than one row and more than one column.

![image](https://github.com/user-attachments/assets/8a331450-86a9-4533-a01c-c5823276b1ce)

## (3) Constructing IM-TQA-x benchmark dataset

### We adjust the original dataset for the following reasons:

Multi-Line Query type problems are too small in the test set

The type of questions in the training set does not affect the training results

### IM-TQA-x benchmark dataset construction steps

find all complex multi-line cell recall questions by applying the R1 rule (Questions that meet this rule can only be answered using the cell matching method, not the row and column matching method)

R1: ∀(i,j) ∈ {(r, c)}, ∃(i, k) ∈ A ∧ ∃(m,j) ∈ A | i ≠ m, k ≠ j

R1 ⟹ (i,j) ∈ A

Where:

{(r, c)} is a table, r is the row index, c is the column index. (i,j) is a table cell with i(th) row and j(th) column. A is the answer table cell set.

identify all the related tables and all the questions on those tables (one table with about four questions).

Organize those tables and questions into the test set

The final dataset is in the IM-TQA-X Benchmark Dataset folder

## (4) IM-TQA-x benchmark statistics

| |train table|train question|test table|test question|
|---| ---| ---| ---| ---|
|total|907|3771|183|768|
|Classification by table type:|
|Complex Table|223|1014|66|306|
|Vertical Table|224|849|45|174|
|Horizontal Table|229|933|38|129|
|Hierarchical Table|231|1075|34|159|
|Classification by question type:|
|Single cell query|-|2112|-|404|
|Single Line Query|-|1630|-|317|
|Multi-Line Query|-|29|-|47|

# 2 Experiment

## (1) Steps to implement form question and answer

![image](https://github.com/user-attachments/assets/2251d4df-6ec3-4596-98a8-babd335539fd)

## (2) Impact validation of cell semantic representation method on question-answering

The following figure shows the experimental results. The indicator is the accuracy of question answering

|Experiments|Overall accuracy|Complex Table|Vertical Table|Horizontal Table|Hierarchical Table|Single cell query|Single Line Query|Multi-Line Query|
|---| ---| ---| ---| ---|---| ---| ---| ---|
| 1 | 40.4% | 19.9% | 62.6% | 46.5% | 50.3% | 46.3% | 37.5% | 8.5%  |
| 2 | 38.3% | 18.6% | 56.3% | 50.4% | 46.5% | 43.1% | 36.3% | 10.6% |
| 3 | 50.5% | 45.4% | 59.8% | 58.2% | 44.0% | 55.9% | 46.1% | 34.0% |
| 4 | 48.3% | 44.1% | 58.0% | 54.3% | 40.9% | 54.5% | 43.2% | 29.8% |
| 5 | 38.9% | 16.3% | 62.1% | 49.6% | 48.4% | 45.8% | 35.6% | 2.1%  |
| 6 | 41.7% | 24.2% | 56.9% | 51.2% | 50.9% | 49.0% | 37.5% | 6.4%  |
| 7 | 49.3% | 41.2% | 60.9% | 58.9% | 44.7% | 55.7% | 44.8% | 25.5% |
| 8 | 49.1% | 41.5% | 57.5% | 55.8% | 49.1% | 55.4% | 44.5% | 25.5% |

Experiments 1 to 4 used different cell semantic representation methods, respectively:

Full row and column text: All cells in the current row and all cells in the current column are concatenated in sequence.

Full row and column attribute index: All attributes and index cells in the current row and all 
attributes and index cells in the current column are concatenated in sequence.

Nearest neighbor attribute and index: The left nearest neighbor row attribute cell, row index cell, the upper nearest neighbor column attribute cell, and column index cell are concatenated in sequence.

Nearest neighbor attribute or index: The left nearest neighbor row attribute cell or row index cell is used. The upper nearest neighbor column attribute or column index is appended.

And the question and answer text concatenation order is: Question text + Cell semantic text

5 to 8 are composed of the same order of cell semantic representation methods and different question-answer text concatenation methods: Cell semantic text + Question text.

## (3) Impact validation of text classification method on question-answering：

The following figure shows the experimental results. Text classification accuracy is the accuracy of a cell being judged correctly, and Table question answering accuracy is the accuracy of the entire table cell being judged correctly

| Text binary classification method | Text classification accuracy | Table question answering accuracy |
|--|--|--|
| SVM               | 71.72% | 1.69%  |
| Random forest     | 78.8%  | 2.21%  |
| k nearest neighbor | 81.64% | 3.25%  |
| Linear classifier | 97%    | 49%    |

# Reference

```
@inproceedings{zheng-etal-2023-im,
    title = "{IM}-{TQA}: A {C}hinese Table Question Answering Dataset with Implicit and Multi-type Table Structures",
    author = "Zheng, Mingyu  and
      Hao, Yang  and
      Jiang, Wenbin  and
      Lin, Zheng  and
      Lyu, Yajuan  and
      She, QiaoQiao  and
      Wang, Weiping",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.278",
    doi = "10.18653/v1/2023.acl-long.278",
    pages = "5074--5094",
}
```
