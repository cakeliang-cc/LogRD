# HDFS 数据流水线（clean → 模板 ID）

## 步骤流水

1. 读原始 **`HDFS.log`**，按行解析时间、提取 **`blk_…`**，过滤规则与现有 HDFS 处理一致（如一行多块的丢弃方式）。

2. 从每行取出**日志消息**（去掉时间、级别等头部，规则固定即可）。

3. 对消息调用 **`clean()`**（`data_clean.py`），得到字符串，视为该行的**模板键**。

4. 扫描全部相关日志行，为每个不同的模板键分配**模板 ID**（整数，建议稳定顺序），汇总写入 **`all_vocab.csv`**（模板键 ↔ 模板 ID）。

5. 按 **`BlockId`** 聚合：块内按时间排序，将每行对应的模板 ID 连成序列，得到 **`contentIds`**。

6. 用 **`anomaly_label.csv`** 划分块：Normal 进训练/正常测试集，Anomaly 进异常测试集，写出 **`train.csv`**、**`test_normal.csv`**、**`testAnomaly.csv`**（格式与现在一致；异常集带 `Label`）。

7. 根据 **`train.csv`** 里实际出现过的模板 ID，生成 **`train_vocab.csv`**（`contentId` → `vocabId`），供下游 LogBERT 使用。

8. 下游训练/预测仍使用上述 CSV + **`train_vocab.csv`**；**生成 CSV 的阶段不做**句级 BERT 嵌入。

9. 根据 **`all_vocab.csv`** 中的每条 **Template（模板键）**，用**有监督 SimCSE** 做语义嵌入；将得到的向量矩阵保存为 **`E:\cake\LogRD\LogRD\WhiteningNpy\HDFS.npy`**（行与 `all_vocab` 中模板 ID 顺序对齐，便于按 ID 索引）。
