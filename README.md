# 实验一：新闻文本十分类

当代人工智能实验一，基于 TF-IDF 特征与经典机器学习算法（朴素贝叶斯、线性 SVM、逻辑回归、多层感知机）完成英文新闻文本十分类，并通过验证集网格调参选择最优模型，对无标签测试集进行预测。

## 目录结构

```
.
├── main.py                     # 实验主程序（数据划分、清洗、特征提取、调参、评估、预测）
├── data/
│   ├── train_data.csv          # 训练集（text, target）
│   └── test_data_unlabeled.csv # 无标签测试集（text）
├── figures/                    # 实验生成的图表（模型对比、混淆矩阵、每类指标）
├── results/                    # 验证集完整结果、每类指标、最优模型记录
├── prediction.csv              # 最优模型对测试集的预测结果（最终提交文件）
└── requirements.txt
```

## 环境配置

```bash
conda create -n project1 python=3.10
conda activate project1
pip install -r requirements.txt
```

## 运行方式

```bash
python main.py
```

运行结束后会自动：

1. 分层划分 80% 训练集 / 20% 验证集（随机种子 42）；
2. 对原始文本基线特征与清洗后的改进特征（unigram+bigram、sublinear TF、停用词过滤）分别建模；
3. 对四种模型在验证集上进行网格调参并输出准确率、macro-F1、训练耗时；
4. 生成 `figures/` 下的模型对比图、混淆矩阵图、每类指标图；
5. 用验证集上的最优配置在全部训练数据上重新训练，输出测试集预测文件 `prediction.csv`。

## 实验设置

- 随机种子固定为 42，保证结果可复现；
- 评价指标：准确率（accuracy）、宏平均 F1（macro-F1）、每类精确率/召回率/F1；
- 文本清洗：去除邮件头、引用行、邮箱与网址、数字、标点，统一小写；
- 模型与调参网格：
  - MultinomialNB：`alpha ∈ {0.01, 0.05, 0.1, 0.3, 0.5, 1.0}`
  - LinearSVC：`C ∈ {0.1, 0.3, 0.5, 1, 3, 5, 10}`
  - LogisticRegression：`C ∈ {0.1, 0.5, 1, 3, 5, 10, 100}`
  - MLPClassifier：隐藏层结构与 `alpha` 共 3 组配置，启用 early stopping
