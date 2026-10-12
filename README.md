# Kaggle Titanic — 二分类入门

Kaggle 经典入门题 [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)：
根据乘客的舱位、性别、年龄、票价等信息，预测其是否生还。

## 结果

| 模型 | 5 折 CV 准确率 |
|---|---|
| **Gradient Boosting（最终采用）** | **0.8395 ± 0.0133** |
| SVM (RBF) | 0.8384 ± 0.0209 |
| Soft Voting 集成（GB+SVM+RF） | 0.8372 ± 0.0115 |
| Random Forest | 0.8350 ± 0.0124 |
| Logistic Regression | 0.8283 ± 0.0126 |
| （性别基线：女性全生还） | 0.7868 |

样本外混淆矩阵（891 人）：

```
            预测遇难   预测生还
真实遇难       487        62
真实生还        81       261
```

## 快速开始

```bash
pip install -r requirements.txt
python titanic.py                 # 完整流程
python titanic.py --cv 10         # 10 折交叉验证
python titanic.py --model random_forest   # 只跑指定模型
python titanic.py --list-models   # 查看可选模型
```

运行后生成：

- `output/submission.csv` — 提交文件（已按 Kaggle 格式自检：418 行、PassengerId 与 test.csv 同序、取值 0/1）
- `output/cv_results.csv` — 各模型交叉验证得分
- `output/model_comparison.png` — 模型对比图
- `output/feature_importance.png` — 特征重要性

## 方法

数据：`data/train.csv`（891 人，含标签）与 `data/test.csv`（418 人）。

特征工程（共 17 列特征）：

| 特征 | 说明 |
|---|---|
| `Title` | 从姓名提取 Mr / Mrs / Miss / Master 等头衔，稀有头衔合并为 Rare |
| `FamilySize` / `IsAlone` | SibSp + Parch + 1；是否独自登船 |
| `TicketFreq` / `TicketPrefix` | 同票号人数（同票常同生共死）、票号前缀 |
| `CabinDeck` | 客舱首字母（甲板层），缺失归为 U |
| `FarePerPerson` / `FareBin` | 人均票价及其分箱，缓解长尾 |
| `IsChild` / `AgeBin` | 是否儿童（<14 岁）、年龄分箱 |

缺失值处理：Age 用「同头衔 + 同舱位」中位数填充，Fare 用同舱位中位数，Embarked 取众数。

建模：数值列标准化、类别列独热编码（`ColumnTransformer` + `Pipeline`，无数据泄漏），
5 个候选模型分层 K 折交叉验证，前三名软投票集成，最终选 CV 最优者
在全量训练集重训后预测测试集。

## 文件结构

```
kaggle-titanic/
├── titanic.py           # 全部代码（一条命令跑通）
├── requirements.txt
├── data/                # train.csv / test.csv（Kaggle 官方数据）
└── output/              # submission.csv、CV 结果、图表（运行生成）
```
