#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle - Titanic: Machine Learning from Disaster
二分类入门题：根据乘客信息预测其是否生还。

用法
----
    python titanic.py                      # 完整流程（默认）
    python titanic.py --cv 10              # 指定交叉验证折数
    python titanic.py --model rf           # 只跑指定模型
    python titanic.py --seed 7             # 换随机种子

流程
----
    1. 读取 data/train.csv 与 data/test.csv
    2. 特征工程：头衔(Title)、家庭规模、票号分组、客舱甲板、票价分箱
    3. 缺失值填充：Age 按 (Title, Pclass) 中位数，Fare 按 Pclass 中位数，Embarked 取众数
    4. 多模型分层 K 折交叉验证对比
    5. 选取 CV 表现最好的模型在全量训练集重训，预测测试集并写出 submission.csv

输出
----
    output/submission.csv        提交文件（PassengerId,Survived）
    output/cv_results.csv        各模型交叉验证得分
    output/model_comparison.png  模型对比图
    output/feature_importance.png 随机森林特征重要性
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"

# 稀有头衔统一归为 Rare，避免类别过碎
RARE_TITLES = {
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
    "Sir", "Jonkheer", "Dona", "the",
}
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}


# --------------------------------------------------------------------------
# 1. 数据读取
# --------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取训练集与测试集，并校验行数是否符合 Kaggle 官方（891 / 418）。"""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    print(f"[数据] 训练集 {train.shape}，测试集 {test.shape}")
    if train.shape[0] != 891 or test.shape[0] != 418:
        print("[警告] 行数与 Kaggle 官方不符，请确认数据文件是否被改动")
    print(f"[数据] 训练集生还率 {train['Survived'].mean():.4f}")
    return train, test


# --------------------------------------------------------------------------
# 2. 特征工程
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """把原始字段加工成模型可用的特征（对数式见函数体注释）。"""
    out = df.copy()

    # --- 头衔：从姓名中提取 Mr / Mrs / Miss / Master，其余合并为 Rare ---
    out["Title"] = out["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    out["Title"] = out["Title"].replace(TITLE_MAP)
    out["Title"] = out["Title"].where(out["Title"].isin(
        {"Mr", "Mrs", "Miss", "Master", "Dr", "Rev"}), "Rare")

    # --- 家庭：父母子女数 + 兄弟姐妹配偶数 + 本人 ---
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # --- 票号：同票号往往同生共死，用票号出现次数和票价人均值刻画 ---
    out["TicketFreq"] = out.groupby("Ticket")["PassengerId"].transform("size")
    out["TicketPrefix"] = (
        out["Ticket"].str.replace(r"[./]", "", regex=True)
        .str.replace(r"\d", "", regex=True)
        .str.strip().replace("", "NONE")
    )

    # --- 客舱：首字母代表甲板层，缺失归为 U(Unknown) ---
    out["CabinDeck"] = out["Cabin"].fillna("U").str[0].str.upper()

    # --- 票价：人均票价 + 分箱，缓解长尾 ---
    out["FarePerPerson"] = out["Fare"] / out["TicketFreq"]
    fare_bins = [-1, 7.9, 14.5, 31.0, 1e9]
    out["FareBin"] = pd.cut(out["Fare"], bins=fare_bins, labels=[0, 1, 2, 3])

    # --- 年龄：先粗略分箱（缺失单独一档），再补精确值 ---
    out["IsChild"] = (out["Age"] < 14).astype(int)
    out["AgeBin"] = pd.cut(
        out["Age"], bins=[-1, 12, 18, 30, 50, 100], labels=[0, 1, 2, 3, 4]
    )

    # --- 出发港缺失取众数（只有 2 条） ---
    out["Embarked"] = out["Embarked"].fillna(out["Embarked"].mode()[0])

    return out


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """用「同头衔同舱位」的中位数补年龄，同舱位中位数补票价——比全局均值更合理。"""
    out = df.copy()

    out["Age"] = out["Age"].fillna(
        out.groupby(["Title", "Pclass"])["Age"].transform("median")
    )
    out["Age"] = out["Age"].fillna(out["Age"].median())

    out["Fare"] = out["Fare"].fillna(
        out.groupby("Pclass")["Fare"].transform("median")
    )
    # 人均票价依赖 Fare，补完重算一次
    out["FarePerPerson"] = out["Fare"] / out["TicketFreq"]

    return out


# --------------------------------------------------------------------------
# 3. 预处理管线 + 模型
# --------------------------------------------------------------------------
NUMERIC_COLS = ["Age", "SibSp", "Parch", "Fare", "FarePerPerson",
                "FamilySize", "TicketFreq", "IsAlone", "IsChild"]
CATEGORICAL_COLS = ["Pclass", "Sex", "Embarked", "Title",
                    "CabinDeck", "TicketPrefix", "FareBin", "AgeBin"]


def make_preprocessor() -> ColumnTransformer:
    """数值列标准化，类别列独热编码（忽略测试集出现但训练集没有的取值）。"""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])


def make_models(seed: int) -> dict[str, Pipeline]:
    """返回「模型名 -> 预处理+分类器 的完整管线」。"""
    def pipe(clf) -> Pipeline:
        return Pipeline([("prep", make_preprocessor()), ("clf", clf)])

    return {
        "logreg": pipe(LogisticRegression(max_iter=2000, C=0.5,
                                          random_state=seed)),
        "random_forest": pipe(RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=3,
            min_samples_split=5, random_state=seed, n_jobs=-1)),
        "gradient_boosting": pipe(GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=seed)),
        # SVC(probability=True) 在新版 sklearn 已弃用，改用手动校准以拿到概率
        "svm_rbf": pipe(CalibratedClassifierCV(
            SVC(C=2.0, gamma=0.05, kernel="rbf", random_state=seed),
            method="sigmoid", ensemble=False)),
    }


# --------------------------------------------------------------------------
# 4. 交叉验证
# --------------------------------------------------------------------------
def evaluate(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
             cv: int, seed: int) -> pd.DataFrame:
    """分层 K 折交叉验证，返回按平均准确率排序的得分表。"""
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=splitter,
                                 scoring="accuracy", n_jobs=-1)
        rows.append({
            "model": name,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
            "folds": " ".join(f"{s:.4f}" for s in scores),
        })
        print(f"[CV] {name:<20} 准确率 {scores.mean():.4f} "
              f"(±{scores.std():.4f})  各折 {[round(float(s), 4) for s in scores]}")
    return pd.DataFrame(rows).sort_values("cv_mean", ascending=False)


def build_ensemble(cv_df: pd.DataFrame, models: dict[str, Pipeline],
                   top_k: int = 3) -> tuple[str, Pipeline]:
    """把 CV 排名前 K 的单模型做软投票集成（需要各模型都能输出概率）。"""
    top = cv_df["model"].head(top_k).tolist()
    ensemble = VotingClassifier(
        estimators=[(n, models[n]) for n in top], voting="soft", n_jobs=-1)
    return f"soft_voting({'+'.join(top)})", ensemble


# --------------------------------------------------------------------------
# 5. 可视化
# --------------------------------------------------------------------------
def plot_scores(cv_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    order = cv_df.iloc[::-1]
    bars = ax.barh(order["model"], order["cv_mean"],
                   xerr=order["cv_std"], color="#4C78A8",
                   error_kw={"ecolor": "#999", "capsize": 3})
    ax.bar_label(bars, fmt="%.4f", padding=8, fontsize=9)
    ax.set_xlim(0.7, max(cv_df["cv_mean"]) + 0.06)
    ax.set_xlabel("Cross-validated accuracy")
    ax.set_title("Titanic — model comparison", fontsize=12, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_comparison.png")
    plt.close(fig)


def plot_importance(model: Pipeline) -> None:
    """画出树模型的特征重要性（独热后的特征合并回原始列名）。"""
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return

    names = prep.get_feature_names_out()
    # num__Age -> Age；cat__Title_Mr -> Title
    pretty = [re.sub(r"^(num|cat)__", "", n).split("_")[0] for n in names]
    imp = (pd.DataFrame({"feature": pretty, "importance": clf.feature_importances_})
           .groupby("feature", as_index=False)["importance"].sum()
           .sort_values("importance").tail(14))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.barh(imp["feature"], imp["importance"], color="#54A24B")
    ax.set_xlabel("Summed importance")
    ax.set_title("Tree model — top features", fontsize=12, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. 主流程
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle Titanic 二分类")
    ap.add_argument("--cv", type=int, default=5, help="交叉验证折数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--model", type=str, default=None,
                    help="只训练指定模型（key 见 --list-models）")
    ap.add_argument("--list-models", action="store_true", help="列出可用模型")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.list_models:
        for n in make_models(args.seed):
            print(n)
        return

    train, test = load_data()

    # 训练集与测试集必须走同一套特征工程：先拼接再加工，避免分布不一致
    n_train = len(train)
    combined = pd.concat([train.drop(columns=["Survived"]), test],
                         ignore_index=True)
    combined = impute(build_features(combined))

    y = train["Survived"].astype(int)
    X = combined.iloc[:n_train].reset_index(drop=True)
    X_test = combined.iloc[n_train:].reset_index(drop=True)

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    print(f"[特征] 共 {X.shape[1]} 列：{', '.join(X.columns)}")

    models = make_models(args.seed)
    if args.model:
        if args.model not in models:
            raise SystemExit(f"未知模型 {args.model}，可选：{list(models)}")
        models = {args.model: models[args.model]}

    # 先给每个单模型打分，再把最好的几个集成起来同场比较
    cv_df = evaluate(models, X, y, args.cv, args.seed)
    if not args.model:
        ens_name, ens_model = build_ensemble(cv_df, models)
        models[ens_name] = ens_model
        cv_df = pd.concat(
            [cv_df, evaluate({ens_name: ens_model}, X, y, args.cv, args.seed)],
            ignore_index=True,
        ).sort_values("cv_mean", ascending=False).reset_index(drop=True)

    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)
    plot_scores(cv_df)

    best_name = cv_df.iloc[0]["model"]
    best = models[best_name]
    print(f"\n[选择] 最优模型 {best_name}（CV {cv_df.iloc[0]['cv_mean']:.4f}）")

    # 交叉验证的样本外预测，用于看混淆矩阵 / 分类报告
    splitter = StratifiedKFold(n_splits=args.cv, shuffle=True,
                               random_state=args.seed)
    oof = cross_val_predict(best, X, y, cv=splitter, n_jobs=-1)
    print(f"[样本外] 准确率 {accuracy_score(y, oof):.4f}")
    print("[混淆矩阵] 行=真实 0/1，列=预测 0/1")
    print(confusion_matrix(y, oof))
    print(classification_report(y, oof, target_names=["遇难", "生还"],
                                digits=4))

    # 全量训练集重训 -> 预测测试集
    best.fit(X, y)
    pred = best.predict(X_test).astype(int)
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"].astype(int),
        "Survived": pred,
    })
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    check_submission(submission, test["PassengerId"])
    print(f"[输出] {OUT_DIR / 'submission.csv'}  {submission.shape}，"
          f"预测生还 {int(pred.sum())} 人（{pred.mean():.2%}）")

    # 性别基线：Kaggle 自带的 gender_submission 就等价于「女性全生还」
    gender_pred = (test["Sex"] == "female").astype(int)
    agree = (gender_pred.values == pred).mean()
    print(f"[对照] 性别基线预测生还 {int(gender_pred.sum())} 人，"
          f"与模型结论一致率 {agree:.2%}")

    if hasattr(best.named_steps["clf"], "feature_importances_"):
        plot_importance(best)


def check_submission(submission: pd.DataFrame, test_ids: pd.Series) -> None:
    """按 Kaggle 的提交格式要求做自检，防止交出格式错误的文件。"""
    assert list(submission.columns) == ["PassengerId", "Survived"], "列名必须为 PassengerId,Survived"
    assert len(submission) == len(test_ids), f"行数应为 {len(test_ids)}，实际 {len(submission)}"
    assert submission["PassengerId"].tolist() == test_ids.astype(int).tolist(), \
        "PassengerId 必须与 test.csv 完全同序"
    assert set(submission["Survived"].unique()) <= {0, 1}, "Survived 只能取 0/1"
    assert submission["Survived"].dtype.kind in "iu", "Survived 必须是整数"
    print("[自检] 提交文件格式通过")


if __name__ == "__main__":
    main()
