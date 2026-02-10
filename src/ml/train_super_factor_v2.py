#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
八大机器学习模型大比拼（完整版）
参考：JoinQuant社区 - 八大机器学习模型大比拼
完整实现：数据获取、预处理、特征选择、多模型对比、完整评估
运行：python ml/train_super_factor_v2.py
"""
import pandas as pd
import numpy as np
from time import time
# from scipy.stats.mstats import winsorize  # 已自己实现winsorize_med
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import os
import sys
import json
import pickle
import warnings
from statistics import mean
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置pandas显示
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import data

# ========== 参数配置区 ==========
MODEL_DIR = "ml/models"
OUTPUT_DIR = "ml/output"

# 样本区间
START_DATE = '2025-10-15'
END_DATE = '2025-10-24'

# 股票池配置
INDEX = 'small'  # 小盘股指数代码

# 标签设置（前30%为1，后30%为0）
PERCENT_SELECT = [0.3, 0.3]

# 读入因子配置
FACTOR_CONFIG_FILE = os.path.join(project_root, "ml", "factor_config.json")

def load_factor_config():
    """从配置文件读取因子列表"""
    if os.path.exists(FACTOR_CONFIG_FILE):
        try:
            with open(FACTOR_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('factors', [])
        except:
            pass
    return []

# 默认因子列表（从已验证结果选择）
DEFAULT_FACTORS = [
    # 增长类
    "sales_growth", "operating_revenue_growth_rate", "total_profit_growth_rate",
    # "np_parent_company_owners_growth_rate", "total_asset_growth_rate",
    # "operating_profit_growth_rate", "growth", "SGI",
    # 盈利类
    "operating_cost_ttm", "total_operating_revenue_ttm", "gross_profit_ttm",
    # "total_profit_ttm", "net_profit_ttm", "EBITDA", "EBIT", "eps_ttm",
    # "operating_profit_per_share", "retained_earnings", "retained_earnings_per_share",
    # "non_recurring_gain_loss", "OperateNetIncome", "np_parent_company_owners_ttm",
    # "total_operating_cost_ttm", "administration_expense_ttm",
    # 现金流类
    "cashflow_per_share_ttm", "cash_flow_to_price_ratio",
    # "net_operate_cash_flow_per_share", "cash_and_equivalents_per_share",
    # "goods_sale_and_service_render_cash_ttm",
    # 规模类
    "market_cap",# "circulating_market_cap", "size", "natural_log_of_market_cap",
    # 技术指标
    "raw_beta", #"beta", "boll_down", "MAC20", "MAC10", "EMAC10", "EMAC12",
]

VALID_FACTORS = DEFAULT_FACTORS#load_factor_config() or DEFAULT_FACTORS

print(f"使用 {len(VALID_FACTORS)} 个因子")

# ========== 数据获取 ==========

def read_data_from_data_module(start_date: str, end_date: str) -> tuple:
    """
    从data.py模块读取数据和价格数据
    返回: (因子数据, 价格数据)
    """
    print(f"\n正在加载数据: {start_date} ~ {end_date}")
    
    # 获取股票池
    stocks = data.get_index_stocks(INDEX)
    print(f"股票池: {len(stocks)} 只股票")
    
    # 读取因子数据
    print("读取因子数据...")
    factor_df = data.read_factor_data(
        codes=stocks,
        start_date=start_date,
        end_date=end_date,
        factors=VALID_FACTORS,
        base_path="uploads"
    )
    
    if factor_df.empty:
        raise ValueError("未读取到因子数据，请检查数据路径")
    
    print(f"因子数据形状: {factor_df.shape}")
    
    # 读取价格数据
    print("读取价格数据...")
    price_df = data.load_oss_stocks(
        stocks,
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date)
    )
    
    if price_df.empty:
        raise ValueError("未读取到价格数据")
    
    print(f"价格数据形状: {price_df.shape}")
    
    return factor_df, price_df


def winsorize_med(df: pd.DataFrame, scale: float = 5, inf2nan: bool = False, axis: int = 0) -> pd.DataFrame:
    """中位数去极值"""
    def _winsor(s):
        if inf2nan:
            s = s.replace([np.inf, -np.inf], np.nan)
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0:
            mad = 1e-5
        return s.clip(lower=median - scale*mad, upper=median + scale*mad)
    return df.apply(_winsor, axis=axis)


def standardlize(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """标准化"""
    return (df - df.mean(axis=axis)) / df.std(axis=axis)


def neutralize(df: pd.DataFrame, style_cols: list, date: str, axis: int = 0) -> pd.DataFrame:
    """风格中性化（简化版）"""
    res = df.copy()
    for col in style_cols:
        if col in df.columns:
            res[col] = df[col] - df[col].mean()
    return res


def factor_processing(df: pd.DataFrame) -> pd.DataFrame:
    """完整的因子处理流程"""
    # 1. 中位数去极值
    df = winsorize_med(df, scale=5, inf2nan=False, axis=0)
    
    # 2. 标准化
    df = standardlize(df, axis=0)
    
    # 3. 移除缺失值
    df = df.fillna(df.mean())
    
    return df


def calculate_returns_and_merge(factor_df: pd.DataFrame, price_df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """计算收益率并合并到因子数据"""
    print(f"计算{period}天后收益率...")
    
    # 确保price_df的索引是date
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
        price_df.index.name = 'date'
    
    # 统一股票代码格式（去掉后缀）
    def clean_code(codes):
        """清理代码格式"""
        if isinstance(codes, pd.Index):
            return codes.str.replace('.XSHG', '').str.replace('.XSHE', '')
        else:
            return pd.Series(codes).str.replace('.XSHG', '').str.replace('.XSHE', '')
    
    # 计算未来收益率
    future_price = price_df.shift(-period)
    rets = future_price / price_df - 1
    
    # 转换为MultiIndex格式
    rets_series = rets.stack()
    rets_df = pd.DataFrame({'return': rets_series})
    
    # 清理factor_df的代码格式
    factor_codes = factor_df.index.get_level_values(1)
    factor_df.index = pd.MultiIndex.from_frame(pd.DataFrame({
        'date': factor_df.index.get_level_values(0),
        'code': clean_code(factor_codes)
    }))
    
    rets_df.index = pd.MultiIndex.from_frame(pd.DataFrame({
        'date': rets_df.index.get_level_values(0),
        'code': clean_code(rets_df.index.get_level_values(1))
    }))
    
    # 合并
    result = factor_df.join(rets_df, how='left')
    
    return result


def create_labels(data: pd.DataFrame, dropna: bool = True):
    """创建标签：前30%为1，后30%为0"""
    ret = data['return']
    ret_sorted = ret.sort_values(ascending=False)
    n_total = len(ret_sorted)
    
    n_top = int(np.around(n_total * PERCENT_SELECT[0]))
    n_bottom = int(np.around(n_total * PERCENT_SELECT[1]))
    
    labels = pd.Series(index=ret_sorted.index, dtype=float)
    labels.iloc[:n_top] = 1
    labels.iloc[-n_bottom:] = 0
    
    # 创建新的DataFrame只包含因子列
    features = data.drop('return', axis=1)
    
    # 对齐标签和特征
    valid_idx = labels.index
    features_labeled = features.loc[valid_idx]
    labels_aligned = labels.loc[valid_idx]
    
    if dropna:
        labels_aligned = labels_aligned.dropna()
        features_labeled = features_labeled.loc[labels_aligned.index]
    
    return features_labeled, labels_aligned


# ========== 特征选择 ==========

def feature_selection_f_test(X: pd.DataFrame, y: pd.Series, p_threshold: float = 0.01):
    """F检验特征选择"""
    F, p_values = f_classif(X, y)
    k = X.shape[1] - (p_values > p_threshold).sum()
    selector = SelectKBest(f_classif, k=k).fit(X, y)
    print(f"  F检验选择 {k}/{X.shape[1]} 个特征 (p<{p_threshold})")
    return selector


def feature_selection_mutual_info(X: pd.DataFrame, y: pd.Series, threshold: float = 0):
    """互信息特征选择"""
    result = mutual_info_classif(X, y)
    k = X.shape[1] - sum(result <= threshold)
    selector = SelectKBest(mutual_info_classif, k=k).fit(X, y)
    print(f"  互信息选择 {k}/{X.shape[1]} 个特征 (MI>{threshold})")
    return selector


def feature_selection_random_forest(X: pd.DataFrame, y: pd.Series, threshold: float = 0.005):
    """随机森林特征重要性"""
    rfc = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    rfc.fit(X, y)
    selector = SelectFromModel(rfc, threshold=threshold).fit(X, y)
    k = sum(selector.get_support())
    print(f"  随机森林选择 {k}/{X.shape[1]} 个特征 (importance>{threshold})")
    return selector


def feature_selection_rfe(X: pd.DataFrame, y: pd.Series, n_features: int = 20):
    """递归特征消除法"""
    rfc = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0)
    rfe = RFE(rfc, n_features_to_select=n_features, step=1)
    rfe.fit(X, y)
    print(f"  RFE选择 {n_features} 个特征")
    return rfe


# ========== 模型训练 ==========

def train_all_models(X: pd.DataFrame, y: pd.Series, selector=None):
    """训练所有模型并对比"""
    models = {}
    scores = {}
    
    # 准备数据
    if selector is not None:
        X_selected = selector.transform(X)
        print(f"\n使用特征选择，数据维度: {X_selected.shape}")
    else:
        X_selected = X
        print(f"\n不使用特征选择，数据维度: {X_selected.shape}")
    
    models_trained = []
    
    # 1. KNN
    print("\n1. 训练 KNN...")
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_selected, y)
    models['KNN'] = knn
    cv_acc = cross_val_score(knn, X_selected, y, scoring='accuracy', cv=5).mean()
    cv_auc = cross_val_score(knn, X_selected, y, scoring='roc_auc', cv=5).mean()
    scores['KNN'] = {'accuracy': cv_acc, 'auc': cv_auc}
    print(f"  Accuracy: {cv_acc:.4f}, AUC: {cv_auc:.4f}")
    
    # 2. 逻辑回归
    print("\n2. 训练 逻辑回归...")
    lg = LogisticRegression(C=1000, max_iter=300, penalty='l2', random_state=42)
    lg.fit(X_selected, y)
    models['LogisticRegression'] = lg
    cv_acc = cross_val_score(lg, X_selected, y, scoring='accuracy', cv=5).mean()
    cv_auc = cross_val_score(lg, X_selected, y, scoring='roc_auc', cv=5).mean()
    scores['LogisticRegression'] = {'accuracy': cv_acc, 'auc': cv_auc}
    print(f"  Accuracy: {cv_acc:.4f}, AUC: {cv_auc:.4f}")
    
    # 3. 决策树
    print("\n3. 训练 决策树...")
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=30, min_samples_split=2, random_state=42)
    tree.fit(X_selected, y)
    models['DecisionTree'] = tree
    cv_acc = cross_val_score(tree, X_selected, y, scoring='accuracy', cv=5).mean()
    cv_auc = cross_val_score(tree, X_selected, y, scoring='roc_auc', cv=5).mean()
    scores['DecisionTree'] = {'accuracy': cv_acc, 'auc': cv_auc}
    print(f"  Accuracy: {cv_acc:.4f}, AUC: {cv_auc:.4f}")
    
    # 4. 朴素贝叶斯
    print("\n4. 训练 朴素贝叶斯...")
    nb = GaussianNB()
    nb.fit(X_selected, y)
    models['GaussianNB'] = nb
    cv_acc = cross_val_score(nb, X_selected, y, scoring='accuracy', cv=5).mean()
    cv_auc = cross_val_score(nb, X_selected, y, scoring='roc_auc', cv=5).mean()
    scores['GaussianNB'] = {'accuracy': cv_acc, 'auc': cv_auc}
    print(f"  Accuracy: {cv_acc:.4f}, AUC: {cv_auc:.4f}")
    
    # 5. 随机森林
    print("\n5. 训练 随机森林...")
    rfc = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20, min_samples_split=20, random_state=42)
    rfc.fit(X_selected, y)
    models['RandomForest'] = rfc
    cv_acc = cross_val_score(rfc, X_selected, y, scoring='accuracy', cv=5).mean()
    cv_auc = cross_val_score(rfc, X_selected, y, scoring='roc_auc', cv=5).mean()
    scores['RandomForest'] = {'accuracy': cv_acc, 'auc': cv_auc}
    print(f"  Accuracy: {cv_acc:.4f}, AUC: {cv_auc:.4f}")
    
    # 6. AdaBoost
    print("\n6. 训练 AdaBoost...")
    adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), learning_rate=0.01, n_estimators=200, random_state=42)
    adb.fit(X_selected, y)
    models['AdaBoost'] = adb
    cv_acc = cross_val_score(adb, X_selected, y, scoring='accuracy', cv=5).mean()
    cv_auc = cross_val_score(adb, X_selected, y, scoring='roc_auc', cv=5).mean()
    scores['AdaBoost'] = {'accuracy': cv_acc, 'auc': cv_auc}
    print(f"  Accuracy: {cv_acc:.4f}, AUC: {cv_auc:.4f}")
    
    # 7. SVM
    print("\n7. 训练 SVM...")
    svm = SVC(C=0.003, kernel='rbf', gamma=0.01, probability=True, random_state=42)
    svm.fit(X_selected, y)
    models['SVM'] = svm
    cv_acc = cross_val_score(svm, X_selected, y, scoring='accuracy', cv=5).mean()
    cv_auc = cross_val_score(svm, X_selected, y, scoring='roc_auc', cv=5).mean()
    scores['SVM'] = {'accuracy': cv_acc, 'auc': cv_auc}
    print(f"  Accuracy: {cv_acc:.4f}, AUC: {cv_auc:.4f}")
    
    # 8. LightGBM
    print("\n8. 训练 LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_selected, y)
    models['LightGBM'] = lgb_model
    cv_acc = cross_val_score(lgb_model, X_selected, y, scoring='accuracy', cv=5).mean()
    cv_auc = cross_val_score(lgb_model, X_selected, y, scoring='roc_auc', cv=5).mean()
    scores['LightGBM'] = {'accuracy': cv_acc, 'auc': cv_auc}
    print(f"  Accuracy: {cv_acc:.4f}, AUC: {cv_auc:.4f}")
    
    return models, scores


# ========== 主函数 ==========

def main():
    print("=" * 60)
    print("八大机器学习模型大比拼（完整版）")
    print("=" * 60)
    print(f"因子数量: {len(VALID_FACTORS)}")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print(f"股票池: {INDEX}")
    print(f"标签策略: 前{PERCENT_SELECT[0]*100}%为1，后{PERCENT_SELECT[1]*100}%为0")
    print("=" * 60)
    
    # 1. 读取数据
    factor_df, price_df = read_data_from_data_module(START_DATE, END_DATE)
    
    # 2. 合并数据并计算收益率
    print("\n合并数据和计算收益率...")
    all_data_df = calculate_returns_and_merge(factor_df, price_df, period=10)
    
    # 4. 因子预处理
    print("\nStep-1: 因子预处理 ...")
    features = all_data_df.drop('return', axis=1)
    returns = all_data_df['return']
    
    # 逐日处理因子
    features_processed = []
    dates_list = sorted(features.index.get_level_values(0).unique())
    
    for date in dates_list:
        date_features = features.loc[features.index.get_level_values(0) == date]
        if len(date_features) < 10:  # 至少10个样本
            continue
        date_processed = factor_processing(date_features)
        features_processed.append(date_processed)
    
    if not features_processed:
        raise ValueError("没有有效的训练数据")
    
    features_final = pd.concat(features_processed)
    
    # 5. 创建标签（逐日处理）
    print("\nStep-2: 创建标签 ...")
    labels_list = []
    
    for date in dates_list:
        date_data = all_data_df.loc[all_data_df.index.get_level_values(0) == date]
        if len(date_data) < 10:
            continue
        if 'return' not in date_data.columns:
            continue
        
        date_labeled, date_labels = create_labels(date_data, dropna=True)
        if len(date_labeled) > 0:
            labels_list.append((date_labeled, date_labels))
    
    if not labels_list:
        raise ValueError("无法创建标签数据")
    
    # 合并所有日期数据
    all_labeled = pd.concat([x for x, y in labels_list])
    all_labels = pd.concat([y for x, y in labels_list])
    
    # 对齐索引
    valid_idx = all_labeled.index.intersection(all_labels.index)
    X = all_labeled.loc[valid_idx]
    y = all_labels.loc[valid_idx]
    
    print(f"训练样本数: {len(X)}")
    print(f"标签分布: 1={sum(y==1)}, 0={sum(y==0)}")
    
    # 6. 特征选择
    print("\nStep-3: 特征选择 ...")
    selector = feature_selection_rfe(X, y, n_features=20)
    X_selected = selector.transform(X)
    
    # 7. 训练所有模型
    print("\nStep-4: 训练8个模型 ...")
    models, scores = train_all_models(X, y, selector)
    
    # 8. 模型对比结果
    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)
    scores_df = pd.DataFrame(scores).T
    scores_df.index.name = 'Model'
    print(scores_df.sort_values('auc', ascending=False))
    
    # 保存对比结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scores_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
    
    # 9. 选择最佳模型
    best_model_name = scores_df['auc'].idxmax()
    best_model = models[best_model_name]
    print(f"\n最佳模型: {best_model_name}")
    print(f"  Accuracy: {scores_df.loc[best_model_name, 'accuracy']:.4f}")
    print(f"  AUC: {scores_df.loc[best_model_name, 'auc']:.4f}")
    
    # 10. 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 保存所有模型
    for name, model in models.items():
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # 保存特征选择器
    selector_path = os.path.join(MODEL_DIR, "feature_selector.pkl")
    with open(selector_path, 'wb') as f:
        pickle.dump(selector, f)
    
    # 保存模型信息
    model_info = {
        "best_model": best_model_name,
        "all_scores": scores,
        "feature_count": X_selected.shape[1],
        "sample_count": len(X),
        "train_date_range": f"{START_DATE} ~ {END_DATE}",
        "stock_pool": INDEX,
        "percent_select": PERCENT_SELECT,
        "factors": VALID_FACTORS
    }
    
    with open(os.path.join(MODEL_DIR, "model_info.json"), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print("\n✅ 训练完成！")
    print(f"所有模型保存在: {MODEL_DIR}")
    print(f"对比结果保存在: {OUTPUT_DIR}/model_comparison.csv")


if __name__ == "__main__":
    main()
