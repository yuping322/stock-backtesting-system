#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用训练好的模型预测选股
运行：python ml/predict_stocks.py --date 2024-10-24 --top-n 10
"""
import pandas as pd
import numpy as np
from scipy.stats import winsorize
import lightgbm as lgb
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import data

# ---------- 配置 ----------
MODEL_DIR = "ml/models"
VALID_FACTORS = [
    "sales_growth", "operating_revenue_growth_rate", "total_profit_growth_rate",
    "np_parent_company_owners_growth_rate", "total_asset_growth_rate",
    "operating_profit_growth_rate", "growth", "SGI",
    "operating_cost_ttm", "total_operating_revenue_ttm", "gross_profit_ttm",
    "total_profit_ttm", "net_profit_ttm", "EBITDA", "EBIT", "eps_ttm",
    "operating_profit_per_share", "retained_earnings", "retained_earnings_per_share",
    "non_recurring_gain_loss", "OperateNetIncome", "np_parent_company_owners_ttm",
    "total_operating_cost_ttm", "administration_expense_ttm",
    "cashflow_per_share_ttm", "cash_flow_to_price_ratio",
    "net_operate_cash_flow_per_share", "cash_and_equivalents_per_share",
    "goods_sale_and_service_render_cash_ttm",
    "market_cap", "circulating_market_cap", "size", "natural_log_of_market_cap",
    "raw_beta", "beta", "boll_down", "MAC20", "MAC10", "EMAC10", "EMAC12",
]
NEUTRAL_STYLE = ["market_cap", "beta"]

def load_model():
    """加载训练好的模型"""
    # 尝试加载pickle格式（v2版本）
    model_pkl_path = os.path.join(MODEL_DIR, "super_factor_model.pkl")
    feature_selector_path = os.path.join(MODEL_DIR, "feature_selector.pkl")
    
    # 尝试加载txt格式（基础版本）
    model_txt_path = os.path.join(MODEL_DIR, "super_factor_model.txt")
    ic_weights_path = os.path.join(MODEL_DIR, "ic_weights.parquet")
    
    info_path = os.path.join(MODEL_DIR, "model_info.json")
    
    # 加载模型信息
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"模型文件不存在，请先运行 train_super_factor.py")
    
    with open(info_path, 'r', encoding='utf-8') as f:
        import json
        model_info = json.load(f)
    
    selector = None
    ic_weights = None
    
    # 加载pickle格式模型（v2）
    if os.path.exists(model_pkl_path):
        print("  加载 v2 模型（pickle格式）...")
        with open(model_pkl_path, 'rb') as f:
            import pickle
            model = pickle.load(f)
        
        if os.path.exists(feature_selector_path):
            with open(feature_selector_path, 'rb') as f:
                selector = pickle.load(f)
    
    # 加载txt格式模型（基础版）
    elif os.path.exists(model_txt_path):
        print("  加载基础模型（txt格式）...")
        model = lgb.Booster(model_file=model_txt_path)
        
        if os.path.exists(ic_weights_path):
            ic_weights = pd.read_parquet(ic_weights_path)
    
    else:
        raise FileNotFoundError(f"未找到模型文件")
    
    return model, ic_weights, model_info, selector

def read_current_data(date: str, stock_pool: str = "small") -> pd.DataFrame:
    """读取当日因子数据"""
    print(f"正在加载数据: {date}")
    
    if stock_pool == "small":
        stocks = data.get_index_stocks("000510")
    else:
        stocks = data.get_index_stocks(stock_pool)
    
    print(f"股票池: {len(stocks)} 只股票")
    
    # 读取因子数据（前后各5天以确保数据完整）
    from datetime import datetime, timedelta
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    
    df = data.read_factor_data(
        codes=stocks,
        start_date=start_date,
        end_date=end_date,
        factors=VALID_FACTORS,
        base_path="uploads"
    )
    
    if df.empty:
        raise ValueError("未读取到因子数据")
    
    # 只取指定日期的数据
    df = df.loc[df.index.get_level_values(0) == date]
    
    if df.empty:
        raise ValueError(f"日期 {date} 无数据")
    
    print(f"数据形状: {df.shape}")
    
    # 添加style变量
    if 'market_cap' in df.columns:
        df['market_cap'] = df['market_cap'].fillna(df['market_cap'].median())
    if 'beta' not in df.columns:
        df['beta'] = 1.0
    
    return df

def clean_factor(s: pd.Series, df: pd.DataFrame) -> pd.Series:
    """单日横截面清洗"""
    def _winsor(x): 
        try:
            return winsorize(x, limits=[0.05, 0.05])
        except:
            return x
    
    s_winsor = _winsor(s)
    
    if 'market_cap' in df.columns:
        X = df[['market_cap']].fillna(0)
        y = s_winsor.reindex(X.index)
        valid_mask = y.notna() & X['market_cap'].notna()
        if valid_mask.sum() > 10:
            beta = np.linalg.lstsq(X[valid_mask], y[valid_mask], rcond=None)[0]
            res = y - X @ beta
        else:
            res = y
    else:
        res = s_winsor
    
    res_std = res.std()
    if res_std > 0:
        return (res - res.mean()) / res_std
    else:
        return res - res.mean()

def compute_super_factor(factors: pd.DataFrame, ic_weights: pd.DataFrame) -> pd.Series:
    """计算超级因子"""
    # 获取最新的IC权重
    latest_date = ic_weights.index.max()
    day_weights = ic_weights.loc[latest_date]
    
    # 加权合成
    super_factor = (factors * day_weights).sum(axis=1)
    
    return super_factor

def neutralize(factor: pd.Series, df: pd.DataFrame, style_cols: list) -> pd.Series:
    """风格中性化"""
    X = df[style_cols].fillna(0)
    y = factor.reindex(X.index)
    
    valid_idx = y.index.intersection(X.index)
    if len(valid_idx) > 10:
        y_sub = y.loc[valid_idx]
        X_sub = X.loc[valid_idx]
        
        try:
            beta = np.linalg.lstsq(X_sub, y_sub, rcond=None)[0]
            res = y_sub - X_sub @ beta
        except:
            res = y_sub
    else:
        res = y
    
    return res

def predict_top_stocks(model, X_data, top_n: int = 10) -> pd.DataFrame:
    """预测Top N股票"""
    # 确保X是DataFrame格式
    if isinstance(X_data, pd.Series):
        X = X_data.to_frame("factor")
    else:
        X = X_data
    
    # 模型预测
    if hasattr(model, 'predict_proba'):
        # sklearn模型
        predictions = model.predict_proba(X.values)[:, 1]
    elif hasattr(model, 'predict'):
        # LightGBM Booster或其他
        predictions = model.predict(X.values)
    else:
        raise ValueError("无法识别的模型类型")
    
    pred_series = pd.Series(predictions, index=X.index, name="score")
    
    # 选择Top N
    top_stocks = pred_series.nlargest(top_n)
    
    # 归一化为权重
    weights = top_stocks / top_stocks.sum()
    
    # 构建结果DataFrame
    result = pd.DataFrame({
        'code': weights.index.get_level_values(1),
        'weight': weights.values
    })
    
    return result

def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型预测选股')
    parser.add_argument('--date', type=str, default=None,
                       help='预测日期 (YYYY-MM-DD)，默认最新交易日')
    parser.add_argument('--top-n', type=int, default=10,
                       help='选择股票数量，默认10')
    parser.add_argument('--stock-pool', type=str, default='small',
                       help='股票池，默认small')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径，默认data/factor_values_sample.csv')
    
    args = parser.parse_args()
    
    # 确定日期
    if args.date is None:
        from datetime import datetime
        args.date = datetime.now().strftime("%Y-%m-%d")
    
    print("=" * 60)
    print("预测选股")
    print("=" * 60)
    print(f"预测日期: {args.date}")
    print(f"股票数量: {args.top_n}")
    print(f"股票池: {args.stock_pool}")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n加载模型...")
    model, ic_weights, model_info, selector = load_model()
    print(f"  ✓ 模型加载成功")
    print(f"  模型类型: {model_info.get('model_name', 'LightGBM')}")
    print(f"  训练期间: {model_info['train_date_range']}")
    if selector is not None:
        print(f"  使用特征选择: ✓")
    
    # 2. 读取当前数据
    print("\n读取数据...")
    df = read_current_data(args.date, args.stock_pool)
    
    factor_cols = [f for f in VALID_FACTORS if f in df.columns]
    print(f"  因子数: {len(factor_cols)}")
    
    factors = df[factor_cols]
    
    # 3. 清洗因子
    print("\n清洗因子...")
    factors_clean = factors.apply(lambda s: clean_factor(s, df))
    
    # 4. 应用特征选择（如果有）
    if selector is not None:
        print("应用特征选择...")
        factors_clean = pd.DataFrame(
            selector.transform(factors_clean),
            index=factors_clean.index,
            columns=[f'f{i}' for i in range(selector.transform(factors_clean).shape[1])]
        )
        X_for_pred = factors_clean
    else:
        # 使用传统方法：IC加权 + 风格中性化
        print("计算超级因子...")
        super_factor = compute_super_factor(factors_clean, ic_weights)
        
        print("风格中性化...")
        super_factor = neutralize(super_factor, df, NEUTRAL_STYLE)
        X_for_pred = super_factor.to_frame("linear")
    
    # 5. 预测Top N
    print(f"\n预测Top {args.top_n}股票...")
    if isinstance(X_for_pred, pd.Series):
        result = predict_top_stocks(model, X_for_pred, args.top_n)
    else:
        result = predict_top_stocks(model, X_for_pred, args.top_n)
    
    # 7. 添加日期列
    result.insert(0, 'date', args.date)
    
    # 8. 保存结果
    if args.output is None:
        output_path = "data/factor_values_sample.csv"
    else:
        output_path = args.output
    
    # 检查是否存在旧文件，append mode
    if os.path.exists(output_path):
        # 读取旧数据
        old_df = pd.read_csv(output_path)
        # 如果同日期已有数据，先删除
        old_df = old_df[old_df['date'] != args.date]
        # 合并
        result = pd.concat([old_df, result], ignore_index=True)
    
    result.to_csv(output_path, index=False)
    
    print(f"\n✅ 预测完成！")
    print(f"结果已保存: {output_path}")
    print("\n选股结果:")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()

