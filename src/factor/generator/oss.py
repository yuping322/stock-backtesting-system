"""
OSS 因子生成器

支持从 OSS 存储中加载预先计算好的因子文件

OSS 因子是用户自己计算并存储在 OSS（对象存储）中的因子数据。
格式：CSV 文件，包含 date, code, 和因子列（如 ALPHA158_001）
"""

import os
import sys
import pandas as pd
from typing import List, Optional, Dict
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
from src.data.data import factor_for_al

from src.factor.generator._base import FactorGenerator, format_factor_dataframe


class OSSFactorGenerator(FactorGenerator):
    """OSS 因子生成器 - 从真实 OSS 存储加载预计算因子"""
    
    def __init__(self, 
                 factor_names: List[str],
                 stock_codes: List[str],
                 start_date: str,
                 end_date: str,
                 output_dir: str = './data/factor_tasks'):
        """
        初始化 OSS 因子生成器
        
        Args:
            factor_names: OSS 因子名称列表 (如 ['PLRC24', 'BIAS10'])
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            output_dir: 输出目录
        """
        super().__init__(stock_codes, start_date, end_date, output_dir)
        
        # 验证因子名称 (移除前缀检查)
        # validate_factor_names(factor_names, 'oss')  # 移除限制，支持任意名称
        self.factor_names = factor_names
    
    def generate(self) -> pd.DataFrame:
        """
        从 OSS 加载所有指定因子数据
        
        Returns:
            pd.DataFrame: 包含所有因子的 DataFrame (date, stock_code, factor1, factor2, ...)
        """
        # 设置任务
        self.setup_task()
        
        print(f"\n开始生成 OSS 因子...")
        print(f"因子列表: {self.factor_names}")
        print(f"股票: {self.stock_codes}")
        print(f"日期: {self.start_date} ~ {self.end_date}")
        
        all_factor_dfs = []
        
        # 为每个因子从 OSS 加载数据
        for factor_name in self.factor_names:
            print(f"  加载因子: {factor_name}")
            try:
                # 使用参考代码模式从 OSS 加载
                factor_series = factor_for_al(
                    codes=self.stock_codes,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    factor_name=factor_name
                )
                
                if factor_series is not None and len(factor_series) > 0:
                    # 转换为长表: date, stock_code, factor_name
                    factor_df = factor_series.reset_index()
                    factor_df.columns = ['date', 'stock_code', factor_name]
                    all_factor_dfs.append(factor_df)
                    print(f"  ✓ {factor_name}: {len(factor_series)} 条数据")
                else:
                    print(f"  ⚠️ {factor_name}: OSS 中无数据")
                    
            except Exception as e:
                print(f"  ❌ {factor_name}: {e}")
                continue
        
        # 合并所有因子
        if not all_factor_dfs:
            print("❌ 未能从 OSS 加载任何因子数据")
            return pd.DataFrame()
        
        # 创建完整的时间网格，避免 NaN 值
        # 获取所有可能的日期和股票组合
        all_dates = []
        all_stocks = set()
        
        for df in all_factor_dfs:
            all_dates.extend(df['date'].unique())
            all_stocks.update(df['stock_code'].unique())
        
        all_dates = sorted(list(set(all_dates)))
        all_stocks = sorted(list(all_stocks))
        
        print(f"  创建完整网格: {len(all_dates)} 个日期 × {len(all_stocks)} 只股票 × {len(all_factor_dfs)} 个因子")
        
        # 创建完整网格的 DataFrame
        full_index = pd.MultiIndex.from_product([all_dates, all_stocks], names=['date', 'stock_code'])
        result = pd.DataFrame(index=full_index).reset_index()
        
        # 为每个因子添加列并填充数据
        for df in all_factor_dfs:
            factor_name = [col for col in df.columns if col not in ['date', 'stock_code']][0]
            # 创建临时 DataFrame 用于合并
            temp_df = df[['date', 'stock_code', factor_name]].copy()
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            
            # 合并到完整网格
            result = result.merge(temp_df, on=['date', 'stock_code'], how='left')
        
        # 重命名因子列，加上 jq_ 前缀
        factor_cols = [col for col in result.columns if col not in ['date', 'stock_code']]
        rename_dict = {col: f"jq_{col}" for col in factor_cols}
        result = result.rename(columns=rename_dict)
        
        # 规范化格式
        result = format_factor_dataframe(result)
        
        print(f"✓ OSS 因子生成完成，共 {len(result)} 条记录 (jq_ 前缀)")
        print(f"  覆盖因子: {len(factor_cols)} 个 (如 jq_PLRC24)")
        
        return result


def generate_oss_factors(
    factor_names: List[str],
    stock_codes: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = './data/factor_tasks'
) -> Dict[str, str]:
    """
    生成 OSS 因子（列名前缀: jq_XXX）
    
    factor_names: 必需，从 config/available_factors.txt 取 (PLRC24 → jq_PLRC24)
    
    修改说明：
    - 直接使用 factor_names 从 OSS 加载因子数据
    - 支持 config/available_factors.txt 中的任意因子名称
    - 匹配 factor_old/factor.py 中的 data.factor_for_al 调用模式
    
    返回值同前
    """
    # 创建生成器
    generator = OSSFactorGenerator(
        factor_names=factor_names,
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    # 生成因子
    factor_df = generator.generate()
    
    if factor_df.empty:
        raise Exception("OSS 因子生成失败（所有指定因子在 OSS 中无数据）")
    
    # 保存因子
    factor_file_out = generator.save_factors(factor_df)
    
    # 获取输出路径
    output_paths = generator.get_output_paths()
    
    print(f"\n✅ OSS 因子生成成功!")
    print(f"  因子文件: {output_paths['factor_file']}")
    print(f"  数据量: {len(factor_df)} 条记录")
    
    return output_paths
