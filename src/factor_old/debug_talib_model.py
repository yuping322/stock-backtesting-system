#!/usr/bin/env python3
"""
TALIB因子模型训练调试脚本
用于手动调试和验证模型建模流程
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import traceback

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_environment():
    """设置调试环境"""
    print("=== 环境设置 ===")

    # 激活虚拟环境
    venv_path = project_root / ".venv" / "bin" / "activate"
    if venv_path.exists():
        print(f"虚拟环境路径: {venv_path}")
    else:
        print("警告: 未找到虚拟环境")

    # 设置路径
    from factor_workflow import paths
    paths.DATA_ROOT = project_root / 'talib_export_small'
    paths.FEATURES_FILE = paths.DATA_ROOT / 'features_panel.pkl'
    paths.LABEL_FILE = paths.DATA_ROOT / 'label_panel.pkl'
    paths.META_FILE = paths.DATA_ROOT / 'meta_series.pkl'
    paths.IC_FILE = paths.DATA_ROOT / 'factor_ic_daily.pkl'

    print(f"数据根目录: {paths.DATA_ROOT}")
    print(f"特征文件: {paths.FEATURES_FILE}")
    print(f"标签文件: {paths.LABEL_FILE}")

    return True

def check_data_files():
    """检查数据文件是否存在"""
    print("\n=== 数据文件检查 ===")

    from factor_workflow import paths

    files_to_check = [
        ('特征数据', paths.FEATURES_FILE),
        ('标签数据', paths.LABEL_FILE),
        ('元数据', paths.META_FILE),
        ('IC数据', paths.IC_FILE)
    ]

    all_exist = True
    for name, file_path in files_to_check:
        if file_path.exists():
            print(f"✓ {name}: {file_path}")
        else:
            print(f"✗ {name}: {file_path} (不存在)")
            all_exist = False

    return all_exist

def load_and_validate_data():
    """加载并验证数据"""
    print("\n=== 数据加载与验证 ===")

    try:
        from factor_workflow import paths

        # 加载特征数据
        features_df = pd.read_pickle(paths.FEATURES_FILE)
        print(f"特征数据形状: {features_df.shape}")
        print(f"特征列数: {len(features_df.columns)}")
        print(f"特征名称: {list(features_df.columns)}")

        # 加载标签数据
        label_df = pd.read_pickle(paths.LABEL_FILE)
        print(f"标签数据形状: {label_df.shape}")

        # 加载IC数据
        ic_df = pd.read_pickle(paths.IC_FILE)
        print(f"IC数据形状: {ic_df.shape}")
        print(f"IC因子: {list(ic_df.columns)}")

        # 基本验证
        print("\n=== 数据验证 ===")
        print(f"特征数据null值: {features_df.isnull().sum().sum()}")
        print(f"标签数据null值: {label_df.isnull().sum()}")
        print(f"IC数据null值: {ic_df.isnull().sum().sum()}")

        # IC统计
        print("\nIC统计:")
        for col in ic_df.columns:
            mean_ic = ic_df[col].mean()
            std_ic = ic_df[col].std()
            print(f"  {col}: 均值={mean_ic:.4f}, 标准差={std_ic:.4f}")

        return features_df, label_df, ic_df

    except Exception as e:
        print(f"数据加载失败: {e}")
        traceback.print_exc()
        return None, None, None

def initialize_qlib():
    """初始化QLib"""
    print("\n=== QLib初始化 ===")

    try:
        import qlib

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        print(f"使用临时目录: {temp_dir}")

        # 初始化QLib
        qlib.init(provider_uri=temp_dir, region='cn')
        print("✓ QLib初始化成功")

        # 导入工作流模块
        from qlib.workflow import R
        print("✓ 工作流模块导入成功")

        return temp_dir

    except Exception as e:
        print(f"✗ QLib初始化失败: {e}")
        traceback.print_exc()
        return None

def create_datasets():
    """创建数据集"""
    print("\n=== 数据集创建 ===")

    try:
        from factor_workflow.dataset_config import long_dataset_config, short_dataset_config
        from qlib.utils import init_instance_by_config

        print("创建long数据集...")
        long_dataset = init_instance_by_config(long_dataset_config)
        print(f"✓ Long数据集创建成功: {type(long_dataset)}")

        print("创建short数据集...")
        short_dataset = init_instance_by_config(short_dataset_config)
        print(f"✓ Short数据集创建成功: {type(short_dataset)}")

        # 检查数据集基本信息
        print("\n数据集信息:")
        try:
            print(f"Long数据集类型: {type(long_dataset)}")
            print(f"Short数据集类型: {type(short_dataset)}")
            # 尝试获取数据集长度（如果可用）
            if hasattr(long_dataset, '__len__'):
                print(f"Long数据集长度: {len(long_dataset)}")
            if hasattr(short_dataset, '__len__'):
                print(f"Short数据集长度: {len(short_dataset)}")
        except:
            print("数据集长度信息不可用")

        return long_dataset, short_dataset

    except Exception as e:
        print(f"✗ 数据集创建失败: {e}")
        traceback.print_exc()
        return None, None

def train_models(long_dataset, short_dataset):
    """训练模型"""
    print("\n=== 模型训练 ===")

    try:
        from factor_workflow.model_pipeline import train_model_suite
        from factor_workflow.models_config import fusion_config, long_model_specs, short_model_specs
        from qlib.workflow import R

        # 训练long模型
        print("训练long模型...")
        with R.start(experiment_name='debug_long_suite'):
            long_suite = train_model_suite('long', long_dataset, long_model_specs, fusion_config['long'])
            print("✓ Long模型训练完成")

        # 训练short模型
        print("训练short模型...")
        with R.start(experiment_name='debug_short_suite'):
            short_suite = train_model_suite('short', short_dataset, short_model_specs, fusion_config['short'])
            print("✓ Short模型训练完成")

        # 分析结果
        pred_long = long_suite.fused_prediction
        pred_short = short_suite.fused_prediction

        print("\n=== 训练结果分析 ===")
        print(f"Long预测样本数: {len(pred_long)}")
        print(f"Short预测样本数: {len(pred_short)}")

        print(f"Long预测统计: 均值={pred_long.mean():.6f}, 标准差={pred_long.std():.6f}")
        print(f"Short预测统计: 均值={pred_short.mean():.6f}, 标准差={pred_short.std():.6f}")

        print(f"Long模型数量: {len(long_suite.model_results)}")
        for i, res in enumerate(long_suite.model_results):
            print(f"  模型{i+1}: {res.name}")

        print(f"Short模型数量: {len(short_suite.model_results)}")
        for i, res in enumerate(short_suite.model_results):
            print(f"  模型{i+1}: {res.name}")

        return long_suite, short_suite

    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        traceback.print_exc()
        return None, None

def save_results(long_suite, short_suite):
    """保存结果"""
    print("\n=== 结果保存 ===")

    try:
        import pandas as pd
        from qlib.workflow import R

        # 创建输出目录
        output_dir = project_root / 'debug_model_results'
        output_dir.mkdir(exist_ok=True)

        # 保存预测结果
        pred_long = long_suite.fused_prediction
        pred_short = short_suite.fused_prediction

        pred_long.to_pickle(output_dir / 'predictions_long.pkl')
        pred_short.to_pickle(output_dir / 'predictions_short.pkl')

        # 保存统计信息
        with open(output_dir / 'debug_summary.txt', 'w', encoding='utf-8') as f:
            f.write('=== TALIB因子模型调试结果 ===\n')
            f.write(f'Long预测样本数: {len(pred_long)}\n')
            f.write(f'Short预测样本数: {len(pred_short)}\n')
            f.write(f'Long预测均值: {pred_long.mean():.6f}\n')
            f.write(f'Short预测均值: {pred_short.mean():.6f}\n')
            f.write(f'Long模型数量: {len(long_suite.model_results)}\n')
            f.write(f'Short模型数量: {len(short_suite.model_results)}\n')

        print(f"✓ 结果已保存到: {output_dir}")
        return True

    except Exception as e:
        print(f"✗ 结果保存失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主调试函数"""
    print("🚀 开始TALIB因子模型训练调试\n")

    # 步骤1: 环境设置
    if not setup_environment():
        return

    # 步骤2: 检查数据文件
    if not check_data_files():
        print("❌ 数据文件不完整，请先运行数据准备流程")
        return

    # 步骤3: 加载并验证数据
    features_df, label_df, ic_df = load_and_validate_data()
    if features_df is None:
        return

    # 步骤4: 初始化QLib
    temp_dir = initialize_qlib()
    if temp_dir is None:
        return

    try:
        # 步骤5: 创建数据集
        long_dataset, short_dataset = create_datasets()
        if long_dataset is None or short_dataset is None:
            return

        # 步骤6: 训练模型
        long_suite, short_suite = train_models(long_dataset, short_dataset)
        if long_suite is None or short_suite is None:
            return

        # 步骤7: 保存结果
        if save_results(long_suite, short_suite):
            print("\n🎉 调试完成！所有步骤成功执行")

    except Exception as e:
        print(f"\n❌ 调试过程中发生错误: {e}")
        traceback.print_exc()

    finally:
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"清理临时目录: {temp_dir}")

if __name__ == "__main__":
    main()