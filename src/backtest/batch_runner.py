import os
import glob
import pandas as pd
import argparse
from datetime import datetime
from src.backtest.main import run_backtest
from src.backtest.config import SystemConfig

def run_batch_analysis(input_dir, top_n_strategies=3):
    """
    批量运行回测，并分析结果
    """
    # 1. 扫描所有 CSV 文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"在 {input_dir} 没有找到 CSV 文件")
        return

    print(f"找到 {len(csv_files)} 个策略文件，开始批量回测...")
    
    results = []
    strategy_holdings = {} # 记录每个策略的最新持仓

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n>>> 正在回测: {file_name}")
        
        # 构造参数模拟命令行参数
        class Args:
            data_file = file_path
            strategy = "direct_execution" # 默认使用直接执行
            benchmark = "sh000300"
            initial_cash = 1_000_000
            commission = 0.0002
            slippage = 0.0
            hold_days = None
            top_n = None
            start_date = None
            end_date = None
            output_dir = None # 不生成详细文件夹，只在内存处理
            
        try:
            # 运行回测
            # 注意：run_backtest 会自动创建输出目录，为了避免垃圾文件，
            # 我们可以指定一个临时目录或者后续清理，这里简化处理直接运行
            system_config, result = run_backtest(Args())
            
            # 收集核心指标
            metrics = result.performance
            total_return = (result.final_value - 1_000_000) / 1_000_000
            
            # 获取夏普比率 (如果计算失败则为0)
            try:
                sharpe = metrics.loc['sharpe_ratio', 'value']
            except:
                sharpe = 0.0
                
            # 获取最大回撤
            try:
                max_dd = metrics.loc['max_drawdown', 'value']
            except:
                max_dd = 0.0

            results.append({
                "Strategy": file_name,
                "Total Return": total_return,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_dd,
                "Final Value": result.final_value
            })
            
            # 获取该策略最新一天的持仓（用于后续共识分析）
            # 读取原始 CSV 找最后一天
            df = pd.read_csv(file_path)
            if not df.empty and 'date' in df.columns:
                latest_date = df['date'].max()
                latest_picks = df[df['date'] == latest_date]['code'].tolist()
                strategy_holdings[file_name] = set(latest_picks)
                
        except Exception as e:
            print(f"回测失败 {file_name}: {e}")

    # 2. 生成排行榜
    if not results:
        return

    df_results = pd.DataFrame(results)
    # 按夏普比率排序
    df_results = df_results.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*50)
    print("🚀 策略排行榜 (按夏普比率)")
    print("="*50)
    # 格式化输出
    print(df_results.to_string(formatters={
        'Total Return': '{:.2%}'.format,
        'Sharpe Ratio': '{:.3f}'.format,
        'Max Drawdown': '{:.2%}'.format,
        'Final Value': '{:,.0f}'.format
    }))
    
    # 3. 寻找“共识股票” (Consensus Picks)
    # 只看排名前 N 的策略
    top_strategies = df_results.head(top_n_strategies)['Strategy'].tolist()
    print(f"\n" + "="*50)
    print(f"💎 靠谱股票推荐 (基于前 {len(top_strategies)} 名策略的共识)")
    print("="*50)
    
    stock_votes = {}
    for strat in top_strategies:
        picks = strategy_holdings.get(strat, set())
        for stock in picks:
            stock_votes[stock] = stock_votes.get(stock, 0) + 1
            
    # 排序：票数多的在前
    sorted_votes = sorted(stock_votes.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_votes:
        print("没有找到共识股票")
    else:
        print(f"{'股票代码':<10} | {'推荐次数':<10} | {'置信度':<10}")
        print("-" * 36)
        for stock, votes in sorted_votes:
            confidence = votes / len(top_strategies)
            if confidence >= 0.5: # 只显示超过一半策略推荐的
                print(f"{stock:<10} | {votes:<10} | {confidence:.0%}")

if __name__ == "__main__":
    # 假设你的 CSV 文件都在这个目录下
    # 你可以创建一个文件夹比如 'my_strategies' 专门放这些 csv
    INPUT_DIR = "." 
    
    # 为了演示，我们先生成几个模拟的策略文件
    # (实际使用时请注释掉这些生成代码)
    print("正在生成模拟策略文件以供演示...")
    import shutil
    if os.path.exists("my_strategy_predictions.csv"):
        shutil.copy("my_strategy_predictions.csv", "strategy_A.csv") # 原始策略
        
        # 策略 B: 稍微改一下权重，模拟另一个版本
        df = pd.read_csv("my_strategy_predictions.csv")
        df['weight'] = df['weight'] * 1.2 
        df.to_csv("strategy_B.csv", index=False)
        
        # 策略 C: 只选部分日期的，模拟择时策略
        df = pd.read_csv("my_strategy_predictions.csv")
        df = df.iloc[::2] # 隔天交易
        df.to_csv("strategy_C.csv", index=False)
    
    # 运行批量分析
    run_batch_analysis(INPUT_DIR)
    
    # 清理演示文件
    # for f in ["strategy_A.csv", "strategy_B.csv", "strategy_C.csv"]:
    #     if os.path.exists(f):
    #         os.remove(f)
