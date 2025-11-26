import akshare as ak
import pandas as pd

try:
    print("尝试获取申万一级行业列表...")
    # 获取申万一级行业列表
    # 注意：接口名称可能会随版本变化，这里尝试常见的接口
    df_index = ak.index_classify_sw(level="L1")
    print(f"成功获取 {len(df_index)} 个一级行业")
    print(df_index.head())
    
    print("\n尝试获取某个行业的成分股 (例如: 计算机)...")
    # 假设第一个是行业代码
    if not df_index.empty:
        industry_code = df_index.iloc[0]['index_code']
        print(f"获取行业 {industry_code} 的成分股...")
        # 注意：这里需要确认获取成分股的正确接口
        # 尝试 stock_board_industry_cons_sw
        # df_cons = ak.stock_board_industry_cons_sw(symbol=industry_code) # 可能需要具体名称
        pass

except Exception as e:
    print(f"AKShare 接口测试失败: {e}")
    print("尝试搜索相关接口...")
    # 在实际开发中，我会查阅文档，这里先打印版本
    print(f"AKShare version: {ak.__version__}")
