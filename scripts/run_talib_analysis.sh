#!/bin/bash
# TA-Lib 技术指标检验程序启动脚本

# 检查是否安装了 talib
python -c "import talib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ TA-Lib 未安装，请先安装：pip install TA-Lib"
    echo "注意：TA-Lib 需要系统依赖，macOS 请运行：brew install ta-lib"
    exit 1
fi

# 检查是否安装了 akshare
python -c "import akshare" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ AKShare 未安装，请先安装：pip install akshare"
    exit 1
fi

echo "� 开始 TA-Lib 技术指标检验..."

# 运行 Python 程序
python talib_indicator_analysis.py

echo ""
echo "✅ TA-Lib 技术指标检验完成！"