#!/bin/bash

echo "正在启动股票回测系统..."

# 加载安全凭证（如果存在）
if [ -f "env.sh" ]; then
    source env.sh
    echo "安全凭证已加载"
else
    echo "警告: env.sh 文件不存在，请创建并配置安全凭证"
fi

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 检查Python
python3 -c "import sys; print('Python版本:', sys.version)"
if [ $? -ne 0 ]; then
    echo "错误: Python3不可用"
    exit 1
fi

# 安装依赖
echo "安装依赖包..."
pip3 install -r requirements.txt

# 创建目录
mkdir -p logs data

# 运行Streamlit应用
echo "启动Streamlit应用..."
streamlit run integrated_backtesting_system.py --server.port 8501 --server.address 0.0.0.0

echo "应用启动完成"