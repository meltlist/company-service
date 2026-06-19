#!/bin/bash
# 企业 RAG 知识库系统 - Linux/Mac 启动脚本

set -e

echo "========================================"
echo "  企业 RAG 知识库系统 - 启动脚本"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python，请先安装 Python 3.11+"
    exit 1
fi

# 创建必要目录
mkdir -p data uploads templates

# 检查依赖
echo "[1/4] 检查依赖..."
if ! python3 -c "import fastapi" &> /dev/null; then
    echo "[提示] 正在安装依赖..."
    pip install -r requirements.txt
fi

# 复制环境变量文件
if [ ! -f ".env" ]; then
    echo "[2/4] 创建环境配置文件..."
    cp ".env.example" ".env"
    echo "[提示] 请编辑 .env 文件配置您的 API Key"
fi

# 初始化向量数据库
echo "[3/4] 初始化向量数据库..."
python3 -c "from vector_service import EmbeddingService, VectorStore; vs = VectorStore(); vs.connect(); es = EmbeddingService(); es.load_model(); print('向量数据库初始化完成')" 2>/dev/null || {
    echo "[提示] 首次运行会下载 Embedding 模型，请耐心等待..."
    python3 -c "from vector_service import EmbeddingService, VectorStore; vs = VectorStore(); vs.connect(); es = EmbeddingService(); es.load_model(); print('模型下载完成')"
}

# 启动服务
echo "[4/4] 启动服务..."
echo ""
echo "========================================"
echo "  服务地址: http://localhost:8000"
echo "  打开浏览器访问开始使用"
echo "========================================"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

python3 main.py
