@echo off
chcp 65001 >nul
echo ========================================
echo   企业 RAG 知识库系统 - Windows 启动脚本
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    pause
    exit /b 1
)

:: 创建必要目录
if not exist "data" mkdir data
if not exist "uploads" mkdir uploads
if not exist "templates" mkdir templates

:: 检查依赖
echo [1/4] 检查依赖...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖...
    pip install -r requirements.txt
)

:: 复制环境变量文件
if not exist ".env" (
    echo [2/4] 创建环境配置文件...
    copy ".env.example" ".env"
    echo [提示] 请编辑 .env 文件配置您的 API Key
)

:: 初始化向量数据库
echo [3/4] 初始化向量数据库...
python -c "from vector_service import EmbeddingService, VectorStore; vs = VectorStore(); vs.connect(); es = EmbeddingService(); es.load_model(); print('向量数据库初始化完成')" 2>nul
if errorlevel 1 (
    echo [提示] 首次运行会下载 Embedding 模型，请耐心等待...
    python -c "from vector_service import EmbeddingService, VectorStore; vs = VectorStore(); vs.connect(); es = EmbeddingService(); es.load_model(); print('模型下载完成')"
)

:: 启动服务
echo [4/4] 启动服务...
echo.
echo ========================================
echo   服务地址: http://localhost:8000
echo   打开浏览器访问开始使用
echo ========================================
echo.
echo 按 Ctrl+C 停止服务
echo.

python main.py

pause
