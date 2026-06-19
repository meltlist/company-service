# 企业 RAG 知识库系统

支持企业权限分级使用的知识库问答系统，基于 RAG 原理构建。

## 功能特性

- **多级权限管理**：超级管理员、企业管理员、部门经理、普通成员
- **文档管理**：支持 PDF、Word、Excel 文件上传和管理
- **智能分块**：自动识别标题结构，支持图片、表格识别
- **向量检索**：Qdrant 向量数据库，支持混合检索（向量 + 关键词）
- **智能问答**：基于检索结果生成回答，支持引用来源标注
- **多 LLM 支持**：DeepSeek、Gemini 等多种模型，由企业管理员配置
- **Token 统计**：记录每个账号消耗的 Token

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制 `.env.example` 为 `.env`，修改其中的配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入您的 API Key：

```env
# DeepSeek API Key（用于问答）
DEEPSEEK_API_KEY=your-api-key-here
```

### 3. 启动服务

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

### 4. 访问

打开浏览器访问 http://localhost:8000

首次注册会成为企业管理员，可以配置 LLM API 和管理企业成员。

## 项目结构

```
├── main.py              # FastAPI 主应用
├── models.py            # 数据库模型
├── auth.py              # 认证模块
├── config.py            # 配置管理
├── document_processor.py # 文档处理
├── vector_service.py    # 向量检索
├── llm_service.py       # LLM 服务
├── templates/          # 前端模板
├── requirements.txt     # Python 依赖
└── run.bat/run.sh       # 启动脚本
```

## API 接口

### 认证
- `POST /api/auth/register` - 注册
- `POST /api/auth/login` - 登录
- `GET /api/auth/me` - 当前用户

### 文档
- `POST /api/documents/upload` - 上传文档
- `POST /api/documents/{id}/process` - 处理文档
- `GET /api/documents` - 文档列表
- `DELETE /api/documents/{id}` - 删除文档

### 知识库
- `POST /api/knowledge-bases` - 创建知识库
- `GET /api/knowledge-bases` - 知识库列表
- `POST /api/knowledge-bases/{id}/documents` - 添加文档到知识库

### 问答
- `POST /api/chat` - 智能问答

### 管理
- `GET /api/admin/settings` - 企业设置
- `PUT /api/admin/settings` - 更新设置
- `GET /api/admin/users` - 用户列表
- `GET /api/admin/token-usage` - Token 使用统计

## 技术栈

- **后端**：FastAPI + SQLAlchemy
- **数据库**：SQLite（本地）+ Qdrant（向量）
- **前端**：HTML + TailwindCSS + JavaScript
- **认证**：JWT + PBKDF2
- **文档处理**：PyMuPDF、python-docx、openpyxl、RapidOCR
