"""FastAPI 主应用"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from auth import (
    TokenData,
    check_permission,
    create_access_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from config import settings
from document_processor import DocumentProcessor, TextChunker, compute_file_hash
from llm_service import LLMService, RAGService
from models import (
    Department,
    Document,
    DocumentChunk,
    Enterprise,
    KnowledgeBase,
    KBDocument,
    TokenUsage,
    User,
    UserKnowledgeBase,
    UserRole,
    init_db,
)
from vector_service import (
    EmbeddingService,
    HybridSearch,
    QueryRewriter,
    VectorStore,
)

# ============= 初始化 =============
app = FastAPI(
    title="企业 RAG 知识库系统",
    description="支持多级权限的文档管理与智能问答",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库
engine, SessionLocal = init_db(settings.DATABASE_URL)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 服务实例
embedding_service = EmbeddingService()
vector_store = VectorStore()
query_rewriter = QueryRewriter()
doc_processor = DocumentProcessor()
text_chunker = TextChunker()


# ============= 依赖 =============
async def get_current_user_dep(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """获取当前登录用户"""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")

    token = auth[7:]
    user = get_current_user(db, token)
    if not user:
        raise HTTPException(status_code=401, detail="登录已过期")

    # 更新最后登录
    user.last_login = datetime.utcnow()
    db.commit()

    return user


# ============= 辅助函数 =============
def get_enterprise_llm(enterprise: Enterprise) -> LLMService:
    """获取企业的 LLM 配置"""
    provider = enterprise.llm_config.get("provider", "deepseek")
    model = enterprise.llm_config.get("model", "deepseek-chat")
    api_keys = enterprise.api_keys or {}

    api_key = api_keys.get(provider, "")
    if not api_key:
        raise HTTPException(status_code=400, detail=f"企业未配置 {provider} API Key")

    return LLMService(api_key=api_key, provider=provider, model=model)


def get_user_accessible_kb_ids(user: User, db: Session) -> list[str]:
    """获取用户可访问的知识库 ID"""
    # 自己的私人 KB
    my_kbs = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.owner_id == user.id)
        .all()
    )

    # 被授权的 KB
    authorized_kbs = (
        db.query(KnowledgeBase)
        .join(UserKnowledgeBase)
        .filter(UserKnowledgeBase.user_id == user.id)
        .all()
    )

    # 上级可见的公共 KB
    public_kbs = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.is_public == True)
        .all()
    )

    all_kbs = {kb.id: kb for kb in my_kbs + authorized_kbs + public_kbs}
    return list(all_kbs.keys())


def get_user_doc_ids(user: User, db: Session, kb_ids: list[str]) -> list[str]:
    """获取用户可检索的文档 ID"""
    # 用户的私人文档
    private_docs = (
        db.query(Document)
        .filter(Document.uploader_id == user.id)
        .all()
    )

    # KB 中的文档
    kb_docs = (
        db.query(Document)
        .join(KBDocument)
        .filter(KBDocument.knowledge_base_id.in_(kb_ids))
        .all()
    )

    all_docs = {doc.id: doc for doc in private_docs + kb_docs}
    return list(all_docs.keys())


# ============= 认证路由 =============
@app.post("/api/auth/register", tags=["认证"])
async def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    full_name: str = Form(None),
    enterprise_name: str = Form(None),
    db: Session = Depends(get_db),
):
    """注册新用户（首个注册的用户自动成为企业管理员；已有企业后注册的为普通成员）"""
    # 检查用户名和邮箱唯一性
    existing = db.query(User).filter(
        (User.username == username) | (User.email == email)
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="用户名或邮箱已存在")

    # 检查是否已有企业
    existing_enterprise = db.query(Enterprise).first()

    if existing_enterprise:
        # 已有企业时，新用户自动加入为普通成员
        enterprise = existing_enterprise
        enterprise_id = enterprise.id
        role = UserRole.MEMBER
    else:
        # 创建新企业，首个用户为企业管理员
        enterprise = Enterprise(
            id=str(uuid.uuid4()),
            name=enterprise_name or f"{username}的企业",
        )
        db.add(enterprise)
        db.flush()
        enterprise_id = enterprise.id
        role = UserRole.ENTERPRISE_ADMIN

    # 创建用户
    user = User(
        id=str(uuid.uuid4()),
        enterprise_id=enterprise_id,
        username=username,
        email=email,
        hashed_password=hash_password(password),
        full_name=full_name or username,
        role=role,
    )
    db.add(user)
    db.commit()

    # 生成 Token
    token_data = TokenData(
        user_id=user.id,
        username=user.username,
        role=user.role,
        enterprise_id=user.enterprise_id,
    )
    access_token = create_access_token(token_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role.value,
        },
    }


@app.post("/api/auth/login", tags=["认证"])
async def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    """用户登录"""
    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已被禁用")

    # 更新登录时间
    user.last_login = datetime.utcnow()
    db.commit()

    # 生成 Token
    token_data = TokenData(
        user_id=user.id,
        username=user.username,
        role=user.role,
        enterprise_id=user.enterprise_id,
    )
    access_token = create_access_token(token_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role.value,
        },
    }


@app.get("/api/auth/me", tags=["认证"])
async def get_me(user: User = Depends(get_current_user_dep)):
    """获取当前用户信息"""
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role.value,
        "enterprise_id": user.enterprise_id,
    }


# ============= 文档管理路由 =============
@app.post("/api/documents/upload", tags=["文档"])
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("private"),
    knowledge_base_id: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """上传文档"""
    # 检查文件类型
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}",
        )

    # 检查文件大小
    content = await file.read()
    file_size = len(content)

    if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制 ({settings.MAX_FILE_SIZE_MB}MB)",
        )

    # 保存文件
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    file_path = settings.upload_dir / f"{file_id}{file_ext}"

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # 计算哈希
    file_hash = compute_file_hash(str(file_path))

    # 创建文档记录
    doc = Document(
        id=file_id,
        enterprise_id=user.enterprise_id,
        uploader_id=user.id,
        title=file.filename.replace(file_ext, ""),
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        file_hash=file_hash,
        doc_type=doc_type,
        status="pending",
    )
    db.add(doc)
    db.commit()

    return {
        "id": doc.id,
        "title": doc.title,
        "status": doc.status,
    }


@app.post("/api/documents/{doc_id}/process", tags=["文档"])
async def process_document(
    doc_id: str,
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """处理文档（提取文本、分块、向量化）"""
    doc = db.query(Document).filter(Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    if doc.enterprise_id != user.enterprise_id:
        raise HTTPException(status_code=403, detail="无权访问")

    try:
        # 1. 提取文本
        doc.status = "processing"
        db.commit()

        file_ext = Path(doc.original_filename).suffix
        extracted = doc_processor.extract(doc.file_path, file_ext)

        # 2. 智能分块
        chunks = text_chunker.chunk_by_semantic(
            extracted["text"],
            metadata={
                "title": extracted.get("metadata", {}).get("title", doc.title),
                **extracted.get("metadata", {}),
            },
        )

        # 3. 存储到向量数据库
        vector_ids = vector_store.insert(doc.id, chunks, embedding_service)

        # 4. 保存块信息
        for i, (chunk, vector_id) in enumerate(zip(chunks, vector_ids)):
            db_chunk = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=doc.id,
                chunk_index=i,
                content=chunk["content"],
                chunk_type=chunk.get("chunk_type", "text"),
                vector_id=vector_id,
                metadata_=chunk.get("metadata", {}),
            )
            db.add(db_chunk)

        # 更新文档状态
        doc.status = "ready"
        doc.metadata_ = extracted.get("metadata", {})
        db.commit()

        return {"status": "success", "chunks": len(chunks)}

    except Exception as e:
        doc.status = "failed"
        doc.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/api/documents", tags=["文档"])
async def list_documents(
    status: str = Query(None),
    doc_type: str = Query(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取文档列表"""
    query = db.query(Document).filter(Document.enterprise_id == user.enterprise_id)

    if status:
        query = query.filter(Document.status == status)
    if doc_type:
        query = query.filter(Document.doc_type == doc_type)

    # 非管理员只能看自己的文档
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        query = query.filter(Document.uploader_id == user.id)

    docs = query.order_by(Document.created_at.desc()).all()

    return [
        {
            "id": d.id,
            "title": d.title,
            "original_filename": d.original_filename,
            "status": d.status,
            "doc_type": d.doc_type,
            "file_size": d.file_size,
            "created_at": d.created_at.isoformat(),
            "uploader": d.uploader.full_name if d.uploader else None,
        }
        for d in docs
    ]


@app.delete("/api/documents/{doc_id}", tags=["文档"])
async def delete_document(
    doc_id: str,
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """删除文档"""
    doc = db.query(Document).filter(Document.id == doc_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    if doc.uploader_id != user.id and user.role not in [
        UserRole.ENTERPRISE_ADMIN,
        UserRole.SUPER_ADMIN,
    ]:
        raise HTTPException(status_code=403, detail="无权删除")

    # 删除向量
    try:
        vector_store.delete_by_doc_id(doc.id)
    except Exception:
        pass

    # 删除文件
    if doc.file_path and Path(doc.file_path).exists():
        Path(doc.file_path).unlink()

    # 删除数据库记录（级联删除 chunks）
    db.delete(doc)
    db.commit()

    return {"status": "success"}


@app.delete("/api/documents", tags=["文档"])
async def clear_knowledge_base(
    confirm: bool = Query(False),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """清空知识库（仅管理员）"""
    if not check_permission(user, [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]):
        raise HTTPException(status_code=403, detail="需要管理员权限")

    if not confirm:
        raise HTTPException(status_code=400, detail="请确认清空操作")

    # 删除所有向量
    vector_store.delete_collection()

    # 删除所有文档记录
    docs = db.query(Document).filter(Document.enterprise_id == user.enterprise_id).all()
    for doc in docs:
        if doc.file_path and Path(doc.file_path).exists():
            Path(doc.file_path).unlink()
        db.delete(doc)

    db.commit()

    return {"status": "success", "deleted": len(docs)}


# ============= 知识库管理路由 =============
@app.post("/api/knowledge-bases", tags=["知识库"])
async def create_knowledge_base(
    name: str = Form(...),
    description: str = Form(None),
    is_public: bool = Form(False),
    parent_kb_id: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """创建知识库"""
    kb = KnowledgeBase(
        id=str(uuid.uuid4()),
        enterprise_id=user.enterprise_id,
        owner_id=user.id,
        name=name,
        description=description,
        is_public=is_public,
        parent_kb_id=parent_kb_id,
    )
    db.add(kb)

    # 创建者自动拥有管理权限
    perm = UserKnowledgeBase(
        id=str(uuid.uuid4()),
        user_id=user.id,
        knowledge_base_id=kb.id,
        permission="admin",
    )
    db.add(perm)
    db.commit()

    return {"id": kb.id, "name": kb.name}


@app.get("/api/knowledge-bases", tags=["知识库"])
async def list_knowledge_bases(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取知识库列表"""
    # 自己拥有的
    owned = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.owner_id == user.id)
        .all()
    )

    # 有权限访问的
    accessible = (
        db.query(KnowledgeBase)
        .join(UserKnowledgeBase)
        .filter(UserKnowledgeBase.user_id == user.id)
        .all()
    )

    # 公共的
    public = (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.is_public == True)
        .all()
    )

    all_kbs = {kb.id: kb for kb in owned + accessible + public}

    return [
        {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "is_public": kb.is_public,
            "owner": kb.owner.full_name if kb.owner else None,
            "doc_count": len(kb.documents),
        }
        for kb in all_kbs.values()
    ]


@app.post("/api/knowledge-bases/{kb_id}/documents", tags=["知识库"])
async def add_document_to_kb(
    kb_id: str,
    doc_id: str = Form(...),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """向知识库添加文档"""
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 检查权限
    perm = (
        db.query(UserKnowledgeBase)
        .filter(
            UserKnowledgeBase.knowledge_base_id == kb_id,
            UserKnowledgeBase.user_id == user.id,
        )
        .first()
    )

    if not perm or perm.permission not in ["admin", "write"]:
        raise HTTPException(status_code=403, detail="需要写入权限")

    # 检查文档
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 添加关联
    existing = (
        db.query(KBDocument)
        .filter(
            KBDocument.knowledge_base_id == kb_id,
            KBDocument.document_id == doc_id,
        )
        .first()
    )

    if existing:
        return {"status": "already_exists"}

    kb_doc = KBDocument(
        id=str(uuid.uuid4()),
        knowledge_base_id=kb_id,
        document_id=doc_id,
        added_by_id=user.id,
    )
    db.add(kb_doc)
    db.commit()

    return {"status": "success"}


# ============= 问答路由 =============
@app.post("/api/chat", tags=["问答"])
async def chat(
    query: str = Form(...),
    knowledge_base_ids: str = Form(None),  # 逗号分隔的 KB ID
    top_k: int = Form(5),
    temperature: float = Form(0.7),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """问答"""
    enterprise = db.query(Enterprise).filter(Enterprise.id == user.enterprise_id).first()
    if not enterprise:
        raise HTTPException(status_code=400, detail="企业不存在")

    try:
        llm = get_enterprise_llm(enterprise)
    except HTTPException as e:
        raise e

    # 确定要检索的文档
    if knowledge_base_ids:
        kb_ids = [kb.strip() for kb in knowledge_base_ids.split(",")]
    else:
        kb_ids = get_user_accessible_kb_ids(user, db)

    doc_ids = get_user_doc_ids(user, db, kb_ids)

    if not doc_ids:
        return {
            "answer": "请先上传并处理文档后再提问。",
            "sources": [],
        }

    # 查询改写
    query_variations = query_rewriter.rewrite(query)

    # 检索
    hybrid_search = HybridSearch(vector_store, embedding_service)
    results = []

    for q in query_variations:
        results.extend(
            hybrid_search.search(
                query=q,
                doc_ids=doc_ids,
                top_k=top_k,
            )
        )

    # 去重并排序
    seen = {}
    for r in results:
        if r["id"] not in seen or r["score"] > seen[r["id"]]["score"]:
            seen[r["id"]] = r

    results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # 生成回答
    rag = RAGService(llm)
    result = rag.generate_answer(query, results, temperature=temperature)

    # 记录 token 使用
    usage = result.get("usage", {})
    if usage.get("total_tokens", 0) > 0:
        token_record = TokenUsage(
            id=str(uuid.uuid4()),
            user_id=user.id,
            model=result.get("model", llm.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cost=0,  # 可根据实际计费
        )
        db.add(token_record)
        db.commit()

    return result


# ============= 企业管理路由 =============
@app.get("/api/admin/settings", tags=["管理"])
async def get_enterprise_settings(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取企业设置"""
    if user.role != UserRole.ENTERPRISE_ADMIN:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    enterprise = db.query(Enterprise).filter(Enterprise.id == user.enterprise_id).first()

    return {
        "name": enterprise.name,
        "api_keys": {k: "****" + v[-4:] if v else "" for k, v in (enterprise.api_keys or {}).items()},
        "llm_config": enterprise.llm_config,
    }


@app.put("/api/admin/settings", tags=["管理"])
async def update_enterprise_settings(
    api_keys: dict = Form(None),
    llm_config: dict = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """更新企业设置"""
    if user.role != UserRole.ENTERPRISE_ADMIN:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    enterprise = db.query(Enterprise).filter(Enterprise.id == user.enterprise_id).first()

    if api_keys:
        current_keys = enterprise.api_keys or {}
        current_keys.update(api_keys)
        enterprise.api_keys = current_keys

    if llm_config:
        enterprise.llm_config = llm_config

    db.commit()
    return {"status": "success"}


@app.get("/api/admin/users", tags=["管理"])
async def list_users(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取企业用户列表"""
    if user.role != UserRole.ENTERPRISE_ADMIN:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    users = db.query(User).filter(User.enterprise_id == user.enterprise_id).all()

    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "full_name": u.full_name,
            "role": u.role.value,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
        }
        for u in users
    ]


@app.get("/api/admin/token-usage", tags=["管理"])
async def get_token_usage(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取 Token 使用统计"""
    if user.role != UserRole.ENTERPRISE_ADMIN:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    from sqlalchemy import func

    stats = (
        db.query(
            User.username,
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
        )
        .join(TokenUsage)
        .filter(User.enterprise_id == user.enterprise_id)
        .group_by(User.id)
        .all()
    )

    return [
        {"username": s[0], "total_tokens": s[1] or 0}
        for s in stats
    ]


# ============= 前端路由 =============
@app.get("/", response_class=HTMLResponse)
async def root():
    """前端页面"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    with open("templates/login.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/register", response_class=HTMLResponse)
async def register_page():
    with open("templates/register.html", "r", encoding="utf-8") as f:
        return f.read()


# ============= 健康检查 =============
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ============= 启动 =============
if __name__ == "__main__":
    import uvicorn

    # 确保目录存在
    Path("./data").mkdir(exist_ok=True)
    Path("./uploads").mkdir(exist_ok=True)
    Path("./templates").mkdir(exist_ok=True)

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
