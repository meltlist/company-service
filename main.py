"""FastAPI 主应用"""
import os
# 设置 HuggingFace 镜像（必须在导入 sentence-transformers 之前）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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
    LLMProvider,
    JoinRequest,
    JoinRequestStatus,
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
def get_enterprise_llm(enterprise: Enterprise, db=None, provider_id: str = None) -> LLMService:
    """获取企业的 LLM 配置。优先从 llm_providers 表读取，兼容旧版 llm_config / api_keys"""

    # 1) 优先从 llm_providers 表读取
    if db is not None:
        q = db.query(LLMProvider).filter(
            LLMProvider.enterprise_id == enterprise.id,
            LLMProvider.is_active == True,
        )
        if provider_id:
            q = q.filter(LLMProvider.id == provider_id)
        else:
            q = q.order_by(LLMProvider.is_default.desc(), LLMProvider.priority.desc())
        provider = q.first()
        if provider:
            return (
                LLMService(
                    api_key=provider.api_key,
                    provider=provider.provider_type,
                    model=provider.default_model or None,
                    base_url=provider.base_url or None,
                ),
                provider,
            )

    # 2) 兼容旧版：从 enterprise.llm_config / api_keys 读取
    provider_name = enterprise.llm_config.get("provider", "deepseek") if enterprise.llm_config else "deepseek"
    model = enterprise.llm_config.get("model", "deepseek-chat") if enterprise.llm_config else "deepseek-chat"
    api_keys = enterprise.api_keys or {}
    api_key = api_keys.get(provider_name, "")
    if not api_key:
        raise HTTPException(status_code=400, detail=f"企业未配置 {provider_name} API Key")

    return LLMService(api_key=api_key, provider=provider_name, model=model), None


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
    title: str = Form(None),
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

    # 标题优先使用用户输入，否则使用文件名
    doc_title = title.strip() if title and title.strip() else file.filename.replace(file_ext, "")

    # 创建文档记录
    doc = Document(
        id=file_id,
        enterprise_id=user.enterprise_id,
        uploader_id=user.id,
        title=doc_title,
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
        llm, provider_obj = get_enterprise_llm(enterprise, db=db)
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

    # 生成回答（含异常兜底）
    try:
        rag = RAGService(llm)
        result = rag.generate_answer(query, results, temperature=temperature)
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Unauthorized" in err_msg:
            err_msg = "API Key 无效或已过期，请在「企业设置」中更新"
        elif "timed out" in err_msg or "timeout" in err_msg.lower():
            err_msg = "LLM 服务请求超时，请稍后再试"
        elif "429" in err_msg:
            err_msg = "请求过于频繁，请稍后再试"
        return {
            "answer": f"生成回答失败：{err_msg}",
            "sources": [],
        }

    # 记录 token 使用（带部门/上级/provider 维度，便于按组织聚合统计）
    usage = result.get("usage", {})
    if usage.get("total_tokens", 0) > 0:
        token_record = TokenUsage(
            id=str(uuid.uuid4()),
            user_id=user.id,
            department_id=user.department_id,
            manager_id=user.manager_id,
            model=result.get("model", llm.model),
            provider=result.get("provider", llm.provider),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        db.add(token_record)

        # 更新 provider 的累计 tokens
        if provider_obj is not None:
            provider_obj.total_tokens_used = (provider_obj.total_tokens_used or 0) + usage.get(
                "total_tokens", 0
            )
            provider_obj.last_used_at = datetime.utcnow()

        db.commit()

    return result


# ============= Provider 管理（企业管理员）=============
@app.get("/api/admin/providers", tags=["Provider"])
async def list_providers(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """列出企业已配置的所有 LLM Provider"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    providers = (
        db.query(LLMProvider)
        .filter(LLMProvider.enterprise_id == user.enterprise_id)
        .order_by(LLMProvider.is_default.desc(), LLMProvider.priority.desc())
        .all()
    )
    return [
        {
            "id": p.id,
            "provider_type": p.provider_type,
            "display_name": p.display_name,
            "api_key_masked": p.api_key_masked,
            "base_url": p.base_url,
            "default_model": p.default_model,
            "is_active": p.is_active,
            "is_default": p.is_default,
            "priority": p.priority,
            "total_tokens_used": p.total_tokens_used or 0,
            "last_used_at": p.last_used_at.isoformat() if p.last_used_at else None,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in providers
    ]


@app.post("/api/admin/providers", tags=["Provider"])
async def create_provider(
    provider_type: str = Form(...),
    api_key: str = Form(...),
    display_name: str = Form(None),
    base_url: str = Form(None),
    default_model: str = Form(None),
    is_active: bool = Form(True),
    is_default: bool = Form(False),
    priority: int = Form(0),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """创建一个 LLM Provider 配置"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    # 如果新配置为默认，取消其他所有默认
    if is_default:
        db.query(LLMProvider).filter(LLMProvider.enterprise_id == user.enterprise_id).update(
            {LLMProvider.is_default: False}
        )

    p = LLMProvider(
        id=str(uuid.uuid4()),
        enterprise_id=user.enterprise_id,
        provider_type=provider_type,
        display_name=display_name or provider_type.upper(),
        api_key=api_key,
        base_url=base_url,
        default_model=default_model,
        is_active=is_active,
        is_default=is_default,
        priority=priority,
        created_by_id=user.id,
    )
    db.add(p)
    db.commit()
    return {"id": p.id, "display_name": p.display_name, "is_default": p.is_default}


@app.put("/api/admin/providers/{provider_id}", tags=["Provider"])
async def update_provider(
    provider_id: str,
    display_name: str = Form(None),
    api_key: str = Form(None),
    base_url: str = Form(None),
    default_model: str = Form(None),
    is_active: bool = Form(None),
    is_default: bool = Form(None),
    priority: int = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """更新 LLM Provider 配置"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    p = (
        db.query(LLMProvider)
        .filter(LLMProvider.id == provider_id, LLMProvider.enterprise_id == user.enterprise_id)
        .first()
    )
    if not p:
        raise HTTPException(status_code=404, detail="Provider 不存在")

    if display_name is not None:
        p.display_name = display_name
    if api_key is not None:
        p.api_key = api_key
    if base_url is not None:
        p.base_url = base_url
    if default_model is not None:
        p.default_model = default_model
    if is_active is not None:
        p.is_active = is_active
    if priority is not None:
        p.priority = priority
    if is_default and is_default:
        db.query(LLMProvider).filter(LLMProvider.enterprise_id == user.enterprise_id).update(
            {LLMProvider.is_default: False}
        )
        p.is_default = True

    db.commit()
    return {"status": "success", "id": p.id, "is_default": p.is_default}


@app.delete("/api/admin/providers/{provider_id}", tags=["Provider"])
async def delete_provider(
    provider_id: str,
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """删除 LLM Provider 配置"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    p = (
        db.query(LLMProvider)
        .filter(LLMProvider.id == provider_id, LLMProvider.enterprise_id == user.enterprise_id)
        .first()
    )
    if not p:
        raise HTTPException(status_code=404, detail="Provider 不存在")

    db.delete(p)
    db.commit()
    return {"status": "success"}


@app.get("/api/providers/types", tags=["Provider"])
async def get_supported_provider_types():
    """返回所有受支持的 provider 类型，供前端下拉选使用"""
    return [
        {
            "type": k,
            "display_name": v["display_name"],
            "api_base": v["api_base"],
            "default_model": v["models"][0] if v["models"] else "",
            "models": v["models"],
        }
        for k, v in {
            "deepseek": {
                "display_name": "DeepSeek",
                "api_base": "https://api.deepseek.com",
                "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
            },
            "gemini": {
                "display_name": "Google Gemini",
                "api_base": "https://generativelanguage.googleapis.com",
                "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            },
            "openai": {
                "display_name": "OpenAI",
                "api_base": "https://api.openai.com/v1",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            },
            "anthropic": {
                "display_name": "Anthropic Claude",
                "api_base": "https://api.anthropic.com/v1",
                "models": ["claude-4-5-sonnet-20250514", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
            },
            "zhipu": {
                "display_name": "智谱 GLM",
                "api_base": "https://open.bigmodel.cn/api/paas/v4",
                "models": ["glm-4-flash", "glm-4-air", "glm-4", "glm-4-plus"],
            },
            "moonshot": {
                "display_name": "月之暗面 Moonshot",
                "api_base": "https://api.moonshot.cn/v1",
                "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
            },
            "qwen": {
                "display_name": "通义千问（阿里云 DashScope）",
                "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-long"],
            },
            "doubao": {
                "display_name": "豆包（字节火山方舟）",
                "api_base": "https://ark.cn-beijing.volces.com/api/v3",
                "models": ["doubao-pro-32k", "doubao-lite-32k", "doubao-pro-256k"],
            },
            "custom": {
                "display_name": "自定义（OpenAI 兼容协议）",
                "api_base": "",
                "models": [],
            },
        }.items()
    ]


# ============= Token 用量聚合查询 ============
@app.get("/api/admin/token-usage/summary", tags=["Token 用量"])
async def get_token_usage_summary(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """企业管理员查看全企业各部门 / 各成员的 token 用量"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    from sqlalchemy import func

    enterprise_user_ids = [
        u[0] for u in db.query(User.id).filter(User.enterprise_id == user.enterprise_id).all()
    ]

    if not enterprise_user_ids:
        return {"total_tokens": 0, "by_department": [], "by_user": [], "by_provider": []}

    # 总 tokens
    total = (
        db.query(func.sum(TokenUsage.total_tokens))
        .filter(TokenUsage.user_id.in_(enterprise_user_ids))
        .scalar() or 0
    )

    # 按部门聚合
    by_dept = (
        db.query(
            Department.id,
            Department.name,
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .outerjoin(User, User.department_id == Department.id)
        .join(TokenUsage, TokenUsage.user_id == User.id)
        .filter(User.enterprise_id == user.enterprise_id)
        .group_by(Department.id, Department.name)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
        .all()
    )

    # 未分配部门的成员
    unassigned = (
        db.query(
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .filter(
            TokenUsage.user_id.in_(enterprise_user_ids),
            TokenUsage.department_id.is_(None),
        )
        .first()
    )

    # 按 user
    by_user = (
        db.query(
            User.id,
            User.username,
            User.full_name,
            Department.name.label("dept_name"),
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .outerjoin(Department, Department.id == User.department_id)
        .join(TokenUsage, TokenUsage.user_id == User.id)
        .filter(User.enterprise_id == user.enterprise_id)
        .group_by(User.id, User.username, User.full_name, Department.name)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
        .limit(50)
        .all()
    )

    # 按 provider
    by_provider = (
        db.query(
            TokenUsage.provider,
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .filter(TokenUsage.user_id.in_(enterprise_user_ids))
        .group_by(TokenUsage.provider)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
        .all()
    )

    return {
        "total_tokens": int(total),
        "by_department": [
            {
                "department_id": d[0],
                "department_name": d[1] or "未分配部门",
                "total_tokens": int(d[2] or 0),
                "requests": int(d[3] or 0),
            }
            for d in by_dept
        ]
        + (
            [
                {
                    "department_id": None,
                    "department_name": "未分配部门",
                    "total_tokens": int(unassigned.total_tokens or 0),
                    "requests": int(unassigned.requests or 0),
                }
            ]
            if unassigned and (unassigned.total_tokens or unassigned.requests)
            else []
        ),
        "by_user": [
            {
                "user_id": u[0],
                "username": u[1],
                "full_name": u[2],
                "department_name": u[3] or "未分配部门",
                "total_tokens": int(u[4] or 0),
                "requests": int(u[5] or 0),
            }
            for u in by_user
        ],
        "by_provider": [
            {
                "provider": p[0] or "unknown",
                "total_tokens": int(p[1] or 0),
                "requests": int(p[2] or 0),
            }
            for p in by_provider
        ],
    }


@app.get("/api/users/token-usage/team", tags=["Token 用量"])
async def get_my_team_token_usage(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """直属上级/部门负责人查看下属 token 用量。

    - 企业管理员：查看所有部门下属
    - 部门负责人：查看本部门成员
    - 普通用户：查看自己的直属下级
    """
    from sqlalchemy import func

    subordinate_user_ids = []

    # 1) 企业管理员：所有成员
    if user.role in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        subordinate_user_ids = [
            u[0]
            for u in db.query(User.id).filter(User.enterprise_id == user.enterprise_id).all()
        ]
        filter_by = "all"
    # 2) 部门负责人：本部门成员
    else:
        dept_manager = (
            db.query(Department)
            .filter(
                Department.manager_id == user.id,
                Department.enterprise_id == user.enterprise_id,
            )
            .first()
        )
        if dept_manager:
            subordinate_user_ids = [
                u[0]
                for u in db.query(User.id).filter(User.department_id == dept_manager.id).all()
            ]
            filter_by = "department"
        else:
            # 3) 普通用户：直属下级
            subordinate_user_ids = [
                u[0] for u in db.query(User.id).filter(User.manager_id == user.id).all()
            ]
            filter_by = "direct_subordinates"

    # 计算下属直接人数
    direct_subordinate_count = db.query(User).filter(User.manager_id == user.id).count()

    if not subordinate_user_ids:
        return {
            "filter_by": filter_by,
            "total_users": 0,
            "total_tokens": 0,
            "direct_subordinate_count": direct_subordinate_count,
            "by_user": [],
        }

    total = (
        db.query(func.sum(TokenUsage.total_tokens))
        .filter(TokenUsage.user_id.in_(subordinate_user_ids))
        .scalar() or 0
    )

    by_user = (
        db.query(
            User.id,
            User.username,
            User.full_name,
            Department.name.label("dept_name"),
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .outerjoin(Department, Department.id == User.department_id)
        .join(TokenUsage, TokenUsage.user_id == User.id)
        .filter(User.id.in_(subordinate_user_ids))
        .group_by(User.id, User.username, User.full_name, Department.name)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
        .all()
    )

    # 处理没有任何 token 使用记录但属于下属的用户（用于展示全员）
    existing_user_ids = {u[0] for u in by_user}
    no_usage = (
        db.query(User.id, User.username, User.full_name, Department.name)
        .outerjoin(Department, Department.id == User.department_id)
        .filter(User.id.in_(subordinate_user_ids), User.id.notin_(existing_user_ids))
        .all()
    )

    return {
        "filter_by": filter_by,
        "total_users": len(subordinate_user_ids),
        "direct_subordinate_count": direct_subordinate_count,
        "total_tokens": int(total or 0),
        "by_user": [
            {
                "user_id": u[0],
                "username": u[1],
                "full_name": u[2],
                "department_name": u[3] or "未分配部门",
                "total_tokens": int(u[4] or 0),
                "requests": int(u[5] or 0),
            }
            for u in by_user
        ]
        + [
            {
                "user_id": u[0],
                "username": u[1],
                "full_name": u[2],
                "department_name": u[3] or "未分配部门",
                "total_tokens": 0,
                "requests": 0,
            }
            for u in no_usage
        ],
    }


@app.get("/api/admin/token-usage/summary", tags=["Token 用量"])
async def get_token_usage_summary(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """企业管理员查看全企业各部门 / 各成员的 token 用量"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    from sqlalchemy import func

    enterprise_user_ids = [
        u[0] for u in db.query(User.id).filter(User.enterprise_id == user.enterprise_id).all()
    ]

    if not enterprise_user_ids:
        return {"total_tokens": 0, "by_department": [], "by_user": [], "by_provider": []}

    # 总 tokens
    total = (
        db.query(func.sum(TokenUsage.total_tokens))
        .filter(TokenUsage.user_id.in_(enterprise_user_ids))
        .scalar() or 0
    )

    # 按部门聚合
    by_dept = (
        db.query(
            Department.id,
            Department.name,
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .outerjoin(User, User.department_id == Department.id)
        .join(TokenUsage, TokenUsage.user_id == User.id)
        .filter(User.enterprise_id == user.enterprise_id)
        .group_by(Department.id, Department.name)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
        .all()
    )

    # 未分配部门的成员
    unassigned = (
        db.query(
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .filter(
            TokenUsage.user_id.in_(enterprise_user_ids),
            TokenUsage.department_id.is_(None),
        )
        .first()
    )

    # 按 user
    by_user = (
        db.query(
            User.id,
            User.username,
            User.full_name,
            Department.name.label("dept_name"),
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .outerjoin(Department, Department.id == User.department_id)
        .join(TokenUsage, TokenUsage.user_id == User.id)
        .filter(User.enterprise_id == user.enterprise_id)
        .group_by(User.id, User.username, User.full_name, Department.name)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
        .limit(50)
        .all()
    )

    # 按 provider
    by_provider = (
        db.query(
            TokenUsage.provider,
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .filter(TokenUsage.user_id.in_(enterprise_user_ids))
        .group_by(TokenUsage.provider)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
        .all()
    )

    return {
        "total_tokens": int(total),
        "by_department": [
            {
                "department_id": d[0],
                "department_name": d[1] or "未分配部门",
                "total_tokens": int(d[2] or 0),
                "requests": int(d[3] or 0),
            }
            for d in by_dept
        ]
        + (
            [
                {
                    "department_id": None,
                    "department_name": "未分配部门",
                    "total_tokens": int(unassigned.total_tokens or 0),
                    "requests": int(unassigned.requests or 0),
                }
            ]
            if unassigned and (unassigned.total_tokens or unassigned.requests)
            else []
        ),
        "by_user": [
            {
                "user_id": u[0],
                "username": u[1],
                "full_name": u[2],
                "department_name": u[3] or "未分配部门",
                "total_tokens": int(u[4] or 0),
                "requests": int(u[5] or 0),
            }
            for u in by_user
        ],
        "by_provider": [
            {
                "provider": p[0] or "unknown",
                "total_tokens": int(p[1] or 0),
                "requests": int(p[2] or 0),
            }
            for p in by_provider
        ],
    }


@app.get("/api/users/token-usage/team", tags=["Token 用量"])
async def get_my_team_token_usage(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """直属上级/部门负责人查看下属 token 用量。

    - 企业管理员：查看所有部门下属
    - 部门负责人：查看本部门成员
    - 普通用户：查看自己的直属下级
    """
    from sqlalchemy import func

    subordinate_user_ids = []

    # 1) 企业管理员：所有成员
    if user.role in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        subordinate_user_ids = [
            u[0]
            for u in db.query(User.id).filter(User.enterprise_id == user.enterprise_id).all()
        ]
        filter_by = "all"
    # 2) 部门负责人：本部门成员
    else:
        dept_manager = (
            db.query(Department)
            .filter(
                Department.manager_id == user.id,
                Department.enterprise_id == user.enterprise_id,
            )
            .first()
        )
        if dept_manager:
            subordinate_user_ids = [
                u[0]
                for u in db.query(User.id).filter(User.department_id == dept_manager.id).all()
            ]
            filter_by = "department"
        else:
            # 3) 普通用户：直属下级
            subordinate_user_ids = [
                u[0] for u in db.query(User.id).filter(User.manager_id == user.id).all()
            ]
            filter_by = "direct_subordinates"

    # 计算下属直接人数
    direct_subordinate_count = db.query(User).filter(User.manager_id == user.id).count()

    if not subordinate_user_ids:
        return {
            "filter_by": filter_by,
            "total_users": 0,
            "total_tokens": 0,
            "direct_subordinate_count": direct_subordinate_count,
            "by_user": [],
        }

    total = (
        db.query(func.sum(TokenUsage.total_tokens))
        .filter(TokenUsage.user_id.in_(subordinate_user_ids))
        .scalar() or 0
    )

    by_user = (
        db.query(
            User.id,
            User.username,
            User.full_name,
            Department.name.label("dept_name"),
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.count(TokenUsage.id).label("requests"),
        )
        .outerjoin(Department, Department.id == User.department_id)
        .join(TokenUsage, TokenUsage.user_id == User.id)
        .filter(User.id.in_(subordinate_user_ids))
        .group_by(User.id, User.username, User.full_name, Department.name)
        .order_by(func.sum(TokenUsage.total_tokens).desc())
        .all()
    )

    # 处理没有任何 token 使用记录但属于下属的用户（用于展示全员）
    existing_user_ids = {u[0] for u in by_user}
    no_usage = (
        db.query(User.id, User.username, User.full_name, Department.name)
        .outerjoin(Department, Department.id == User.department_id)
        .filter(User.id.in_(subordinate_user_ids), User.id.notin_(existing_user_ids))
        .all()
    )

    return {
        "filter_by": filter_by,
        "total_users": len(subordinate_user_ids),
        "direct_subordinate_count": direct_subordinate_count,
        "total_tokens": int(total or 0),
        "by_user": [
            {
                "user_id": u[0],
                "username": u[1],
                "full_name": u[2],
                "department_name": u[3] or "未分配部门",
                "total_tokens": int(u[4] or 0),
                "requests": int(u[5] or 0),
            }
            for u in by_user
        ]
        + [
            {
                "user_id": u[0],
                "username": u[1],
                "full_name": u[2],
                "department_name": u[3] or "未分配部门",
                "total_tokens": 0,
                "requests": 0,
            }
            for u in no_usage
        ],
    }


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
        "api_keys_masked": {k: "****" + v[-4:] if v else "" for k, v in (enterprise.api_keys or {}).items()},
        "llm_config": enterprise.llm_config,
    }


@app.put("/api/admin/settings", tags=["管理"])
async def update_enterprise_settings(
    api_keys: str = Form(None),
    llm_config: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """更新企业设置"""
    if user.role != UserRole.ENTERPRISE_ADMIN:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    enterprise = db.query(Enterprise).filter(Enterprise.id == user.enterprise_id).first()

    import json as _json
    if api_keys:
        try:
            keys_dict = _json.loads(api_keys)
            # 创建新字典替换，确保 SQLAlchemy 追踪到变更
            current_keys = dict(enterprise.api_keys or {})
            current_keys.update(keys_dict)
            enterprise.api_keys = current_keys
            db.flush()  # 立即刷新确认变更
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="api_keys 格式无效")

    if llm_config:
        try:
            config_dict = _json.loads(llm_config)
            enterprise.llm_config = config_dict
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="llm_config 格式无效")

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


# ============= 组织架构 & 人员管理 =============

# ---------- 部门管理 ----------
@app.get("/api/departments", tags=["组织"])
async def list_departments(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取部门列表（扁平）"""
    departments = (
        db.query(Department).filter(Department.enterprise_id == user.enterprise_id).all()
    )
    return [
        {
            "id": d.id,
            "parent_id": d.parent_id,
            "name": d.name,
            "description": d.description,
            "manager_id": d.manager_id,
            "manager_name": (
                db.query(User).filter(User.id == d.manager_id).first().full_name
                if d.manager_id
                else None
            ),
            "member_count": len(d.users) if d.users else 0,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in departments
    ]


@app.get("/api/departments/tree", tags=["组织"])
async def get_department_tree(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取组织架构树（包含成员）"""
    departments = (
        db.query(Department).filter(Department.enterprise_id == user.enterprise_id).all()
    )
    dept_map = {d.id: d for d in departments}

    def build_node(dept: Department):
        children = [c for c in departments if c.parent_id == dept.id]
        members = (
            db.query(User)
            .filter(
                User.department_id == dept.id,
                User.enterprise_id == user.enterprise_id,
            )
            .all()
        )
        return {
            "id": dept.id,
            "parent_id": dept.parent_id,
            "name": dept.name,
            "description": dept.description,
            "manager_id": dept.manager_id,
            "manager_name": (
                dept_map[dept.manager_id].full_name
                if dept.manager_id and dept.manager_id in dept_map
                else None
            ),
            "children": [build_node(c) for c in children],
            "members": [
                {
                    "id": m.id,
                    "username": m.username,
                    "full_name": m.full_name,
                    "email": m.email,
                    "role": m.role.value,
                    "manager_id": m.manager_id,
                    "is_active": m.is_active,
                }
                for m in members
            ],
        }

    roots = [d for d in departments if d.parent_id is None or d.parent_id not in dept_map]
    if not roots:
        return []
    return [build_node(r) for r in roots]


@app.post("/api/departments", tags=["组织"])
async def create_department(
    name: str = Form(...),
    description: str = Form(None),
    parent_id: str = Form(None),
    manager_id: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """创建部门（仅企业管理员）"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    dept = Department(
        id=str(uuid.uuid4()),
        enterprise_id=user.enterprise_id,
        parent_id=parent_id if parent_id and parent_id.strip() else None,
        name=name,
        description=description,
        manager_id=manager_id if manager_id and manager_id.strip() else None,
    )
    db.add(dept)
    db.commit()
    return {"id": dept.id, "name": dept.name}


@app.put("/api/departments/{dept_id}", tags=["组织"])
async def update_department(
    dept_id: str,
    name: str = Form(None),
    description: str = Form(None),
    parent_id: str = Form(None),
    manager_id: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """更新部门"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    dept = (
        db.query(Department)
        .filter(
            Department.id == dept_id,
            Department.enterprise_id == user.enterprise_id,
        )
        .first()
    )
    if not dept:
        raise HTTPException(status_code=404, detail="部门不存在")

    if name:
        dept.name = name
    if description is not None:
        dept.description = description
    if parent_id is not None:
        dept.parent_id = parent_id.strip() if parent_id.strip() else None
    if manager_id is not None:
        dept.manager_id = manager_id.strip() if manager_id.strip() else None

    db.commit()
    return {"status": "success"}


@app.delete("/api/departments/{dept_id}", tags=["组织"])
async def delete_department(
    dept_id: str,
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """删除部门（仅企业管理员）"""
    if user.role not in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(status_code=403, detail="需要企业管理员权限")

    dept = (
        db.query(Department)
        .filter(
            Department.id == dept_id,
            Department.enterprise_id == user.enterprise_id,
        )
        .first()
    )
    if not dept:
        raise HTTPException(status_code=404, detail="部门不存在")

    # 将子部门 parent_id 设为 null
    children = db.query(Department).filter(Department.parent_id == dept_id).all()
    for c in children:
        c.parent_id = None

    # 解除成员的部门归属
    members = db.query(User).filter(User.department_id == dept_id).all()
    for m in members:
        m.department_id = None

    db.delete(dept)
    db.commit()
    return {"status": "success"}


# ---------- 人员管理（下属查询） ----------
@app.get("/api/users/team", tags=["人员"])
async def get_my_team(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取我的下属人员树（递归）"""
    all_users = (
        db.query(User).filter(User.enterprise_id == user.enterprise_id).all()
    )
    user_map = {u.id: u for u in all_users}

    def build_subtree(user_id: str):
        subordinates = [u for u in all_users if u.manager_id == user_id]
        nodes = []
        for sub in subordinates:
            child = {
                "id": sub.id,
                "username": sub.username,
                "full_name": sub.full_name,
                "email": sub.email,
                "role": sub.role.value,
                "department_id": sub.department_id,
                "department_name": (
                    user_map[sub.department_id].department.name
                    if sub.department_id and sub.department_id in user_map and sub.department
                    else None
                ),
                "manager_id": sub.manager_id,
                "is_active": sub.is_active,
                "children": [],
            }
            # 递归查找下属的下属
            grand_children = build_subtree(sub.id)
            if grand_children:
                child["children"] = grand_children
            nodes.append(child)
        return nodes

    return build_subtree(user.id)


@app.get("/api/users/details", tags=["人员"])
async def get_user_details(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """获取当前用户详细信息（含部门、上级）"""
    manager = (
        db.query(User).filter(User.id == user.manager_id).first()
        if user.manager_id
        else None
    )
    department = (
        db.query(Department).filter(Department.id == user.department_id).first()
        if user.department_id
        else None
    )
    subordinate_count = (
        db.query(User).filter(User.manager_id == user.id).count()
    )
    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role.value,
        "department_id": user.department_id,
        "department_name": department.name if department else None,
        "manager_id": user.manager_id,
        "manager_name": manager.full_name if manager else None,
        "subordinate_count": subordinate_count,
        "is_admin": user.role in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN],
    }


@app.put("/api/users/{target_id}/department", tags=["人员"])
async def set_user_department(
    target_id: str,
    department_id: str = Form(None),
    manager_id: str = Form(None),
    role: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """分配用户部门/上级/角色（企业管理员或部门负责人）"""
    target = (
        db.query(User)
        .filter(User.id == target_id, User.enterprise_id == user.enterprise_id)
        .first()
    )
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")

    is_admin = user.role in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]
    # 部门负责人可以调整本部门成员的上级，但不能改角色
    is_dept_manager = False
    target_dept = (
        db.query(Department).filter(Department.id == target.department_id).first()
        if target.department_id
        else None
    )
    if target_dept and target_dept.manager_id == user.id:
        is_dept_manager = True

    if not is_admin and not is_dept_manager:
        raise HTTPException(status_code=403, detail="无权操作")

    if department_id is not None:
        target.department_id = (
            department_id.strip() if department_id.strip() else None
        )
    if manager_id is not None:
        target.manager_id = manager_id.strip() if manager_id.strip() else None
    if role and is_admin:
        try:
            target.role = UserRole(role)
        except ValueError:
            raise HTTPException(status_code=400, detail="无效角色")

    db.commit()
    return {"status": "success"}


# ---------- 申请加入部门 ----------
@app.post("/api/join-requests", tags=["组织"])
async def create_join_request(
    department_id: str = Form(None),
    target_manager_id: str = Form(None),
    message: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """提交加入部门/团队申请"""
    if not department_id and not target_manager_id:
        raise HTTPException(status_code=400, detail="请选择部门或上级")

    # 检查是否已有待处理申请
    existing = (
        db.query(JoinRequest)
        .filter(
            JoinRequest.user_id == user.id,
            JoinRequest.status == JoinRequestStatus.PENDING,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="已有待处理申请")

    dept = None
    if department_id:
        dept = (
            db.query(Department)
            .filter(
                Department.id == department_id,
                Department.enterprise_id == user.enterprise_id,
            )
            .first()
        )
        if not dept:
            raise HTTPException(status_code=404, detail="部门不存在")

    if target_manager_id:
        target = (
            db.query(User)
            .filter(User.id == target_manager_id, User.enterprise_id == user.enterprise_id)
            .first()
        )
        if not target:
            raise HTTPException(status_code=404, detail="目标上级不存在")

    request = JoinRequest(
        id=str(uuid.uuid4()),
        enterprise_id=user.enterprise_id,
        user_id=user.id,
        department_id=department_id if department_id else None,
        target_manager_id=target_manager_id if target_manager_id else None,
        message=message,
    )
    db.add(request)
    db.commit()
    return {"status": "success", "id": request.id}


@app.get("/api/join-requests/mine", tags=["组织"])
async def get_my_requests(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """查看我提交的申请"""
    requests = (
        db.query(JoinRequest).filter(JoinRequest.user_id == user.id).order_by(JoinRequest.created_at.desc()).all()
    )
    return [
        {
            "id": r.id,
            "department_id": r.department_id,
            "department_name": (
                db.query(Department).filter(Department.id == r.department_id).first().name
                if r.department_id
                else None
            ),
            "target_manager_id": r.target_manager_id,
            "target_manager_name": (
                db.query(User).filter(User.id == r.target_manager_id).first().full_name
                if r.target_manager_id
                else None
            ),
            "message": r.message,
            "status": r.status.value,
            "review_note": r.review_note,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
        }
        for r in requests
    ]


@app.get("/api/join-requests/pending", tags=["组织"])
async def get_pending_requests(
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """查看我需要审批的申请（部门负责人或企业管理员）"""
    is_admin = user.role in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]

    # 获取我管理的部门
    my_depts = (
        db.query(Department)
        .filter(Department.manager_id == user.id, Department.enterprise_id == user.enterprise_id)
        .all()
    )
    my_dept_ids = [d.id for d in my_depts]

    if is_admin:
        requests = (
            db.query(JoinRequest)
            .filter(
                JoinRequest.enterprise_id == user.enterprise_id,
                JoinRequest.status == JoinRequestStatus.PENDING,
            )
            .order_by(JoinRequest.created_at.desc())
            .all()
        )
    elif my_dept_ids:
        requests = (
            db.query(JoinRequest)
            .filter(
                JoinRequest.enterprise_id == user.enterprise_id,
                JoinRequest.status == JoinRequestStatus.PENDING,
                JoinRequest.department_id.in_(my_dept_ids) | (JoinRequest.target_manager_id == user.id),
            )
            .order_by(JoinRequest.created_at.desc())
            .all()
        )
    else:
        # 作为被指定的目标上级查看
        requests = (
            db.query(JoinRequest)
            .filter(
                JoinRequest.enterprise_id == user.enterprise_id,
                JoinRequest.status == JoinRequestStatus.PENDING,
                JoinRequest.target_manager_id == user.id,
            )
            .order_by(JoinRequest.created_at.desc())
            .all()
        )

    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "user_name": (
                db.query(User).filter(User.id == r.user_id).first().full_name
                if r.user_id
                else None
            ),
            "user_email": (
                db.query(User).filter(User.id == r.user_id).first().email
                if r.user_id
                else None
            ),
            "department_id": r.department_id,
            "department_name": (
                db.query(Department).filter(Department.id == r.department_id).first().name
                if r.department_id
                else None
            ),
            "target_manager_id": r.target_manager_id,
            "target_manager_name": (
                db.query(User).filter(User.id == r.target_manager_id).first().full_name
                if r.target_manager_id
                else None
            ),
            "message": r.message,
            "status": r.status.value,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in requests
    ]


@app.post("/api/join-requests/{request_id}/approve", tags=["组织"])
async def approve_join_request(
    request_id: str,
    review_note: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """通过加入申请"""
    request = (
        db.query(JoinRequest)
        .filter(JoinRequest.id == request_id, JoinRequest.enterprise_id == user.enterprise_id)
        .first()
    )
    if not request:
        raise HTTPException(status_code=404, detail="申请不存在")
    if request.status != JoinRequestStatus.PENDING:
        raise HTTPException(status_code=400, detail="申请已处理")

    # 权限检查
    is_admin = user.role in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]
    target_dept = (
        db.query(Department).filter(Department.id == request.department_id).first()
        if request.department_id
        else None
    )
    is_dept_manager = target_dept and target_dept.manager_id == user.id
    is_target_manager = request.target_manager_id == user.id

    if not is_admin and not is_dept_manager and not is_target_manager:
        raise HTTPException(status_code=403, detail="无权审批")

    # 审批通过，分配部门和上级
    target_user = db.query(User).filter(User.id == request.user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if request.department_id:
        target_user.department_id = request.department_id
    if request.target_manager_id:
        target_user.manager_id = request.target_manager_id
    elif target_dept and target_dept.manager_id:
        target_user.manager_id = target_dept.manager_id

    request.status = JoinRequestStatus.APPROVED
    request.review_note = review_note
    request.reviewed_by_id = user.id
    request.reviewed_at = datetime.utcnow()
    db.commit()
    return {"status": "success"}


@app.post("/api/join-requests/{request_id}/reject", tags=["组织"])
async def reject_join_request(
    request_id: str,
    review_note: str = Form(None),
    user: User = Depends(get_current_user_dep),
    db: Session = Depends(get_db),
):
    """拒绝加入申请"""
    request = (
        db.query(JoinRequest)
        .filter(JoinRequest.id == request_id, JoinRequest.enterprise_id == user.enterprise_id)
        .first()
    )
    if not request:
        raise HTTPException(status_code=404, detail="申请不存在")
    if request.status != JoinRequestStatus.PENDING:
        raise HTTPException(status_code=400, detail="申请已处理")

    is_admin = user.role in [UserRole.ENTERPRISE_ADMIN, UserRole.SUPER_ADMIN]
    target_dept = (
        db.query(Department).filter(Department.id == request.department_id).first()
        if request.department_id
        else None
    )
    is_dept_manager = target_dept and target_dept.manager_id == user.id
    is_target_manager = request.target_manager_id == user.id

    if not is_admin and not is_dept_manager and not is_target_manager:
        raise HTTPException(status_code=403, detail="无权审批")

    request.status = JoinRequestStatus.REJECTED
    request.review_note = review_note
    request.reviewed_by_id = user.id
    request.reviewed_at = datetime.utcnow()
    db.commit()
    return {"status": "success"}


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


@app.get("/enterprise", response_class=HTMLResponse)
async def enterprise_page():
    with open("templates/enterprise.html", "r", encoding="utf-8") as f:
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
