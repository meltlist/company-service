"""数据库模型"""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SAEnum, Float,
    ForeignKey, Integer, String, Text, JSON, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"       # 系统超级管理员
    ENTERPRISE_ADMIN = "enterprise_admin"  # 企业管理员
    MANAGER = "manager"                # 部门/知识库管理员
    MEMBER = "member"                  # 普通成员


class DocumentType(str, Enum):
    PRIVATE = "private"                # 私人文档
    DEPARTMENT = "department"          # 部门文档
    PUBLIC = "public"                  # 企业公共文档


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    enterprise_id = Column(String(36), ForeignKey("enterprises.id"), nullable=False)
    uploader_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    doc_type = Column(SAEnum(DocumentType), default=DocumentType.PRIVATE)
    title = Column(String(500), nullable=False)
    original_filename = Column(String(500))
    file_path = Column(String(1000))
    file_size = Column(Integer)
    file_hash = Column(String(64))
    status = Column(String(20), default="pending")  # pending, processing, ready, failed
    error_message = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)  # 标题结构、页数等
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    enterprise = relationship("Enterprise", back_populates="documents")
    uploader = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chunk_type = Column(String(20))  # text, table, image
    page_number = Column(Integer)
    vector_id = Column(String(100))  # Qdrant 中的向量 ID
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class Department(Base):
    __tablename__ = "departments"

    id = Column(String(36), primary_key=True)
    enterprise_id = Column(String(36), ForeignKey("enterprises.id"), nullable=False)
    parent_id = Column(String(36), ForeignKey("departments.id"), nullable=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    manager_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    enterprise = relationship("Enterprise", back_populates="departments")
    parent = relationship("Department", remote_side=[id], back_populates="children")
    children = relationship("Department", back_populates="parent")
    users = relationship("User", back_populates="department", foreign_keys="User.department_id")


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    enterprise_id = Column(String(36), ForeignKey("enterprises.id"), nullable=False)
    department_id = Column(String(36), ForeignKey("departments.id"), nullable=True)
    manager_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # 直接上级
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    full_name = Column(String(200))
    role = Column(SAEnum(UserRole), default=UserRole.MEMBER)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    enterprise = relationship("Enterprise", back_populates="users")
    department = relationship("Department", back_populates="users", foreign_keys=[department_id])
    manager = relationship("User", remote_side=[id], back_populates="subordinates")
    subordinates = relationship("User", back_populates="manager")
    documents = relationship("Document", back_populates="uploader")
    knowledge_bases = relationship("UserKnowledgeBase", back_populates="user")
    token_usage = relationship("TokenUsage", back_populates="user", foreign_keys="TokenUsage.user_id")


class Enterprise(Base):
    __tablename__ = "enterprises"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    api_keys = Column(JSON, default=dict)  # {"deepseek": "xxx", "gemini": "xxx"}
    llm_config = Column(JSON, default=dict)  # {"default_model": "xxx", "temperature": 0.7}
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="enterprise")
    departments = relationship("Department", back_populates="enterprise")
    documents = relationship("Document", back_populates="enterprise")
    knowledge_bases = relationship("KnowledgeBase", back_populates="enterprise")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(36), primary_key=True)
    enterprise_id = Column(String(36), ForeignKey("enterprises.id"), nullable=False)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    is_public = Column(Boolean, default=False)  # 是否对上级可见
    parent_kb_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    enterprise = relationship("Enterprise", back_populates="knowledge_bases")
    owner = relationship("User", foreign_keys=[owner_id])
    parent = relationship("KnowledgeBase", remote_side=[id])
    documents = relationship("KBDocument", back_populates="knowledge_base")
    users = relationship("UserKnowledgeBase", back_populates="knowledge_base")


class KBDocument(Base):
    __tablename__ = "kb_documents"

    id = Column(String(36), primary_key=True)
    knowledge_base_id = Column(String(36), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False)
    added_by_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    document = relationship("Document")
    added_by = relationship("User")


class UserKnowledgeBase(Base):
    __tablename__ = "user_knowledge_bases"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    knowledge_base_id = Column(String(36), ForeignKey("knowledge_bases.id"), nullable=False)
    permission = Column(String(20), default="read")  # read, write, admin
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="knowledge_bases")
    knowledge_base = relationship("KnowledgeBase", back_populates="users")


class TokenUsage(Base):
    __tablename__ = "token_usage"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    department_id = Column(String(36), ForeignKey("departments.id"), nullable=True)
    manager_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    model = Column(String(100), nullable=False)
    provider = Column(String(50), nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    request_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", foreign_keys=[user_id])
    department = relationship("Department", foreign_keys=[department_id])


class JoinRequestStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class JoinRequest(Base):
    __tablename__ = "join_requests"

    id = Column(String(36), primary_key=True)
    enterprise_id = Column(String(36), ForeignKey("enterprises.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    department_id = Column(String(36), ForeignKey("departments.id"), nullable=True)
    target_manager_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    message = Column(Text)
    status = Column(SAEnum(JoinRequestStatus), default=JoinRequestStatus.PENDING)
    review_note = Column(Text)
    reviewed_by_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", foreign_keys=[user_id])
    department = relationship("Department", foreign_keys=[department_id])
    target_manager = relationship("User", foreign_keys=[target_manager_id])


class LLMProvider(Base):
    """企业 LLM 提供商配置表。企业管理员可以配置多个不同的 API Key。"""
    __tablename__ = "llm_providers"

    id = Column(String(36), primary_key=True)
    enterprise_id = Column(String(36), ForeignKey("enterprises.id"), nullable=False)
    provider_type = Column(String(50), nullable=False)  # deepseek, gemini, openai, anthropic, zhipu, moonshot, qwen, doubao, custom
    display_name = Column(String(200), nullable=False)  # 展示名，如"公司研发组 DeepSeek"
    api_key = Column(String(500), nullable=False)  # API Key，建议使用加密或只保存 hash
    base_url = Column(String(500), nullable=True)  # 自定义 API 地址
    default_model = Column(String(200), nullable=True)  # 推荐模型
    is_active = Column(Boolean, default=True)  # 是否启用
    is_default = Column(Boolean, default=False)  # 是否为默认 provider
    priority = Column(Integer, default=0)  # 优先级，数值越大越优先
    config = Column(JSON, default=dict)  # 其它灵活配置（如自定义 headers、代理等）
    total_tokens_used = Column(Integer, default=0)  # 该 provider 累计消耗 tokens
    last_used_at = Column(DateTime, nullable=True)  # 最近一次使用时间
    created_by_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    enterprise = relationship("Enterprise", foreign_keys=[enterprise_id])
    created_by = relationship("User", foreign_keys=[created_by_id])

    @property
    def api_key_masked(self) -> str:
        """返回脱敏后的 API Key。"""
        if not self.api_key:
            return ""
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return self.api_key[:4] + "****" + self.api_key[-4:]


# ---- 主流 Provider 元信息 ----
LLM_PROVIDER_META = {
    "deepseek": {
        "display_name": "DeepSeek",
        "api_base": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "auth_style": "bearer",  # Authorization: Bearer <key>
        "api_type": "openai_compatible",
        "price_per_1m_input": 1.0,   # CNY/USD 参考
        "price_per_1m_output": 2.0,
        "docs_url": "https://api.deepseek.com",
    },
    "gemini": {
        "display_name": "Google Gemini",
        "api_base": "https://generativelanguage.googleapis.com",
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "auth_style": "query_key",  # ?key=<api_key>
        "api_type": "gemini",
        "price_per_1m_input": 3.0,
        "price_per_1m_output": 6.0,
        "docs_url": "https://ai.google.dev/",
    },
    "openai": {
        "display_name": "OpenAI",
        "api_base": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "auth_style": "bearer",
        "api_type": "openai_compatible",
        "price_per_1m_input": 5.0,
        "price_per_1m_output": 15.0,
        "docs_url": "https://platform.openai.com/docs",
    },
    "anthropic": {
        "display_name": "Anthropic Claude",
        "api_base": "https://api.anthropic.com/v1",
        "models": ["claude-4-5-sonnet-20250514", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
        "auth_style": "anthropic",  # x-api-key + anthropic-version
        "api_type": "openai_compatible",
        "price_per_1m_input": 30.0,
        "price_per_1m_output": 150.0,
        "docs_url": "https://docs.anthropic.com/",
    },
    "zhipu": {
        "display_name": "智谱 GLM",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-flash", "glm-4-air", "glm-4", "glm-4-plus"],
        "auth_style": "bearer",
        "api_type": "openai_compatible",
        "price_per_1m_input": 1.0,
        "price_per_1m_output": 4.0,
        "docs_url": "https://bigmodel.cn/dev/api/intro",
    },
    "moonshot": {
        "display_name": "月之暗面 Moonshot",
        "api_base": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "auth_style": "bearer",
        "api_type": "openai_compatible",
        "price_per_1m_input": 12.0,
        "price_per_1m_output": 12.0,
        "docs_url": "https://platform.moonshot.cn/",
    },
    "qwen": {
        "display_name": "通义千问（阿里云 DashScope）",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-long"],
        "auth_style": "bearer",
        "api_type": "openai_compatible",
        "price_per_1m_input": 2.0,
        "price_per_1m_output": 6.0,
        "docs_url": "https://help.aliyun.com/zh/dashscope/",
    },
    "doubao": {
        "display_name": "豆包（字节火山方舟）",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "models": ["doubao-pro-32k", "doubao-lite-32k", "doubao-pro-256k"],
        "auth_style": "bearer",
        "api_type": "openai_compatible",
        "price_per_1m_input": 2.0,
        "price_per_1m_output": 8.0,
        "docs_url": "https://www.volcengine.com/docs/82379",
    },
    "custom": {
        "display_name": "自定义（兼容 OpenAI 协议）",
        "api_base": "",
        "models": [],
        "auth_style": "bearer",
        "api_type": "openai_compatible",
        "price_per_1m_input": 0.0,
        "price_per_1m_output": 0.0,
        "docs_url": "",
    },
}


def init_db(database_url: str) -> tuple:
    """初始化数据库，返回 engine 和 SessionLocal"""
    Path("./data").mkdir(exist_ok=True)
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        echo=False,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # SQLite 动态增加列/表，确保老库用户升级兼容
    with engine.connect() as conn:
        # --- users: manager_id ---
        try:
            conn.execute(__import__("sqlalchemy").text("SELECT manager_id FROM users LIMIT 1"))
        except Exception:
            try:
                conn.execute(__import__("sqlalchemy").text("ALTER TABLE users ADD COLUMN manager_id VARCHAR(36)"))
                conn.commit()
            except Exception:
                pass

        # --- join_requests ---
        try:
            conn.execute(__import__("sqlalchemy").text("SELECT 1 FROM join_requests LIMIT 1"))
        except Exception:
            try:
                conn.execute(__import__("sqlalchemy").text("""
                    CREATE TABLE join_requests (
                        id VARCHAR(36) PRIMARY KEY,
                        enterprise_id VARCHAR(36) NOT NULL,
                        user_id VARCHAR(36) NOT NULL,
                        department_id VARCHAR(36),
                        target_manager_id VARCHAR(36),
                        message TEXT,
                        status VARCHAR(20) DEFAULT 'pending',
                        review_note TEXT,
                        reviewed_by_id VARCHAR(36),
                        reviewed_at DATETIME,
                        created_at DATETIME
                    )
                """))
                conn.commit()
            except Exception:
                pass

        # --- token_usage: provider / department_id / manager_id / request_count ---
        try:
            conn.execute(__import__("sqlalchemy").text("SELECT provider FROM token_usage LIMIT 1"))
        except Exception:
            for col_sql in [
                "ALTER TABLE token_usage ADD COLUMN provider VARCHAR(50) NOT NULL DEFAULT 'deepseek'",
                "ALTER TABLE token_usage ADD COLUMN department_id VARCHAR(36)",
                "ALTER TABLE token_usage ADD COLUMN manager_id VARCHAR(36)",
                "ALTER TABLE token_usage ADD COLUMN request_count INTEGER DEFAULT 1",
            ]:
                try:
                    conn.execute(__import__("sqlalchemy").text(col_sql))
                    conn.commit()
                except Exception:
                    pass

        # --- llm_providers ---
        try:
            conn.execute(__import__("sqlalchemy").text("SELECT 1 FROM llm_providers LIMIT 1"))
        except Exception:
            try:
                conn.execute(__import__("sqlalchemy").text("""
                    CREATE TABLE llm_providers (
                        id VARCHAR(36) PRIMARY KEY,
                        enterprise_id VARCHAR(36) NOT NULL,
                        provider_type VARCHAR(50) NOT NULL,
                        display_name VARCHAR(200) NOT NULL,
                        api_key VARCHAR(500) NOT NULL,
                        base_url VARCHAR(500),
                        default_model VARCHAR(200),
                        is_active BOOLEAN DEFAULT 1,
                        is_default BOOLEAN DEFAULT 0,
                        priority INTEGER DEFAULT 0,
                        config TEXT DEFAULT '{}',
                        total_tokens_used INTEGER DEFAULT 0,
                        last_used_at DATETIME,
                        created_by_id VARCHAR(36),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
            except Exception:
                pass

    return engine, SessionLocal
