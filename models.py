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
    token_usage = relationship("TokenUsage", back_populates="user")


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
    model = Column(String(100), nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="token_usage")


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

    # 确保 SQLite 中新增的列存在
    with engine.connect() as conn:
        try:
            conn.execute(
                __import__("sqlalchemy").text(
                    "ALTER TABLE users ADD COLUMN manager_id VARCHAR(36)"
                )
            )
            conn.commit()
        except Exception:
            pass

        # 创建 join_requests 表（如果不存在）
        try:
            conn.execute(
                __import__("sqlalchemy").text("SELECT 1 FROM join_requests LIMIT 1")
            )
        except Exception:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        """
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
                        """
                    )
                )
                conn.commit()
            except Exception:
                pass

    return engine, SessionLocal
