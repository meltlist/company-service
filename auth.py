"""认证模块：JWT + 密码哈希"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

import jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from models import User, UserRole


class TokenData(BaseModel):
    user_id: str
    username: str
    role: UserRole
    enterprise_id: str


def hash_password(password: str) -> str:
    """PBKDF2 密码哈希"""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000,
    )
    return f"{salt}${key.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    try:
        salt, key_hex = hashed.split("$")
        expected = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            100000,
        )
        return secrets.compare_digest(expected.hex(), key_hex)
    except (ValueError, AttributeError):
        return False


def create_access_token(data: TokenData) -> str:
    """创建 JWT Token"""
    expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRE_HOURS)
    payload = {
        **data.model_dump(),
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    """解码 JWT Token"""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return TokenData(**payload)
    except jwt.ExpiredSignatureError:
        return None
    except (jwt.InvalidTokenError, TypeError):
        return None


def get_current_user(db: Session, token: str) -> Optional[User]:
    """从 Token 获取当前用户"""
    token_data = decode_token(token)
    if not token_data:
        return None
    return db.query(User).filter(User.id == token_data.user_id, User.is_active == True).first()


def check_permission(user: User, required_roles: list[UserRole]) -> bool:
    """检查用户权限"""
    if user.role in required_roles:
        return True
    # 企业管理员可以管理本企业成员
    if user.role == UserRole.ENTERPRISE_ADMIN:
        return True
    return False
