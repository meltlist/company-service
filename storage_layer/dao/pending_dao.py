"""
挂起请求 DAO — CRUD 封装
"""
import logging
from typing import Optional
from datetime import datetime

from storage_layer.database.connection import get_connection
from storage_layer.utils.validator import validate_pending_request

logger = logging.getLogger(__name__)


def save_pending_request(request: dict) -> str:
    """
    创建挂起请求，返回 request_id。
    调用方：模块 4（理解决策层）
    """
    validate_pending_request(request)

    conn = get_connection()
    conn.execute(
        """INSERT INTO pending_requests
           (request_id, event_id, action_type, action_payload, status)
           VALUES (?, ?, ?, ?, ?)""",
        (
            request["request_id"],
            request.get("event_id"),
            request["action_type"],
            request["action_payload"],
            request.get("status", "awaiting_user"),
        ),
    )
    conn.commit()
    logger.info("挂起请求已创建: %s", request["request_id"])
    return request["request_id"]


def get_pending_request(request_id: str) -> Optional[dict]:
    """获取单个挂起请求"""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM pending_requests WHERE request_id = ?", (request_id,)
    ).fetchone()
    return dict(row) if row else None


def get_active_pending_request() -> Optional[dict]:
    """
    获取当前活跃的挂起请求（status='awaiting_user'）。
    调用方：模块 4（理解决策层）、模块 6（应用层）
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM pending_requests WHERE status = 'awaiting_user' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def resolve_pending_request(request_id: str, status: str) -> bool:
    """
    解决挂起请求（executed / cancelled / timeout）。
    调用方：模块 4（理解决策层）
    """
    valid_statuses = {"executed", "cancelled", "timeout"}
    if status not in valid_statuses:
        raise ValueError(f"非法状态: {status}，合法值: {valid_statuses}")

    resolved_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE pending_requests SET status = ?, resolved_at = ? WHERE request_id = ?",
        (status, resolved_at, request_id),
    )
    conn.commit()

    updated = cursor.rowcount > 0
    if updated:
        logger.info("挂起请求已解决: %s -> %s", request_id, status)
    return updated


def get_pending_by_event(event_id: str) -> Optional[dict]:
    """根据事件 ID 查找关联的挂起请求"""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM pending_requests WHERE event_id = ? ORDER BY created_at DESC LIMIT 1",
        (event_id,),
    ).fetchone()
    return dict(row) if row else None