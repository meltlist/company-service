"""
事件 DAO — CRUD 封装
"""
import logging
from typing import Optional, Tuple, List

from storage_layer.database.connection import get_connection
from storage_layer.utils.validator import validate_event, validate_event_update

logger = logging.getLogger(__name__)


def save_event(event: dict) -> str:
    """
    插入新事件，返回 event_id。
    调用方：模块 3（边缘推理层）
    """
    validate_event(event)

    conn = get_connection()
    conn.execute(
        """INSERT INTO events
           (event_id, timestamp, object_class, action, confidence,
            image_path, small_model_output, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event["event_id"],
            event["timestamp"],
            event.get("object_class"),
            event.get("action"),
            event.get("confidence"),
            event.get("image_path"),
            event.get("small_model_output"),
            event.get("status", "recorded"),
        ),
    )
    conn.commit()
    logger.info("事件已保存: %s", event["event_id"])
    return event["event_id"]


def update_event(event_id: str, updates: dict) -> bool:
    """
    更新事件字段（如 llm_input, llm_output, final_action, status 等）。
    调用方：模块 4（理解决策层）、模块 5（执行反馈层）
    """
    if not updates:
        return False

    validate_event_update(updates)

    allowed_fields = {
        "llm_input", "llm_output", "final_action", "action_type",
        "status", "object_class", "action", "confidence", "image_path",
        "small_model_output",
    }
    filtered = {k: v for k, v in updates.items() if k in allowed_fields}
    if not filtered:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in filtered)
    set_clause += ", updated_at = datetime('now', 'localtime')"
    values = list(filtered.values()) + [event_id]

    conn = get_connection()
    cursor = conn.execute(
        f"UPDATE events SET {set_clause} WHERE event_id = ?", values
    )
    conn.commit()

    updated = cursor.rowcount > 0
    if updated:
        logger.info("事件已更新: %s, 字段: %s", event_id, list(filtered.keys()))
    return updated


def get_events(
    page: int = 1, page_size: int = 20, status: str = None
) -> Tuple[List[dict], int]:
    """
    分页查询事件列表，返回 (事件列表, 总数)。
    调用方：模块 6（应用层）
    """
    conn = get_connection()

    if status:
        count_row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE status = ?", (status,)
        ).fetchone()
        total = count_row[0]
        rows = conn.execute(
            "SELECT * FROM events WHERE status = ? "
            "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (status, page_size, (page - 1) * page_size),
        ).fetchall()
    else:
        count_row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        total = count_row[0]
        rows = conn.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (page_size, (page - 1) * page_size),
        ).fetchall()

    return [dict(r) for r in rows], total


def get_event_by_id(event_id: str) -> Optional[dict]:
    """获取单个事件完整详情"""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM events WHERE event_id = ?", (event_id,)
    ).fetchone()
    return dict(row) if row else None


def get_recent_events(limit: int = 10) -> List[dict]:
    """
    获取最近 N 条事件（供大模型上下文使用）。
    调用方：模块 4（理解决策层）
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]