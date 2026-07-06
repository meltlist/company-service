"""
房间模型 DAO — CRUD 封装
"""
import logging
from typing import Optional
from datetime import datetime

from storage_layer.database.connection import get_connection

logger = logging.getLogger(__name__)


def save_room_model(
    grid_data: str, text_description: str, large_items: str
) -> bool:
    """
    保存/覆盖房间模型（INSERT OR REPLACE，id 固定为 1）。
    调用方：模块 2（感知层）
    """
    conn = get_connection()
    updated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conn.execute(
        """INSERT OR REPLACE INTO room_model (id, updated_at, grid_data, text_description, large_items)
           VALUES (1, ?, ?, ?, ?)""",
        (updated_at, grid_data, text_description, large_items),
    )
    conn.commit()
    logger.info("房间模型已保存/更新")
    return True


def get_room_model() -> Optional[dict]:
    """
    获取当前房间模型，无数据时返回 None。
    调用方：模块 2（感知层）、模块 4（理解决策层）、模块 6（应用层）
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM room_model WHERE id = 1"
    ).fetchone()
    return dict(row) if row else None