"""
上下文压缩逻辑
"""
import logging
from datetime import datetime, timedelta
from typing import List

from storage_layer.database.connection import get_connection

logger = logging.getLogger(__name__)


def compress_event(event: dict) -> dict:
    """
    将事件压缩为摘要，清理大字段。
    返回压缩后的字段字典。
    """
    summary = (
        f"[{event['timestamp']}] "
        f"目标: {event.get('object_class', '未知')}, "
        f"动作: {event.get('action', '未知')}, "
        f"结果: {event.get('action_type', '未知')} -> {event.get('final_action', '未知')}"
    )
    return {
        "small_model_output": summary,
        "llm_input": None,
        "llm_output": None,
        "status": "compressed",
    }


def run_compression(retention_days: int = 7, batch_size: int = 100) -> int:
    """
    扫描 events 表中超过 retention_days 天且 status != 'compressed' 的记录，
    批量压缩。返回压缩的记录数。
    """
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    conn = get_connection()

    # 查询需要压缩的记录
    rows = conn.execute(
        "SELECT * FROM events WHERE timestamp < ? AND status != 'compressed' "
        "ORDER BY timestamp ASC",
        (cutoff,),
    ).fetchall()

    if not rows:
        logger.info("没有需要压缩的事件")
        return 0

    total_compressed = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        try:
            for row in batch:
                event = dict(row)
                compressed = compress_event(event)
                conn.execute(
                    """UPDATE events
                       SET small_model_output = ?, llm_input = ?, llm_output = ?,
                           status = ?, updated_at = datetime('now', 'localtime')
                       WHERE event_id = ?""",
                    (
                        compressed["small_model_output"],
                        compressed["llm_input"],
                        compressed["llm_output"],
                        compressed["status"],
                        event["event_id"],
                    ),
                )
            conn.commit()
            total_compressed += len(batch)
            logger.info("已压缩 %d 条事件", total_compressed)
        except Exception as e:
            logger.error("压缩批次失败（第 %d 条起）: %s", i, e)
            conn.rollback()

    logger.info("压缩完成，共压缩 %d 条事件", total_compressed)
    return total_compressed