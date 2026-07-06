"""
数据校验 — 所有 DAO 写入操作的入口校验
"""
import re
from datetime import datetime

VALID_ACTION_TYPES = {"none", "inform", "confirm", "emergency"}
VALID_STATUSES = {
    "recorded", "pending", "executed", "notified", "cancelled", "timeout", "compressed",
}
VALID_PENDING_STATUSES = {"awaiting_user", "executed", "cancelled", "timeout"}

ISO8601_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
)


def validate_event(event: dict) -> None:
    """校验事件写入数据"""
    if not event.get("event_id"):
        raise ValueError("event_id 不能为空")

    ts = event.get("timestamp")
    if not ts:
        raise ValueError("timestamp 不能为空")
    if not ISO8601_PATTERN.match(ts):
        raise ValueError(f"timestamp 格式非法（需 ISO 8601）: {ts}")

    confidence = event.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)):
            raise ValueError("confidence 必须是数值")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence 必须在 0.0–1.0 之间: {confidence}")

    status = event.get("status")
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status 非法: {status}，合法值: {VALID_STATUSES}")


def validate_event_update(updates: dict) -> None:
    """校验事件更新数据"""
    if "action_type" in updates and updates["action_type"] is not None:
        if updates["action_type"] not in VALID_ACTION_TYPES:
            raise ValueError(
                f"action_type 非法: {updates['action_type']}，"
                f"合法值: {VALID_ACTION_TYPES}"
            )

    if "status" in updates and updates["status"] is not None:
        if updates["status"] not in VALID_STATUSES:
            raise ValueError(
                f"status 非法: {updates['status']}，合法值: {VALID_STATUSES}"
            )

    if "confidence" in updates and updates["confidence"] is not None:
        c = updates["confidence"]
        if not isinstance(c, (int, float)) or not (0.0 <= c <= 1.0):
            raise ValueError(f"confidence 必须在 0.0–1.0 之间: {c}")


def validate_pending_request(request: dict) -> None:
    """校验挂起请求数据"""
    if not request.get("request_id"):
        raise ValueError("request_id 不能为空")

    action_type = request.get("action_type")
    if not action_type:
        raise ValueError("action_type 不能为空")
    if action_type not in VALID_ACTION_TYPES:
        raise ValueError(
            f"action_type 非法: {action_type}，合法值: {VALID_ACTION_TYPES}"
        )

    if not request.get("action_payload"):
        raise ValueError("action_payload 不能为空")

    status = request.get("status", "awaiting_user")
    if status not in VALID_PENDING_STATUSES:
        raise ValueError(
            f"status 非法: {status}，合法值: {VALID_PENDING_STATUSES}"
        )