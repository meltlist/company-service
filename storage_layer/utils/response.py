"""
统一响应格式
"""
from typing import Any, Optional


def success_response(
    data: Any = None, message: str = "success", code: int = 0
) -> dict:
    return {"code": code, "message": message, "data": data}


def error_response(
    code: int = 40000, message: str = "未知错误", data: Any = None
) -> dict:
    return {"code": code, "message": message, "data": data}