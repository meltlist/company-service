"""
挂起请求 API 端点 — /api/pending_requests
"""
from fastapi import APIRouter

from storage_layer.dao.pending_dao import get_active_pending_request
from storage_layer.utils.response import success_response

router = APIRouter(prefix="/api/pending_requests", tags=["pending_requests"])


@router.get("")
async def get_active_pending():
    pending = get_active_pending_request()
    if pending is None:
        return success_response({"status": "no_pending"})
    return success_response(pending)