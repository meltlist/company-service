"""
事件 API 端点 — /api/events
"""
from fastapi import APIRouter, HTTPException, Query

from storage_layer.dao.event_dao import get_events, get_event_by_id
from storage_layer.utils.response import success_response, error_response

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("")
async def list_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str = Query(None),
):
    events, total = get_events(page, page_size, status)
    return success_response({
        "total": total,
        "page": page,
        "page_size": page_size,
        "events": events,
    })


@router.get("/{event_id}")
async def get_event(event_id: str):
    event = get_event_by_id(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="事件不存在")
    return success_response(event)