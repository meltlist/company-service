"""
房间模型 API 端点 — /api/room_model
"""
from fastapi import APIRouter

from storage_layer.dao.room_model_dao import get_room_model, save_room_model
from storage_layer.utils.response import success_response

router = APIRouter(prefix="/api/room_model", tags=["room_model"])


@router.get("")
async def get_room_model_api():
    model = get_room_model()
    if model is None:
        return success_response({
            "grid_data": None,
            "text_description": "尚未构建",
            "large_items": None,
        })
    return success_response(model)


@router.post("")
async def update_room_model_api(model: dict):
    save_room_model(
        model.get("grid_data", ""),
        model.get("text_description", ""),
        model.get("large_items", "[]"),
    )
    return success_response({"status": "ok"})