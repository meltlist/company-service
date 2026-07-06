"""
测试 — 房间模型 DAO
"""
import os
import pytest

from storage_layer.database.connection import init_db, close_db
from storage_layer.database.schema import create_tables
from storage_layer.dao.room_model_dao import save_room_model, get_room_model


@pytest.fixture(autouse=True)
def setup_db():
    db_path = "./data/test_room_model_dao.db"
    try:
        close_db()
    except Exception:
        pass
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db(db_path)
    create_tables()
    yield
    close_db()


class TestRoomModelDAO:
    def test_save_and_get(self):
        """保存后应能正确读出"""
        save_room_model(
            grid_data='{"grid": [[0,1],[1,0]]}',
            text_description="房间西北角有床，东南角有桌子",
            large_items='[{"name": "床", "position": "西北角", "confirmed": true}]',
        )
        model = get_room_model()
        assert model is not None
        assert model["id"] == 1
        assert "床" in model["text_description"]
        assert "西北角" in model["large_items"]

    def test_overwrite(self):
        """INSERT OR REPLACE 应覆盖旧数据"""
        save_room_model("grid_v1", "desc_v1", "[]")
        save_room_model("grid_v2", "desc_v2", "[]")
        model = get_room_model()
        assert model["grid_data"] == "grid_v2"
        assert model["text_description"] == "desc_v2"

    def test_get_empty_returns_none(self):
        """无数据时应返回 None"""
        assert get_room_model() is None