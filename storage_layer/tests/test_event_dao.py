"""
测试 — 事件 DAO
"""
import os
import pytest

from storage_layer.database.connection import init_db, close_db
from storage_layer.database.schema import create_tables
from storage_layer.dao.event_dao import (
    save_event,
    update_event,
    get_events,
    get_event_by_id,
    get_recent_events,
)


@pytest.fixture(autouse=True)
def setup_db():
    db_path = "./data/test_event_dao.db"
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


class TestEventDAO:
    def test_save_and_get(self):
        """写入后应能正确读出"""
        event = {
            "event_id": "evt_20260706_143000_001",
            "timestamp": "2026-07-06T14:30:00",
            "object_class": "person",
            "action": "walking",
            "confidence": 0.95,
            "image_path": "/data/frames/evt_001.jpg",
            "small_model_output": '{"objects": ["person"]}',
        }
        save_event(event)
        result = get_event_by_id("evt_20260706_143000_001")
        assert result is not None
        assert result["event_id"] == "evt_20260706_143000_001"
        assert result["object_class"] == "person"
        assert result["confidence"] == 0.95

    def test_update_event(self):
        """更新部分字段后应能正确读出"""
        event = {
            "event_id": "evt_20260706_150000_001",
            "timestamp": "2026-07-06T15:00:00",
        }
        save_event(event)

        updates = {
            "llm_input": "用户输入测试",
            "llm_output": '{"response": "ok"}',
            "action_type": "inform",
            "status": "executed",
        }
        assert update_event("evt_20260706_150000_001", updates)

        result = get_event_by_id("evt_20260706_150000_001")
        assert result["llm_input"] == "用户输入测试"
        assert result["llm_output"] == '{"response": "ok"}'
        assert result["action_type"] == "inform"
        assert result["status"] == "executed"

    def test_update_nonexistent(self):
        """更新不存在的事件应返回 False"""
        assert not update_event("nonexistent_id", {"status": "executed"})

    def test_get_events_pagination(self):
        """分页查询应正确"""
        for i in range(25):
            save_event({
                "event_id": f"evt_20260706_{i:04d}",
                "timestamp": f"2026-07-06T10:{i:02d}:00",
            })

        events, total = get_events(page=1, page_size=10)
        assert total == 25
        assert len(events) == 10

        events, _ = get_events(page=3, page_size=10)
        assert len(events) == 5

    def test_get_events_by_status(self):
        """按状态过滤应正确"""
        for i in range(5):
            save_event({
                "event_id": f"evt_executed_{i:04d}",
                "timestamp": f"2026-07-06T10:{i:02d}:00",
                "status": "executed",
            })
        save_event({
            "event_id": "evt_pending_001",
            "timestamp": "2026-07-06T11:00:00",
            "status": "pending",
        })

        events, total = get_events(status="executed")
        assert total == 5
        assert len(events) == 5

    def test_get_recent_events(self):
        """获取最近事件应正确"""
        for i in range(15):
            save_event({
                "event_id": f"evt_recent_{i:04d}",
                "timestamp": f"2026-07-06T10:{i:02d}:00",
            })
        result = get_recent_events(limit=10)
        assert len(result) == 10

    def test_validation_rejects_invalid(self):
        """非法数据应抛出异常"""
        with pytest.raises(ValueError, match="event_id"):
            save_event({"timestamp": "2026-07-06T10:00:00"})

        with pytest.raises(ValueError, match="timestamp"):
            save_event({"event_id": "evt_001"})

        with pytest.raises(ValueError, match="confidence"):
            save_event({
                "event_id": "evt_001",
                "timestamp": "2026-07-06T10:00:00",
                "confidence": 1.5,
            })

        with pytest.raises(ValueError, match="action_type"):
            update_event("evt_001", {"action_type": "invalid"})