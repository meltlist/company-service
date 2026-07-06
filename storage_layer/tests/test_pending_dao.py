"""
测试 — 挂起请求 DAO
"""
import os
import pytest

from storage_layer.database.connection import init_db, close_db
from storage_layer.database.schema import create_tables
from storage_layer.dao.event_dao import save_event
from storage_layer.dao.pending_dao import (
    save_pending_request,
    get_pending_request,
    get_active_pending_request,
    resolve_pending_request,
    get_pending_by_event,
)


@pytest.fixture(autouse=True)
def setup_db():
    db_path = "./data/test_pending_dao.db"
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


class TestPendingDAO:
    def _create_event(self, event_id: str):
        """创建测试用事件，满足外键约束"""
        save_event({
            "event_id": event_id,
            "timestamp": "2026-07-06T10:00:00",
        })

    def _make_pending(self, request_id: str, event_id: str = None):
        """创建挂起请求的辅助方法，自动创建关联事件"""
        if event_id:
            self._create_event(event_id)
        return save_pending_request({
            "request_id": request_id,
            "event_id": event_id,
            "action_type": "confirm",
            "action_payload": '{"action": "test"}',
        })

    def test_save_and_get(self):
        """创建后应能正确读出"""
        self._make_pending("req_001", "evt_001")
        result = get_pending_request("req_001")
        assert result is not None
        assert result["action_type"] == "confirm"
        assert result["status"] == "awaiting_user"

    def test_get_active_pending(self):
        """获取活跃请求应返回最新的 awaiting_user"""
        import time
        self._make_pending("req_001", "evt_001")
        time.sleep(1.1)  # 确保 created_at 时间戳不同
        self._make_pending("req_002", "evt_002")
        active = get_active_pending_request()
        assert active["request_id"] == "req_002"

    def test_resolve_pending(self):
        """解决挂起请求后状态应更新"""
        self._make_pending("req_001", "evt_001")
        assert resolve_pending_request("req_001", "executed")
        result = get_pending_request("req_001")
        assert result["status"] == "executed"
        assert result["resolved_at"] is not None

    def test_resolve_invalid_status(self):
        """非法状态应抛出异常"""
        self._make_pending("req_001", "evt_001")
        with pytest.raises(ValueError, match="非法状态"):
            resolve_pending_request("req_001", "invalid")

    def test_resolve_nonexistent(self):
        """解决不存在的请求应返回 False"""
        assert not resolve_pending_request("nonexistent", "executed")

    def test_get_pending_by_event(self):
        """根据事件 ID 查找关联请求"""
        self._make_pending("req_001", "evt_001")
        result = get_pending_by_event("evt_001")
        assert result is not None
        assert result["request_id"] == "req_001"

    def test_validation_rejects_invalid(self):
        """非法数据应抛出异常"""
        with pytest.raises(ValueError, match="request_id"):
            save_pending_request({"action_type": "confirm", "action_payload": "{}"})

        with pytest.raises(ValueError, match="action_type"):
            save_pending_request({
                "request_id": "req_001",
                "action_type": "invalid",
                "action_payload": "{}",
            })

        with pytest.raises(ValueError, match="action_payload"):
            save_pending_request({
                "request_id": "req_001",
                "action_type": "confirm",
            })