"""
测试 — 上下文压缩
"""
import os
import pytest
from datetime import datetime, timedelta

from storage_layer.database.connection import init_db, close_db, get_connection
from storage_layer.database.schema import create_tables
from storage_layer.dao.event_dao import save_event, update_event, get_event_by_id
from storage_layer.compression.compressor import run_compression, compress_event


@pytest.fixture(autouse=True)
def setup_db():
    db_path = "./data/test_compression.db"
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


class TestCompression:
    def _make_old_event(self, event_id: str, days_ago: int):
        ts = (datetime.now() - timedelta(days=days_ago)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        save_event({
            "event_id": event_id,
            "timestamp": ts,
            "object_class": "person",
            "action": "walking",
            "confidence": 0.9,
            "small_model_output": '{"detections": [{"class": "person", "confidence": 0.9}]}',
        })
        # llm_input / llm_output / action_type / status 由后续模块填充
        update_event(event_id, {
            "llm_input": "一段很长的用户输入文本" * 20,
            "llm_output": '{"response": "一段很长的LLM返回文本"}' * 20,
            "action_type": "inform",
            "status": "executed",
        })

    def test_compress_event_format(self):
        """压缩后摘要格式正确，大字段被清空"""
        event = {
            "event_id": "evt_001",
            "timestamp": "2026-06-01T10:00:00",
            "object_class": "person",
            "action": "walking",
            "action_type": "inform",
            "final_action": "语音播报",
        }
        result = compress_event(event)
        assert "2026-06-01T10:00:00" in result["small_model_output"]
        assert "person" in result["small_model_output"]
        assert "walking" in result["small_model_output"]
        assert result["llm_input"] is None
        assert result["llm_output"] is None
        assert result["status"] == "compressed"

    def test_compression_only_affects_old_events(self):
        """压缩只影响超过7天的事件，7天内不受影响"""
        # 插入 10 条 8 天前的事件
        for i in range(10):
            self._make_old_event(f"evt_old_{i:04d}", days_ago=8)
        # 插入 5 条 3 天前的事件
        for i in range(5):
            self._make_old_event(f"evt_recent_{i:04d}", days_ago=3)

        count = run_compression(retention_days=7)
        assert count == 10

        # 旧事件应被压缩
        for i in range(10):
            evt = get_event_by_id(f"evt_old_{i:04d}")
            assert evt["status"] == "compressed"
            assert evt["llm_input"] is None
            assert evt["llm_output"] is None

        # 近期事件不应受影响
        for i in range(5):
            evt = get_event_by_id(f"evt_recent_{i:04d}")
            assert evt["status"] == "executed"
            assert evt["llm_input"] is not None

    def test_compression_skips_already_compressed(self):
        """已压缩的事件应跳过"""
        self._make_old_event("evt_old_001", days_ago=8)
        # 手动标记为已压缩
        conn = get_connection()
        conn.execute(
            "UPDATE events SET status = 'compressed' WHERE event_id = 'evt_old_001'"
        )
        conn.commit()

        count = run_compression(retention_days=7)
        assert count == 0

    def test_compression_empty(self):
        """无过期事件时返回 0"""
        count = run_compression(retention_days=7)
        assert count == 0