"""
测试 — 建表测试
"""
import os
import pytest

from storage_layer.database.connection import init_db, close_db, get_connection
from storage_layer.database.schema import create_tables


@pytest.fixture(autouse=True)
def setup_db():
    """每个测试用例使用独立的临时数据库"""
    db_path = "./data/test_schema.db"
    # 关闭已有连接
    try:
        close_db()
    except Exception:
        pass
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db(db_path)
    yield
    close_db()


class TestSchema:
    def test_create_tables(self):
        """建表应成功，重复执行不报错"""
        create_tables()
        create_tables()  # 幂等测试

        conn = get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r[0] for r in tables}
        assert "events" in table_names
        assert "room_model" in table_names
        assert "pending_requests" in table_names

    def test_events_table_columns(self):
        """events 表字段应完整"""
        create_tables()
        conn = get_connection()
        cols = conn.execute("PRAGMA table_info(events)").fetchall()
        col_names = {c[1] for c in cols}
        expected = {
            "event_id", "timestamp", "object_class", "action",
            "confidence", "image_path", "small_model_output",
            "llm_input", "llm_output", "final_action", "action_type",
            "status", "created_at", "updated_at",
        }
        assert expected.issubset(col_names)

    def test_room_model_id_constraint(self):
        """room_model 表 id 必须为 1"""
        create_tables()
        conn = get_connection()
        # id=1 应成功
        conn.execute(
            "INSERT INTO room_model (id, updated_at) VALUES (1, '2026-01-01T00:00:00')"
        )
        conn.commit()
        # id=2 应失败
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO room_model (id, updated_at) VALUES (2, '2026-01-01T00:00:00')"
            )

    def test_indexes_exist(self):
        """索引应存在"""
        create_tables()
        conn = get_connection()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {r[0] for r in indexes}
        assert "idx_events_timestamp" in index_names
        assert "idx_events_status" in index_names