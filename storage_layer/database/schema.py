"""
建表 DDL
"""
import logging
from .connection import get_connection

logger = logging.getLogger(__name__)

DDL_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    object_class    TEXT,
    action          TEXT,
    confidence      REAL,
    image_path      TEXT,
    small_model_output  TEXT,
    llm_input       TEXT,
    llm_output      TEXT,
    final_action    TEXT,
    action_type     TEXT,
    status          TEXT,
    created_at      TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at      TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
"""

DDL_ROOM_MODEL = """
CREATE TABLE IF NOT EXISTS room_model (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    updated_at      TEXT NOT NULL,
    grid_data       TEXT,
    text_description TEXT,
    large_items     TEXT
);
"""

DDL_PENDING_REQUESTS = """
CREATE TABLE IF NOT EXISTS pending_requests (
    request_id      TEXT PRIMARY KEY,
    event_id        TEXT,
    action_type     TEXT NOT NULL,
    action_payload  TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'awaiting_user',
    created_at      TEXT DEFAULT (datetime('now', 'localtime')),
    resolved_at     TEXT,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
"""


def create_tables() -> None:
    """创建所有表（幂等，重复执行不会报错）"""
    conn = get_connection()
    conn.executescript(DDL_EVENTS)
    conn.executescript(DDL_ROOM_MODEL)
    conn.executescript(DDL_PENDING_REQUESTS)
    conn.commit()
    logger.info("数据库表初始化完成")