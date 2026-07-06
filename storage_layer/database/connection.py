"""
数据库连接管理（单例模式，WAL 模式）
"""
import sqlite3
import os
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_connection: sqlite3.Connection | None = None
_lock = threading.Lock()
_db_path: str = ""


def _load_config() -> dict:
    """加载 config.yaml 获取数据库路径"""
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """
    初始化数据库连接（单例）。
    如果 db_path 未指定，从 config.yaml 读取。
    自动创建 data/ 目录。
    """
    global _connection, _db_path

    with _lock:
        if _connection is not None:
            return _connection

        if db_path is None:
            config = _load_config()
            db_path = config.get("database", {}).get("path", "./data/jarvis_phase1.db")

        _db_path = db_path

        # 确保 data/ 目录存在
        data_dir = os.path.dirname(db_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)

        _connection = sqlite3.connect(db_path, check_same_thread=False)
        _connection.row_factory = sqlite3.Row

        # 开启 WAL 模式，提升并发读写性能
        _connection.execute("PRAGMA journal_mode=WAL;")
        _connection.execute("PRAGMA foreign_keys=ON;")

        logger.info("数据库连接已建立: %s (WAL 模式)", db_path)
        return _connection


def get_connection() -> sqlite3.Connection:
    """获取当前数据库连接，未初始化时自动初始化"""
    global _connection
    if _connection is None:
        return init_db()
    return _connection


def close_db() -> None:
    """关闭数据库连接"""
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None
            logger.info("数据库连接已关闭")