"""
定时调度 — 上下文压缩定时任务
"""
import logging
import yaml
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler

from storage_layer.compression.compressor import run_compression

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _load_compression_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            return config.get("database", {}).get("compression", {})
    return {}


def _compression_job():
    """压缩任务执行入口"""
    config = _load_compression_config()
    retention_days = config.get("retention_days", 7)
    try:
        count = run_compression(retention_days=retention_days)
        logger.info("定时压缩任务完成，压缩 %d 条", count)
    except Exception as e:
        logger.error("定时压缩任务失败: %s", e)


def start_scheduler():
    """启动定时调度器"""
    global _scheduler
    if _scheduler is not None:
        return

    config = _load_compression_config()
    cron = config.get("cron", "0 3 * * *")
    minute, hour, day, month, day_of_week = cron.split()

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _compression_job,
        "cron",
        minute=minute,
        hour=hour,
        day=day,
        month=month,
        day_of_week=day_of_week,
        id="compression_job",
    )
    _scheduler.start()
    logger.info("压缩调度器已启动，cron: %s", cron)


def stop_scheduler():
    """停止定时调度器"""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("压缩调度器已停止")


def trigger_compression():
    """手动触发一次压缩（供测试/调试使用）"""
    logger.info("手动触发压缩任务")
    _compression_job()