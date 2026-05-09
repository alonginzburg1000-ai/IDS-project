from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CONFIG_PATH = "config.json"


@dataclass(frozen=True)
class RuntimeConfig:
    flask_host: str
    flask_port: int
    binary_threshold: float
    scapy_interface: Optional[str]
    packet_limit: Optional[int]
    log_level: str
    artifacts_path: Path
    agent_server_url: str
    agent_enabled: bool
    request_timeout_seconds: float
    logs_path: Path
    traffic_store_limit: int
    config_path: Path


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> RuntimeConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError("Missing runtime config: {}".format(path))

    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    base_dir = path.parent
    artifacts_path = _resolve_path(str(raw.get("artifacts_path", "artifacts")), base_dir)
    logs_path = _resolve_path(str(raw.get("logs_path", "logs")), base_dir)

    flask_host = str(raw.get("flask_host", "127.0.0.1"))
    flask_port = int(raw.get("flask_port", 5000))
    binary_threshold = float(raw.get("binary_threshold", 0.55))
    if not 0.0 <= binary_threshold <= 1.0:
        raise ValueError("binary_threshold must be between 0 and 1.")

    packet_limit_raw = raw.get("packet_limit", 0)
    packet_limit = int(packet_limit_raw) if packet_limit_raw is not None else 0
    if packet_limit <= 0:
        packet_limit = None

    scapy_interface_raw = raw.get("scapy_interface")
    scapy_interface = str(scapy_interface_raw).strip() if scapy_interface_raw else None
    agent_server_url = str(raw.get("agent_server_url") or _default_server_url(flask_host, flask_port))

    return RuntimeConfig(
        flask_host=flask_host,
        flask_port=flask_port,
        binary_threshold=binary_threshold,
        scapy_interface=scapy_interface,
        packet_limit=packet_limit,
        log_level=str(raw.get("log_level", "INFO")).upper(),
        artifacts_path=artifacts_path,
        agent_server_url=agent_server_url,
        agent_enabled=bool(raw.get("agent_enabled", True)),
        request_timeout_seconds=float(raw.get("request_timeout_seconds", 3.0)),
        logs_path=logs_path,
        traffic_store_limit=int(raw.get("traffic_store_limit", 1000)),
        config_path=path,
    )


def configure_logging(config: RuntimeConfig, logger_name: str, log_file_name: str) -> logging.Logger:
    config.logs_path.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, config.log_level, logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    _clear_handlers(logger)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(config.logs_path / log_file_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger


def build_json_file_logger(
    config: RuntimeConfig,
    logger_name: str,
    log_file_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    config.logs_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    _clear_handlers(logger)

    file_handler = logging.FileHandler(config.logs_path / log_file_name, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _default_server_url(flask_host: str, flask_port: int) -> str:
    host = flask_host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return "http://{}:{}/predict".format(host, flask_port)


def _clear_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
