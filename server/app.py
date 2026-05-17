from __future__ import annotations

import argparse
import atexit
import json
import logging
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, redirect, render_template, request, url_for

from agent.agent import run_agent
from server.config import (
    RuntimeConfig,
    build_json_file_logger,
    configure_logging,
    load_config,
)
from server.inference import InferenceEngine
from server.model_loader import load_runtime_models
from server.preprocessing import BadPacketError
from server.storage import TrafficStorage

"""
מאתחלת את שרת ה-Flask,
 טוענת את מודלי הבינה המלאכותית ומגדירה את נתיבי ה-API
"""
def create_app(config: RuntimeConfig, start_agent: bool = False, agent_allowed: bool = True) -> Flask:
    logger = configure_logging(config, "ids.server", "server.log")
    attack_logger = build_json_file_logger(config, "ids.attack", "attacks.log")
    models = load_runtime_models(config.artifacts_path)
    engine = InferenceEngine(models, binary_threshold=config.binary_threshold)
    storage = TrafficStorage(max_records=config.traffic_store_limit)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["IDS_CONFIG"] = config
    app.config["IDS_ENGINE"] = engine
    app.config["IDS_STORAGE"] = storage
    app.config["IDS_LOGGER"] = logger
    app.config["IDS_ATTACK_LOGGER"] = attack_logger
    app.config["IDS_AGENT_THREAD"] = None
    app.config["IDS_AGENT_STOP_EVENT"] = None
    app.config["IDS_AGENT_LOCK"] = Lock()
    app.config["IDS_AGENT_ALLOWED"] = config.agent_enabled and agent_allowed
    app.config["IDS_AGENT_ATEXIT_REGISTERED"] = False

    @app.get("/")
    def index() -> Any:
        return redirect(url_for("dashboard"))

    @app.get("/dashboard")
    def dashboard() -> str:
        return render_template("dashboard.html")

    @app.get("/health")
    def health() -> Tuple[Any, int]:
        return jsonify(
            {
                "status": "ok",
                "models_loaded": True,
                "binary_threshold": config.binary_threshold,
                "traffic_count": storage.traffic_count(),
                "attack_count": storage.attack_count(),
                "agent_enabled": config.agent_enabled,
                "agent_running": _agent_thread_alive(app),
                "sniffer": _sniffer_status(app),
            }
        ), 200

    @app.get("/api/traffic")
    def api_traffic() -> Tuple[Any, int]:
        return jsonify({"records": storage.all_traffic()}), 200

    @app.get("/api/suspicious")
    def api_suspicious() -> Tuple[Any, int]:
        return jsonify({"records": storage.suspicious_traffic()}), 200

    @app.get("/api/stats/attack-types")
    def api_attack_type_stats() -> Tuple[Any, int]:
        return jsonify({"attack_types": storage.attack_type_counts()}), 200

    @app.get("/api/sniffer/status")
    def api_sniffer_status() -> Tuple[Any, int]:
        return jsonify(_sniffer_status(app)), 200

    @app.post("/api/sniffer/start")
    def api_sniffer_start() -> Tuple[Any, int]:
        status, status_code = start_background_agent(app)
        return jsonify(status), status_code

    @app.post("/api/sniffer/stop")
    def api_sniffer_stop() -> Tuple[Any, int]:
        status, status_code = stop_background_agent(app)
        return jsonify(status), status_code

    @app.post("/predict")
    def predict() -> Tuple[Any, int]:
        payload = request.get_json(silent=True)
        logger.info("incoming_request endpoint=/predict remote_addr=%s", request.remote_addr)

        if payload is None:
            logger.warning("bad_request reason=invalid_json")
            return jsonify({"error": "Request body must be valid JSON."}), 400

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("predict_payload=%s", json.dumps(payload, ensure_ascii=False, sort_keys=True))

        try:
            result, _ = process_packet_payload(
                payload=payload,
                engine=engine,
                storage=storage,
                logger=logger,
                attack_logger=attack_logger,
                source="api",
            )
        except BadPacketError as exc:
            logger.warning("bad_request reason=%s", exc)
            return jsonify({"error": str(exc)}), 400
        except Exception:
            logger.exception("prediction_error")
            return jsonify({"error": "Prediction failed."}), 500

        return jsonify(result), 200

    if start_agent and app.config["IDS_AGENT_ALLOWED"]:
        start_background_agent(app)

    return app

"""
מקבלת נתוני חבילת תקשורת,
 מריצה עליהם את מודל ה-AI כדי לזהות מתקפות, 
 שומרת את התוצאה בבסיס הנתונים ומפעילה לוגים בהתאם.
"""
def process_packet_payload(
    payload: Dict[str, Any],
    engine: InferenceEngine,
    storage: TrafficStorage,
    logger: logging.Logger,
    attack_logger: logging.Logger,
    source: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    start = time.perf_counter()
    result = engine.predict(payload)
    response_time_ms = _elapsed_ms(start)

    record = _build_traffic_record(payload, result, response_time_ms, source)
    stored_record = storage.add(record)

    if result["binary_prediction"] == "attack":
        _log_attack(attack_logger, stored_record)

    logger.info(
        "model_decision source=%s record_id=%s binary_prediction=%s binary_confidence=%.6f "
        "attack_type=%s multiclass_confidence=%s response_time_ms=%.2f attack_count=%d",
        source,
        stored_record["id"],
        result["binary_prediction"],
        result["binary_confidence"],
        result["attack_type"],
        _format_optional_float(result["multiclass_confidence"]),
        response_time_ms,
        storage.attack_count(),
    )
    return result, stored_record

"""
מריצה בתוך תהליכון (Thread) נפרד וברקע את סוכן ה-Scapy שמסניף חבילות תקשורת בזמן אמת,
 תוך הגנה מפני הרצות כפולות בעזרת Lock.
"""
def start_background_agent(app: Flask) -> Tuple[Dict[str, Any], int]:
    config: RuntimeConfig = app.config["IDS_CONFIG"]
    logger: logging.Logger = app.config["IDS_LOGGER"]
    agent_lock: Lock = app.config["IDS_AGENT_LOCK"]

    with agent_lock:
        if not app.config["IDS_AGENT_ALLOWED"]:
            return _sniffer_status(app, message="sniffer is disabled"), 403

        if _agent_thread_alive(app):
            return _sniffer_status(app, message="sniffer already running"), 200

        agent_logger = configure_logging(config, "ids.agent", "agent.log")
        stop_event = Event()

        def packet_callback(fields: Dict[str, Any]) -> None:
            try:
                process_packet_payload(
                    payload=fields,
                    engine=app.config["IDS_ENGINE"],
                    storage=app.config["IDS_STORAGE"],
                    logger=logger,
                    attack_logger=app.config["IDS_ATTACK_LOGGER"],
                    source="agent",
                )
            except BadPacketError as exc:
                logger.warning("agent_bad_packet reason=%s", exc)
            except Exception:
                logger.exception("agent_prediction_error")

        def runner() -> None:
            agent_logger.info("agent_thread_start")
            run_agent(config=config, packet_callback=packet_callback, stop_event=stop_event)
            agent_logger.info("agent_thread_stop")

        thread = Thread(target=runner, name="ids-scapy-agent", daemon=True)
        app.config["IDS_AGENT_THREAD"] = thread
        app.config["IDS_AGENT_STOP_EVENT"] = stop_event
        thread.start()

        if not app.config["IDS_AGENT_ATEXIT_REGISTERED"]:
            atexit.register(stop_background_agent, app)
            app.config["IDS_AGENT_ATEXIT_REGISTERED"] = True

        return _sniffer_status(app, message="sniffer started"), 200


"""
עוצרת בצורה בטוחה את תהליכון הסנפת
 החבילות על ידי סימון אירוע עצירה (Stop Event) 
 והמתנה מוגדרת מראש לסיום הריצה שלו.
"""
def stop_background_agent(app: Flask) -> Tuple[Dict[str, Any], int]:
    agent_lock: Optional[Lock] = app.config.get("IDS_AGENT_LOCK")
    if agent_lock is None:
        return _sniffer_status(app, message="sniffer was not initialized"), 200

    with agent_lock:
        thread: Optional[Thread] = app.config.get("IDS_AGENT_THREAD")
        stop_event: Optional[Event] = app.config.get("IDS_AGENT_STOP_EVENT")
        if not thread or not thread.is_alive():
            app.config["IDS_AGENT_THREAD"] = None
            app.config["IDS_AGENT_STOP_EVENT"] = None
            return _sniffer_status(app, message="sniffer already stopped"), 200

        if stop_event is not None:
            stop_event.set()

        thread.join(timeout=2)
        if thread.is_alive():
            return _sniffer_status(app, message="sniffer stopping"), 202

        app.config["IDS_AGENT_THREAD"] = None
        app.config["IDS_AGENT_STOP_EVENT"] = None
        return _sniffer_status(app, message="sniffer stopped"), 200


def _agent_thread_alive(app: Flask) -> bool:
    thread: Optional[Thread] = app.config.get("IDS_AGENT_THREAD")
    return bool(thread and thread.is_alive())


def _sniffer_status(app: Flask, message: Optional[str] = None) -> Dict[str, Any]:
    sniffing = _agent_thread_alive(app)
    status = "active" if sniffing else "stopped"
    payload = {
        "enabled": bool(app.config.get("IDS_AGENT_ALLOWED", False)),
        "sniffing": sniffing,
        "status": status,
        "display_status": "הסנפה פעילה" if sniffing else "הסנפה עצורה",
    }
    if message is not None:
        payload["message"] = message
    return payload

"""
פונקציית עזר המארגנת ומאחדת את נתוני חבילת התקשורת ה
גולמיים יחד עם תוצאות הניתוח של ה-AI למבנה נתונים אחיד

"""
def _build_traffic_record(
    payload: Dict[str, Any],
    result: Dict[str, Any],
    response_time_ms: float,
    source: str,
) -> Dict[str, Any]:
    return {
        "source": source,
        "timestamp": payload.get("timestamp"),
        "src_ip": payload.get("src_ip"),
        "dst_ip": payload.get("dst_ip"),
        "src_port": payload.get("src_port"),
        "dst_port": payload.get("dst_port"),
        "protocol": payload.get("protocol"),
        "packet_len": payload.get("packet_len"),
        "tcp_flags": payload.get("tcp_flags"),
        "ttl": payload.get("ttl"),
        "ip_len": payload.get("ip_len"),
        "payload_len": payload.get("payload_len"),
        "wrong_fragment": payload.get("wrong_fragment"),
        "urgent": payload.get("urgent"),
        "binary_prediction": result.get("binary_prediction"),
        "binary_confidence": result.get("binary_confidence"),
        "attack_type": result.get("attack_type"),
        "multiclass_confidence": result.get("multiclass_confidence"),
        "response_time_ms": response_time_ms,
    }

"""
פונקציית עזר השולפת את נתוני המתקפה מתוך רשומת 
התעבורה וכותבת אותם בצורה 
מרוכזת ופורמלית לקובץ לוג ייעודי למתקפות (attacks.log).
"""
def _log_attack(attack_logger: logging.Logger, record: Dict[str, Any]) -> None:
    attack_record = {
        "timestamp": record.get("timestamp"),
        "src_ip": record.get("src_ip"),
        "dst_ip": record.get("dst_ip"),
        "src_port": record.get("src_port"),
        "dst_port": record.get("dst_port"),
        "protocol": record.get("protocol"),
        "binary_confidence": record.get("binary_confidence"),
        "attack_type": record.get("attack_type"),
        "multiclass_confidence": record.get("multiclass_confidence"),
        "response_time_ms": record.get("response_time_ms"),
    }
    attack_logger.info(json.dumps(attack_record, ensure_ascii=False, sort_keys=True))


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "null"
    return "{:.6f}".format(float(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IDS runtime Flask dashboard server")
    parser.add_argument("--config", default="config.json", help="Path to runtime config.json")
    parser.add_argument("--no-agent", action="store_true", help="Start Flask without the embedded Scapy agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    app = create_app(config, start_agent=False, agent_allowed=not args.no_agent)
    try:
        app.run(host=config.flask_host, port=config.flask_port, use_reloader=False)
    finally:
        stop_background_agent(app)


if __name__ == "__main__":
    main()
