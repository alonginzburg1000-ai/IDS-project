from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any, Deque, Dict, List


ATTACK_TYPES = ("dos", "probe", "r2l", "u2r")


class TrafficStorage:
    def __init__(self, max_records: int = 1000):
        if max_records <= 0:
            raise ValueError("max_records must be positive.")
        self._max_records = max_records
        self._traffic: Deque[Dict[str, Any]] = deque(maxlen=max_records)
        self._suspicious: Deque[Dict[str, Any]] = deque(maxlen=max_records)
        self._attack_type_counts: Dict[str, int] = {attack_type: 0 for attack_type in ATTACK_TYPES}
        self._next_id = 1
        self._lock = Lock()

    def add(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            stored = dict(record)
            stored["id"] = self._next_id
            self._next_id += 1

            self._traffic.append(stored)
            if stored.get("binary_prediction") == "attack":
                self._suspicious.append(stored)
                attack_type = stored.get("attack_type")
                if attack_type in self._attack_type_counts:
                    self._attack_type_counts[attack_type] += 1
            return dict(stored)

    def all_traffic(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in reversed(self._traffic)]

    def suspicious_traffic(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in reversed(self._suspicious)]

    def attack_type_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._attack_type_counts)

    def attack_count(self) -> int:
        with self._lock:
            return len(self._suspicious)

    def traffic_count(self) -> int:
        with self._lock:
            return len(self._traffic)

