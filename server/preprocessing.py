from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


SUPPORTED_PROTOCOLS = {"tcp", "udp", "icmp"}

NUMERIC_FEATURES = [
    "0",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
]

PORT_SERVICE_MAP = {
    5: "rje",
    7: "echo",
    9: "discard",
    11: "systat",
    13: "daytime",
    20: "ftp_data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    37: "time",
    43: "whois",
    53: "domain",
    69: "tftp_u",
    70: "gopher",
    79: "finger",
    80: "http",
    95: "supdup",
    101: "hostnames",
    102: "iso_tsap",
    105: "csnet_ns",
    109: "pop_2",
    110: "pop_3",
    111: "sunrpc",
    113: "auth",
    117: "uucp_path",
    119: "nntp",
    123: "ntp_u",
    137: "netbios_ns",
    138: "netbios_dgm",
    139: "netbios_ssn",
    143: "imap4",
    179: "bgp",
    389: "ldap",
    443: "http_443",
    512: "exec",
    513: "login",
    514: "shell",
    515: "printer",
    540: "uucp",
    543: "klogin",
    544: "kshell",
    1521: "sql_net",
    2784: "http_2784",
    6000: "X11",
    6667: "IRC",
    8001: "http_8001",
    8080: "http",
}


class BadPacketError(ValueError):
    """Raised when a packet JSON cannot be converted into the runtime feature space."""


@dataclass(frozen=True)
class PacketFields:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    packet_len: int
    tcp_flags: str
    ttl: Optional[int]
    ip_len: int
    payload_len: int
    wrong_fragment: int
    urgent: int


def packet_json_to_normalized_vector(
    payload: Mapping[str, Any],
    feature_names: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    packet = validate_packet_json(payload)
    raw_features = build_raw_feature_map(packet)
    vector = align_feature_map(raw_features, feature_names)
    return normalize_vector(vector, mean, std)


def validate_packet_json(payload: Mapping[str, Any]) -> PacketFields:
    if not isinstance(payload, Mapping):
        raise BadPacketError("Request body must be a JSON object.")

    _require_keys(payload, ["timestamp", "src_ip", "dst_ip", "protocol", "packet_len"])

    protocol = str(payload.get("protocol", "")).strip().lower()
    if protocol not in SUPPORTED_PROTOCOLS:
        raise BadPacketError("protocol must be one of: tcp, udp, icmp.")

    src_ip = _parse_ipv4(payload["src_ip"], "src_ip")
    dst_ip = _parse_ipv4(payload["dst_ip"], "dst_ip")
    timestamp = _parse_float(payload["timestamp"], "timestamp")
    packet_len = _parse_non_negative_int(payload["packet_len"], "packet_len")
    ip_len = _parse_non_negative_int(payload.get("ip_len", packet_len), "ip_len")
    payload_len = _parse_non_negative_int(payload.get("payload_len", 0), "payload_len")
    ttl = _parse_optional_non_negative_int(payload.get("ttl"), "ttl")

    src_port = _parse_optional_port(payload.get("src_port"), "src_port")
    dst_port = _parse_optional_port(payload.get("dst_port"), "dst_port")
    if protocol in {"tcp", "udp"} and (src_port is None or dst_port is None):
        raise BadPacketError("{} packets require src_port and dst_port.".format(protocol))

    tcp_flags = str(payload.get("tcp_flags") or "")
    wrong_fragment = _parse_non_negative_int(payload.get("wrong_fragment", 0), "wrong_fragment")
    urgent = _parse_non_negative_int(payload.get("urgent", 0), "urgent")

    return PacketFields(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        packet_len=packet_len,
        tcp_flags=tcp_flags,
        ttl=ttl,
        ip_len=ip_len,
        payload_len=payload_len,
        wrong_fragment=wrong_fragment,
        urgent=urgent,
    )


def build_raw_feature_map(packet: PacketFields) -> Dict[str, float]:
    features = {name: 0.0 for name in NUMERIC_FEATURES}

    # NSL-KDD was trained on connection-level records. In v1 runtime, a single packet
    # can only approximate connection bytes; historical and host-window counters stay 0.
    features["0"] = 0.0
    features["4"] = float(packet.payload_len)
    features["5"] = 0.0
    features["6"] = float(derive_land(packet))
    features["7"] = float(packet.wrong_fragment)
    features["8"] = float(packet.urgent)

    features["1_{}".format(packet.protocol)] = 1.0
    features["2_{}".format(derive_service(packet.protocol, packet.src_port, packet.dst_port))] = 1.0
    features["3_{}".format(derive_flag(packet.protocol, packet.tcp_flags))] = 1.0
    return features


def align_feature_map(raw_features: Mapping[str, float], feature_names: Sequence[str]) -> np.ndarray:
    vector = np.zeros((1, len(feature_names)), dtype=np.float32)
    for idx, feature_name in enumerate(feature_names):
        vector[0, idx] = float(raw_features.get(str(feature_name), 0.0))
    return vector


def normalize_vector(vector: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if vector.shape[1] != mean.shape[1] or vector.shape[1] != std.shape[1]:
        raise ValueError(
            "Feature vector shape {} does not match preprocess shapes mean={} std={}.".format(
                vector.shape, mean.shape, std.shape
            )
        )
    return (vector.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)


def derive_service(protocol: str, src_port: Optional[int], dst_port: Optional[int]) -> str:
    for port in (dst_port, src_port):
        if port is None:
            continue
        service = PORT_SERVICE_MAP.get(int(port))
        if service:
            return service
    return "other"


def derive_flag(protocol: str, tcp_flags: str) -> str:
    if protocol != "tcp":
        return "OTH"

    flags = _normalize_tcp_flags(tcp_flags)
    if "R" in flags and "A" in flags:
        return "REJ"
    if "R" in flags:
        return "RSTO"
    if "S" in flags and "A" not in flags:
        return "S0"
    if "S" in flags and "A" in flags:
        return "S1"
    if "A" in flags or "P" in flags or "F" in flags:
        return "SF"
    return "OTH"


def derive_land(packet: PacketFields) -> int:
    if packet.src_port is None or packet.dst_port is None:
        return 0
    return int(packet.src_ip == packet.dst_ip and packet.src_port == packet.dst_port)


def _normalize_tcp_flags(tcp_flags: str) -> str:
    flags = str(tcp_flags or "").upper()
    if flags.isdigit():
        value = int(flags)
        decoded = ""
        if value & 0x01:
            decoded += "F"
        if value & 0x02:
            decoded += "S"
        if value & 0x04:
            decoded += "R"
        if value & 0x08:
            decoded += "P"
        if value & 0x10:
            decoded += "A"
        if value & 0x20:
            decoded += "U"
        return decoded
    return "".join(ch for ch in flags if ch in {"F", "S", "R", "P", "A", "U", "E", "C"})


def _require_keys(payload: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise BadPacketError("Missing required packet fields: {}".format(", ".join(missing)))


def _parse_ipv4(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BadPacketError("{} must be a non-empty IPv4 string.".format(field_name))
    try:
        ip_value = ipaddress.ip_address(value.strip())
    except ValueError:
        raise BadPacketError("{} is not a valid IP address.".format(field_name))
    if ip_value.version != 4:
        raise BadPacketError("{} must be IPv4 for this NSL-KDD runtime.".format(field_name))
    return str(ip_value)


def _parse_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise BadPacketError("{} must be numeric.".format(field_name))


def _parse_non_negative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise BadPacketError("{} must be an integer.".format(field_name))
    if parsed < 0:
        raise BadPacketError("{} must be non-negative.".format(field_name))
    return parsed


def _parse_optional_non_negative_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _parse_non_negative_int(value, field_name)


def _parse_optional_port(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    port = _parse_non_negative_int(value, field_name)
    if port > 65535:
        raise BadPacketError("{} must be between 0 and 65535.".format(field_name))
    return port

