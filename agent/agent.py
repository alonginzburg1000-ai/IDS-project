from __future__ import annotations

import argparse
import ipaddress
import logging
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Optional, Set
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from scapy.all import ARP, ICMP, IP, TCP, UDP, Ether, sniff

from server.config import RuntimeConfig, configure_logging, load_config


LOGGER = logging.getLogger("ids.agent")
PacketCallback = Callable[[Dict[str, Any]], None]


@dataclass(frozen=True)
class ServerEndpoint:
    host: Optional[str]
    port: Optional[int]
    ips: Set[str]


def packet_to_raw_fields(packet: Any, endpoint: ServerEndpoint) -> Optional[Dict[str, Any]]:
    if ARP in packet or IP not in packet:
        return None

    ip_layer = packet[IP]
    src_ip = str(ip_layer.src)
    dst_ip = str(ip_layer.dst)
    if _is_ignored_ip(src_ip) or _is_ignored_ip(dst_ip):
        return None
    if _is_ethernet_broadcast(packet):
        return None

    protocol: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    tcp_flags = ""
    payload_len = 0
    urgent = 0

    if TCP in packet:
        tcp = packet[TCP]
        protocol = "tcp"
        src_port = int(tcp.sport)
        dst_port = int(tcp.dport)
        tcp_flags = str(tcp.flags)
        payload_len = len(tcp.payload)
        urgent = 1 if int(tcp.urgptr or 0) > 0 or "U" in tcp_flags.upper() else 0
    elif UDP in packet:
        udp = packet[UDP]
        protocol = "udp"
        src_port = int(udp.sport)
        dst_port = int(udp.dport)
        payload_len = len(udp.payload)
    elif ICMP in packet:
        icmp = packet[ICMP]
        protocol = "icmp"
        payload_len = len(icmp.payload)
    else:
        return None

    fields = {
        "timestamp": float(getattr(packet, "time", time.time())),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": protocol,
        "packet_len": int(len(packet)),
        "tcp_flags": tcp_flags,
        "ttl": int(getattr(ip_layer, "ttl", 0)),
        "ip_len": int(getattr(ip_layer, "len", 0) or len(ip_layer)),
        "payload_len": int(payload_len),
        "wrong_fragment": int(_has_fragment(ip_layer)),
        "urgent": int(urgent),
    }

    if _is_agent_server_traffic(fields, endpoint):
        return None
    return fields


def send_prediction(fields: Dict[str, Any], server_url: str, timeout_seconds: float) -> None:
    try:
        response = requests.post(server_url, json=fields, timeout=timeout_seconds)
    except requests.RequestException as exc:
        LOGGER.warning("send_failed error=%s dst=%s", exc, server_url)
        return

    if response.status_code >= 400:
        LOGGER.warning("server_rejected status=%s body=%s", response.status_code, response.text[:500])
        return

    try:
        result = response.json()
    except ValueError:
        LOGGER.warning("server_returned_non_json status=%s body=%s", response.status_code, response.text[:500])
        return

    if result.get("binary_prediction") == "attack":
        LOGGER.warning(
            "attack_prediction src=%s:%s dst=%s:%s protocol=%s attack_type=%s binary_confidence=%s",
            fields.get("src_ip"),
            fields.get("src_port"),
            fields.get("dst_ip"),
            fields.get("dst_port"),
            fields.get("protocol"),
            result.get("attack_type"),
            result.get("binary_confidence"),
        )
    else:
        LOGGER.debug("normal_prediction binary_confidence=%s", result.get("binary_confidence"))


def run_agent(
    config: RuntimeConfig,
    packet_callback: Optional[PacketCallback] = None,
    stop_event: Optional[Event] = None,
) -> None:
    endpoint = resolve_server_endpoint(config.agent_server_url)
    LOGGER.info(
        "agent_start interface=%s limit=%s server_url=%s",
        config.scapy_interface or "default",
        config.packet_limit or "infinite",
        config.agent_server_url,
    )

    seen_packets = 0

    def handle_packet(packet: Any) -> None:
        nonlocal seen_packets
        seen_packets += 1
        fields = packet_to_raw_fields(packet, endpoint)
        if fields is None:
            return
        LOGGER.debug(
            "packet_fields src=%s dst=%s protocol=%s len=%s",
            fields["src_ip"],
            fields["dst_ip"],
            fields["protocol"],
            fields["packet_len"],
        )
        if packet_callback is not None:
            packet_callback(fields)
            return
        send_prediction(fields, config.agent_server_url, config.request_timeout_seconds)

    def should_stop(_: Any) -> bool:
        return bool(stop_event and stop_event.is_set())

    try:
        if stop_event is None:
            sniff(
                iface=config.scapy_interface,
                prn=handle_packet,
                store=False,
                count=config.packet_limit or 0,
            )
            return

        while not stop_event.is_set():
            if config.packet_limit is None:
                count = 0
            else:
                count = max(config.packet_limit - seen_packets, 0)
                if count == 0:
                    break

            sniff(
                iface=config.scapy_interface,
                prn=handle_packet,
                store=False,
                count=count,
                timeout=1,
                stop_filter=should_stop,
            )
    except Exception:
        LOGGER.exception("agent_sniff_failed")


def resolve_server_endpoint(server_url: str) -> ServerEndpoint:
    parsed = urlparse(server_url)
    host = parsed.hostname
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    ips: Set[str] = set()
    if host and host not in {"0.0.0.0", "::"}:
        try:
            for addr_info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
                ip = addr_info[4][0]
                if _is_ipv4(ip):
                    ips.add(ip)
        except socket.gaierror:
            LOGGER.debug("could_not_resolve_server_host host=%s", host)

    return ServerEndpoint(host=host, port=port, ips=ips)


def _is_agent_server_traffic(fields: Dict[str, Any], endpoint: ServerEndpoint) -> bool:
    if fields.get("protocol") != "tcp" or endpoint.port is None:
        return False
    if endpoint.port not in {fields.get("src_port"), fields.get("dst_port")}:
        return False
    if not endpoint.ips:
        return False
    return fields.get("src_ip") in endpoint.ips or fields.get("dst_ip") in endpoint.ips


def _is_ignored_ip(address: str) -> bool:
    try:
        ip_value = ipaddress.ip_address(address)
    except ValueError:
        return True
    return (
        ip_value.is_loopback
        or ip_value.is_multicast
        or ip_value.is_unspecified
        or str(ip_value) == "255.255.255.255"
    )


def _is_ethernet_broadcast(packet: Any) -> bool:
    if Ether not in packet:
        return False
    return str(packet[Ether].dst).lower() == "ff:ff:ff:ff:ff:ff"


def _has_fragment(ip_layer: Any) -> bool:
    return bool(int(getattr(ip_layer, "frag", 0)) != 0 or "MF" in str(getattr(ip_layer, "flags", "")))


def _is_ipv4(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).version == 4
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IDS live traffic Scapy agent")
    parser.add_argument("--config", default="config.json", help="Path to runtime config.json")
    parser.add_argument("--server-url", default=None, help="Override Flask /predict URL")
    parser.add_argument("--interface", default=None, help="Override Scapy interface")
    parser.add_argument("--limit", type=int, default=None, help="Packet limit for test mode")
    return parser.parse_args()


def apply_cli_overrides(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    packet_limit = config.packet_limit
    if args.limit is not None:
        packet_limit = args.limit if args.limit > 0 else None

    return RuntimeConfig(
        flask_host=config.flask_host,
        flask_port=config.flask_port,
        binary_threshold=config.binary_threshold,
        scapy_interface=args.interface if args.interface is not None else config.scapy_interface,
        packet_limit=packet_limit,
        log_level=config.log_level,
        artifacts_path=config.artifacts_path,
        agent_server_url=args.server_url if args.server_url is not None else config.agent_server_url,
        agent_enabled=config.agent_enabled,
        request_timeout_seconds=config.request_timeout_seconds,
        logs_path=config.logs_path,
        traffic_store_limit=config.traffic_store_limit,
        config_path=config.config_path,
    )


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    global LOGGER
    LOGGER = configure_logging(config, "ids.agent", "agent.log")
    run_agent(config)


if __name__ == "__main__":
    main()
