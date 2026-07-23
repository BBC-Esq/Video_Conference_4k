import re
import socket
import threading
import time
import urllib.request
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

from ..utils.common import get_logger

logger = get_logger("UPnP")

SSDP_ADDR = "239.255.255.250"
SSDP_PORT = 1900
IGD_DEVICE = "urn:schemas-upnp-org:device:InternetGatewayDevice:1"
WAN_SERVICES = (
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)


def get_local_ip() -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return None
    finally:
        sock.close()


def _parse_ssdp_location(response: str) -> Optional[str]:
    for line in response.splitlines():
        if line.lower().startswith("location:"):
            return line.split(":", 1)[1].strip()
    return None


def _http_get(url: str, timeout: float = 3.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _find_control_url(device_xml: str, base_url: str) -> Optional[Dict[str, str]]:
    try:
        root = ElementTree.fromstring(device_xml)
    except ElementTree.ParseError:
        return None

    def strip_ns(tag: str) -> str:
        return tag.split("}", 1)[-1]

    for service in root.iter():
        if strip_ns(service.tag) != "service":
            continue
        stype = control = None
        for child in service:
            name = strip_ns(child.tag)
            if name == "serviceType":
                stype = (child.text or "").strip()
            elif name == "controlURL":
                control = (child.text or "").strip()
        if stype in WAN_SERVICES and control:
            return {"service_type": stype, "control_url": urljoin(base_url, control)}
    return None


def discover_igd(timeout: float = 3.0) -> Optional[Dict[str, str]]:
    request = "\r\n".join([
        "M-SEARCH * HTTP/1.1",
        "HOST: {}:{}".format(SSDP_ADDR, SSDP_PORT),
        'MAN: "ssdp:discover"',
        "MX: 2",
        "ST: {}".format(IGD_DEVICE),
        "", "",
    ]).encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    try:
        sock.sendto(request, (SSDP_ADDR, SSDP_PORT))
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, _ = sock.recvfrom(65507)
            except socket.timeout:
                break
            location = _parse_ssdp_location(data.decode("utf-8", errors="replace"))
            if not location:
                continue
            try:
                device_xml = _http_get(location, timeout=timeout)
            except Exception:
                continue
            igd = _find_control_url(device_xml, location)
            if igd:
                return igd
    except Exception as e:
        logger.debug("UPnP discovery error: {}".format(e))
    finally:
        sock.close()
    return None


def _build_soap(service_type: str, action: str, args: List) -> bytes:
    body = "".join("<{0}>{1}</{0}>".format(k, v) for k, v in args)
    envelope = (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        '<s:Body><u:{action} xmlns:u="{stype}">{body}</u:{action}></s:Body>'
        "</s:Envelope>"
    ).format(action=action, stype=service_type, body=body)
    return envelope.encode("utf-8")


def _soap_call(igd: Dict[str, str], action: str, args: List, timeout: float = 3.0) -> Optional[str]:
    service_type = igd["service_type"]
    payload = _build_soap(service_type, action, args)
    headers = {
        "Content-Type": 'text/xml; charset="utf-8"',
        "SOAPAction": '"{}#{}"'.format(service_type, action),
    }
    req = urllib.request.Request(igd["control_url"], data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug("SOAP {} failed: {}".format(action, e))
        return None


def get_external_ip(igd: Dict[str, str]) -> Optional[str]:
    response = _soap_call(igd, "GetExternalIPAddress", [])
    if not response:
        return None
    match = re.search(r"<NewExternalIPAddress>\s*([^<\s]+)\s*</NewExternalIPAddress>", response)
    return match.group(1) if match else None


def add_port_mapping(
    igd: Dict[str, str],
    external_port: int,
    internal_port: int,
    internal_ip: str,
    protocol: str = "UDP",
    description: str = "VideoConference4k",
    lease: int = 3600,
) -> bool:
    args = [
        ("NewRemoteHost", ""),
        ("NewExternalPort", external_port),
        ("NewProtocol", protocol.upper()),
        ("NewInternalPort", internal_port),
        ("NewInternalClient", internal_ip),
        ("NewEnabled", 1),
        ("NewPortMappingDescription", description),
        ("NewLeaseDuration", lease),
    ]
    return _soap_call(igd, "AddPortMapping", args) is not None


def delete_port_mapping(igd: Dict[str, str], external_port: int, protocol: str = "UDP") -> bool:
    args = [
        ("NewRemoteHost", ""),
        ("NewExternalPort", external_port),
        ("NewProtocol", protocol.upper()),
    ]
    return _soap_call(igd, "DeletePortMapping", args) is not None


class UPnPPortMapper:

    def __init__(self, description: str = "VideoConference4k", lease: int = 3600, logging: bool = False):
        self.__logging = logging if isinstance(logging, bool) else False
        self.__description = description
        self.__lease = lease
        self.__igd: Optional[Dict[str, str]] = None
        self.__mappings: List[Dict] = []
        self.__local_ip = get_local_ip()
        self.__terminate = threading.Event()
        self.__thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        return self.__igd is not None

    @property
    def local_ip(self) -> Optional[str]:
        return self.__local_ip

    def discover(self, timeout: float = 3.0) -> bool:
        self.__igd = discover_igd(timeout=timeout)
        self.__logging and logger.debug(
            "UPnP IGD {}".format("found" if self.__igd else "not found")
        )
        return self.__igd is not None

    def external_ip(self) -> Optional[str]:
        return get_external_ip(self.__igd) if self.__igd else None

    def map_port(self, port: int, protocol: str = "UDP", internal_port: int = None) -> bool:
        if self.__igd is None or self.__local_ip is None:
            return False
        internal_port = internal_port if internal_port is not None else port
        ok = add_port_mapping(
            self.__igd, port, internal_port, self.__local_ip,
            protocol=protocol, description=self.__description, lease=self.__lease,
        )
        if ok:
            self.__mappings.append({"port": port, "protocol": protocol, "internal_port": internal_port})
            self.__logging and logger.debug("Mapped {} {} -> {}:{}".format(protocol, port, self.__local_ip, internal_port))
            self.__ensure_refresh()
        return ok

    def __ensure_refresh(self) -> None:
        if self.__lease <= 0 or self.__thread is not None:
            return
        self.__thread = threading.Thread(target=self.__refresh_loop, name="UPnPRefresh", daemon=True)
        self.__thread.start()

    def __refresh_loop(self) -> None:
        interval = max(30, self.__lease // 2)
        while not self.__terminate.wait(interval):
            if self.__igd is None:
                continue
            for m in list(self.__mappings):
                add_port_mapping(
                    self.__igd, m["port"], m["internal_port"], self.__local_ip,
                    protocol=m["protocol"], description=self.__description, lease=self.__lease,
                )

    def close(self) -> None:
        self.__terminate.set()
        if self.__thread is not None:
            self.__thread.join(timeout=2)
            self.__thread = None
        if self.__igd is not None:
            for m in list(self.__mappings):
                delete_port_mapping(self.__igd, m["port"], m["protocol"])
        self.__mappings = []
        self.__logging and logger.debug("UPnP mappings released.")
