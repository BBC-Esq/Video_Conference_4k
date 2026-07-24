import asyncio
import os
import socket
from typing import Optional, Tuple, Dict, Any
from os.path import expanduser

from ..utils.common import (
    get_logger,
    check_WriteAccess,
    import_dependency_safe,
)
from ..utils.auth import (
    generate_auth_certificates,
)

zmq = import_dependency_safe("zmq", pkg_name="pyzmq", error="silent", min_version="4.0")
if zmq is not None:
    from zmq import auth
    from zmq.auth.thread import ThreadAuthenticator

logger = get_logger("NetBase")

VALID_PATTERNS_SYNC = {
    0: (zmq.PAIR, zmq.PAIR) if zmq else None,
    1: (zmq.REQ, zmq.REP) if zmq else None,
    2: (zmq.PUB, zmq.SUB) if zmq else None,
}

VALID_PATTERNS_ASYNC = {
    0: (zmq.PAIR, zmq.PAIR) if zmq else None,
    1: (zmq.REQ, zmq.REP) if zmq else None,
    2: (zmq.PUB, zmq.SUB) if zmq else None,
    3: (zmq.PUSH, zmq.PULL) if zmq else None,
}

VALID_SECURITY_MECHANISMS = {
    0: "Grasslands",
    1: "StoneHouse",
    2: "IronHouse",
}

VALID_PROTOCOLS = ["tcp", "ipc"]

DSCP_VIDEO = 34
DSCP_AUDIO = 46


def apply_socket_qos(socket: Any, dscp: int) -> None:
    if not dscp or zmq is None:
        return
    try:
        socket.setsockopt(zmq.TOS, int(dscp) << 2)
    except Exception:
        pass


def validate_pattern(pattern: int, async_mode: bool = False) -> Tuple[int, Tuple]:
    valid_patterns = VALID_PATTERNS_ASYNC if async_mode else VALID_PATTERNS_SYNC

    if isinstance(pattern, int) and pattern in valid_patterns:
        return pattern, valid_patterns[pattern]

    logger.warning(
        "Invalid pattern {}. Defaulting to `zmq.PAIR`!".format(pattern)
    )
    return 0, valid_patterns[0]


def validate_protocol(protocol: str) -> str:
    if protocol and protocol in VALID_PROTOCOLS:
        return protocol

    logger.warning(
        "Protocol is not supported or not provided. Defaulting to `tcp` protocol!"
    )
    return "tcp"


def validate_address(address: str, receive_mode: bool) -> str:
    if address is not None:
        return address
    return "*" if receive_mode else "localhost"


def validate_port(port: str) -> str:
    if port is not None:
        return port
    return "5555"


def build_connection_string(protocol: str, address: str, port: str) -> str:
    return "{}://{}:{}".format(protocol, address, port)


def setup_authenticator(
    context: Any,
    address: str,
    secure_mode: int,
    custom_cert_location: str = "",
    overwrite_cert: bool = False,
    logging: bool = False,
) -> Tuple[Optional[Any], str, str]:
    auth_publickeys_dir = ""
    auth_secretkeys_dir = ""
    z_auth = None

    if secure_mode <= 0:
        return None, "", ""

    try:
        if custom_cert_location:
            assert check_WriteAccess(
                custom_cert_location,
                is_windows=os.name == "nt",
                logging=logging,
            ), "[NetBase:ERROR] :: Permission Denied! Cannot write ZMQ authentication certificates to '{}' directory!".format(
                custom_cert_location
            )
            cert_path = custom_cert_location
        else:
            cert_path = os.path.join(expanduser("~"), ".videoconference4k")

        (
            auth_cert_dir,
            auth_secretkeys_dir,
            auth_publickeys_dir,
        ) = generate_auth_certificates(
            cert_path, overwrite=overwrite_cert, logging=logging
        )

        logging and logger.debug(
            "`{}` is the default location for storing ZMQ authentication certificates/keys.".format(
                auth_cert_dir
            )
        )

        if os.name == "nt":
            # pyzmq's ThreadAuthenticator runs an asyncio-based ZAP handler in
            # its own thread; on Windows it needs a selector event loop (the
            # default Proactor loop lacks add_reader, killing the handler and
            # hanging every CURVE handshake). Same policy AsyncTransport sets.
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        z_auth = ThreadAuthenticator(context)
        z_auth.start()
        if address and str(address) != "*":
            try:
                z_auth.allow(socket.gethostbyname(str(address)))
            except Exception:
                z_auth.allow(str(address))

        if secure_mode == 2:
            z_auth.configure_curve(
                domain="*", location=auth_publickeys_dir
            )
        else:
            z_auth.configure_curve(
                domain="*", location=auth.CURVE_ALLOW_ANY
            )

    except zmq.ZMQError as e:
        if "Address in use" in str(e):
            # An authenticator from another instance is already running on
            # this context; certificates were generated fine, so CURVE
            # security still applies to this connection.
            logger.info("ZMQ Authenticator already running.")
        else:
            logger.exception(str(e))
            raise RuntimeError(
                "[NetBase:ERROR] :: Failed to activate ZMQ Security Mechanism!"
            ) from e

    return z_auth, auth_secretkeys_dir, auth_publickeys_dir


def apply_socket_security(
    socket: Any,
    secure_mode: int,
    is_server: bool,
    auth_secretkeys_dir: str,
    auth_publickeys_dir: str,
) -> None:
    if secure_mode <= 0:
        return

    if is_server:
        server_secret_file = os.path.join(auth_secretkeys_dir, "server.key_secret")
        server_public, server_secret = auth.load_certificate(server_secret_file)
        socket.curve_secretkey = server_secret
        socket.curve_publickey = server_public
        socket.curve_server = True
    else:
        client_secret_file = os.path.join(auth_secretkeys_dir, "client.key_secret")
        client_public, client_secret = auth.load_certificate(client_secret_file)
        socket.curve_secretkey = client_secret
        socket.curve_publickey = client_public
        server_public_file = os.path.join(auth_publickeys_dir, "server.key")
        server_public, _ = auth.load_certificate(server_public_file)
        socket.curve_serverkey = server_public


def create_frame_message(
    terminate_flag: bool = False,
    compression: Any = False,
    message: Any = None,
    pattern: int = 0,
    dtype: str = "",
    shape: Any = "",
    port: str = None,
    multiserver_mode: bool = False,
    ack: bool = True,
) -> Dict[str, Any]:
    msg_dict = {}

    if multiserver_mode and port:
        msg_dict["port"] = port

    msg_dict.update({
        "terminate_flag": terminate_flag,
        "compression": compression,
        "message": message,
        "pattern": str(pattern),
        "dtype": dtype,
        "shape": shape,
        "ack": ack,
    })

    return msg_dict


def create_return_message(
    return_type: str,
    return_data: Any = None,
    compression: Any = False,
    array_dtype: str = "",
    array_shape: Any = "",
    port: str = None,
    multiclient_mode: bool = False,
) -> Dict[str, Any]:
    msg_dict = {}

    if multiclient_mode and port:
        msg_dict["port"] = port

    msg_dict.update({
        "return_type": return_type,
        "compression": compression,
        "array_dtype": array_dtype,
        "array_shape": array_shape,
        "data": return_data,
    })

    return msg_dict


def create_async_frame_message(
    terminate: bool = False,
    bi_mode: bool = False,
    data: Any = "",
    gpu_accelerated: bool = False,
    software_accelerated: bool = False,
    gpu_codec: str = "",
    width: int = 0,
    height: int = 0,
) -> Dict[str, Any]:
    return {
        "terminate": terminate,
        "bi_mode": bi_mode,
        "data": data,
        "gpu_accelerated": gpu_accelerated,
        "software_accelerated": software_accelerated,
        "gpu_codec": gpu_codec,
        "width": width,
        "height": height,
    }


def create_async_return_message(
    return_type: str,
    return_data: Any = None,
    gpu_accelerated: bool = False,
    software_accelerated: bool = False,
    gpu_codec: str = "",
    width: int = 0,
    height: int = 0,
) -> Dict[str, Any]:
    return {
        "return_type": return_type,
        "return_data": return_data,
        "gpu_accelerated": gpu_accelerated,
        "software_accelerated": software_accelerated,
        "gpu_codec": gpu_codec,
        "width": width,
        "height": height,
    }