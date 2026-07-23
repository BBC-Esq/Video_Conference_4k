import os
import time
import asyncio
import platform
import string
import secrets
import numpy as np
from threading import Thread, Lock, Event
from collections import deque
from numpy.typing import NDArray
from typing import Optional, Any, Tuple

from ..utils.common import (
    get_logger,
    check_open_port,
    import_dependency_safe,
    log_version,
)
from .base import (
    validate_pattern,
    validate_protocol,
    validate_address,
    validate_port,
    build_connection_string,
    setup_authenticator,
    apply_socket_security,
    create_frame_message,
    create_return_message,
    VALID_SECURITY_MECHANISMS,
)
from .compression import (
    CompressionHandler,
    CompressionType,
    decode_sync_frame,
)

zmq = import_dependency_safe("zmq", pkg_name="pyzmq", error="silent", min_version="4.0")
if zmq is not None:
    from zmq import ssh
    from zmq.auth.thread import ThreadAuthenticator
    from zmq.error import ZMQError
simplejpeg = import_dependency_safe("simplejpeg", error="silent", min_version="1.6.1")
paramiko = import_dependency_safe("paramiko", error="silent")

logger = get_logger("SyncTransport")


class SyncTransport:

    def __init__(
        self,
        address: str = None,
        port: str = None,
        protocol: str = None,
        pattern: int = 0,
        receive_mode: bool = False,
        logging: bool = False,
        gpu_accelerated: bool = False,
        gpu_id: int = 0,
        gpu_resolution: Tuple[int, int] = None,
        gpu_bitrate: int = 8000000,
        gpu_codec: str = "h264",
        **options: dict
    ):
        self.__logging = logging if isinstance(logging, bool) else False

        log_version(logging=self.__logging)

        import_dependency_safe(
            "zmq" if zmq is None else "", min_version="4.0", pkg_name="pyzmq"
        )
        import_dependency_safe(
            "simplejpeg" if simplejpeg is None else "", error="log", min_version="1.6.1"
        )

        self.__gpu_codec = gpu_codec
        self.__compression_handler = CompressionHandler(
            gpu_accelerated=gpu_accelerated,
            gpu_id=gpu_id,
            gpu_bitrate=gpu_bitrate,
            gpu_codec=gpu_codec,
            logging=logging,
        )

        self.__pattern, msg_pattern = validate_pattern(pattern, async_mode=False)
        protocol = validate_protocol(protocol)

        self.__msg_flag = 0
        self.__msg_copy = False
        self.__msg_track = False

        self.__z_auth = None
        self.__auth_publickeys_dir = ""
        self.__auth_secretkeys_dir = ""

        self.__ssh_tunnel_mode = None
        self.__ssh_tunnel_pwd = None
        self.__ssh_tunnel_keyfile = None
        self.__paramiko_present = False if paramiko is None else True

        self.__multiserver_mode = False
        self.__multiclient_mode = False
        self.__bi_mode = False

        self.__secure_mode = 0
        overwrite_cert = False
        custom_cert_location = ""

        self.__jpeg_compression = (
            True if simplejpeg is not None and not self.__compression_handler.is_nvidia and not self.__compression_handler.is_software else False
        )
        self.__jpeg_compression_quality = 90
        self.__jpeg_compression_fastdct = True
        self.__jpeg_compression_fastupsample = False
        self.__jpeg_compression_colorspace = "BGR"

        self.__return_data = None
        self.__return_data_lock = Lock()

        self.__id = "".join(
            secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)
        )

        self.__terminate = Event()

        if self.__pattern < 2:
            self.__poll = zmq.Poller()
            self.__max_retries = 3
            self.__request_timeout = 4000
        else:
            self.__subscriber_timeout = None

        options = {str(k).strip(): v for k, v in options.items()}

        for key, value in options.items():
            if key == "multiserver_mode" and isinstance(value, bool):
                if self.__pattern > 0:
                    self.__multiserver_mode = value
                else:
                    self.__multiserver_mode = False
                    logger.critical("Multi-Server Mode is disabled!")
                    raise ValueError(
                        "[SyncTransport:ERROR] :: `{}` pattern is not valid when Multi-Server Mode is enabled. Kindly refer Docs for more Information.".format(
                            self.__pattern
                        )
                    )

            elif key == "multiclient_mode" and isinstance(value, bool):
                if self.__pattern > 0:
                    self.__multiclient_mode = value
                else:
                    self.__multiclient_mode = False
                    logger.critical("Multi-Client Mode is disabled!")
                    raise ValueError(
                        "[SyncTransport:ERROR] :: `{}` pattern is not valid when Multi-Client Mode is enabled. Kindly refer Docs for more Information.".format(
                            self.__pattern
                        )
                    )

            elif key == "bidirectional_mode" and isinstance(value, bool):
                if self.__pattern < 2:
                    self.__bi_mode = value
                else:
                    self.__bi_mode = False
                    logger.warning("Bidirectional data transmission is disabled!")
                    raise ValueError(
                        "[SyncTransport:ERROR] :: `{}` pattern is not valid when Bidirectional Mode is enabled. Kindly refer Docs for more Information!".format(
                            self.__pattern
                        )
                    )

            elif key == "secure_mode" and isinstance(value, int) and value in VALID_SECURITY_MECHANISMS:
                self.__secure_mode = value

            elif key == "custom_cert_location" and isinstance(value, str):
                custom_cert_location = os.path.abspath(value)
                assert os.path.isdir(custom_cert_location), \
                    "[SyncTransport:ERROR] :: `custom_cert_location` value must be the path to a valid directory!"

            elif key == "overwrite_cert" and isinstance(value, bool):
                overwrite_cert = value

            elif key == "ssh_tunnel_mode" and isinstance(value, str):
                self.__ssh_tunnel_mode = value.strip()

            elif key == "ssh_tunnel_pwd" and isinstance(value, str):
                self.__ssh_tunnel_pwd = value

            elif key == "ssh_tunnel_keyfile" and isinstance(value, str):
                self.__ssh_tunnel_keyfile = value if os.path.isfile(value) else None
                if self.__ssh_tunnel_keyfile is None:
                    logger.warning(
                        "Discarded invalid or non-existential SSH Tunnel Key-file at {}!".format(value)
                    )

            elif key == "jpeg_compression" and simplejpeg is not None and isinstance(value, (bool, str)) and not self.__compression_handler.is_nvidia and not self.__compression_handler.is_software:
                if isinstance(value, str) and value.strip().upper() in [
                    "RGB", "BGR", "RGBX", "BGRX", "XBGR", "XRGB",
                    "GRAY", "RGBA", "BGRA", "ABGR", "ARGB", "CMYK",
                ]:
                    self.__jpeg_compression_colorspace = value.strip().upper()
                    self.__jpeg_compression = True
                else:
                    self.__jpeg_compression = value

            elif key == "jpeg_compression_quality" and isinstance(value, (int, float)):
                if 10 <= value <= 100:
                    self.__jpeg_compression_quality = int(value)
                else:
                    logger.warning("Skipped invalid `jpeg_compression_quality` value!")

            elif key == "jpeg_compression_fastdct" and isinstance(value, bool):
                self.__jpeg_compression_fastdct = value

            elif key == "jpeg_compression_fastupsample" and isinstance(value, bool):
                self.__jpeg_compression_fastupsample = value

            elif key == "max_retries" and isinstance(value, int) and self.__pattern < 2:
                if value >= 0:
                    self.__max_retries = value
                else:
                    logger.warning("Invalid `max_retries` value skipped!")

            elif key == "request_timeout" and isinstance(value, int) and self.__pattern < 2:
                if value >= 4:
                    self.__request_timeout = value * 1000
                else:
                    logger.warning("Invalid `request_timeout` value skipped!")

            elif key == "subscriber_timeout" and isinstance(value, int) and self.__pattern == 2:
                if value > 0:
                    self.__subscriber_timeout = value * 1000
                else:
                    logger.warning("Invalid `subscriber_timeout` value skipped!")

            elif key == "flag" and isinstance(value, int):
                self.__msg_flag = value
                self.__msg_flag and logger.warning(
                    "The flag optional value is set to `1` (NOBLOCK) for this run. This might cause SyncTransport to not terminate gracefully."
                )

            elif key == "copy" and isinstance(value, bool):
                self.__msg_copy = value

            elif key == "track" and isinstance(value, bool):
                self.__msg_track = value
                self.__msg_copy and self.__msg_track and logger.info(
                    "The `track` optional value will be ignored for this run because `copy=True` is also defined."
                )

        # Forward the parsed JPEG options to the compression handler
        # (constructed earlier with defaults). `jpeg_compression=False`
        # disables JPEG entirely, so frames travel raw (lossless).
        self.__compression_handler.configure_jpeg(
            enabled=bool(self.__jpeg_compression),
            quality=self.__jpeg_compression_quality,
            colorspace=self.__jpeg_compression_colorspace,
            fastdct=self.__jpeg_compression_fastdct,
            fastupsample=self.__jpeg_compression_fastupsample,
        )

        if self.__ssh_tunnel_mode is not None:
            if receive_mode:
                logger.error("SSH Tunneling cannot be enabled for Client-end!")
            else:
                ssh_address = self.__ssh_tunnel_mode
                ssh_address, ssh_port = (
                    ssh_address.split(":") if ":" in ssh_address else [ssh_address, "22"]
                )
                if "47" in ssh_port:
                    self.__ssh_tunnel_mode = self.__ssh_tunnel_mode.replace(":47", "")
                else:
                    ssh_user, ssh_ip = (
                        ssh_address.split("@") if "@" in ssh_address else ["", ssh_address]
                    )
                    assert check_open_port(ssh_ip, port=int(ssh_port)), \
                        "[SyncTransport:ERROR] :: Host `{}` is not available for SSH Tunneling at port-{}!".format(
                            ssh_address, ssh_port
                        )

        if self.__multiclient_mode and self.__multiserver_mode:
            raise ValueError(
                "[SyncTransport:ERROR] :: Multi-Client and Multi-Server Mode cannot be enabled simultaneously!"
            )
        elif self.__multiserver_mode or self.__multiclient_mode:
            if self.__bi_mode:
                self.__logging and logger.debug(
                    "Bidirectional Data Transmission is also enabled for this connection!"
                )
            if self.__ssh_tunnel_mode:
                raise ValueError(
                    "[SyncTransport:ERROR] :: SSH Tunneling and {} Mode cannot be enabled simultaneously. Kindly refer docs!".format(
                        "Multi-Server" if self.__multiserver_mode else "Multi-Client"
                    )
                )
        elif self.__bi_mode:
            self.__logging and logger.debug(
                "Bidirectional Data Transmission is enabled for this connection!"
            )
        elif self.__ssh_tunnel_mode:
            self.__logging and logger.debug(
                "SSH Tunneling is enabled for host:`{}` with `{}` back-end.".format(
                    self.__ssh_tunnel_mode,
                    "paramiko" if self.__paramiko_present else "pexpect",
                )
            )

        platform.system() == "Windows" and asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )

        self.__msg_context = zmq.Context.instance()
        self.__receive_mode = receive_mode

        address = validate_address(address, receive_mode)

        if self.__secure_mode > 0:
            if receive_mode:
                overwrite_cert = False
            else:
                overwrite_cert and self.__logging and logger.info(
                    "Overwriting ZMQ Authentication certificates over previous ones!"
                )

            self.__z_auth, self.__auth_secretkeys_dir, self.__auth_publickeys_dir = setup_authenticator(
                self.__msg_context,
                address,
                self.__secure_mode,
                custom_cert_location,
                overwrite_cert,
                self.__logging,
            )

            if self.__z_auth is None:
                # Security was explicitly requested; never continue unencrypted.
                raise RuntimeError(
                    "[SyncTransport:ERROR] :: Failed to enable ZMQ Security Mechanism "
                    "(secure_mode={})! Refusing to continue with an unencrypted connection.".format(
                        self.__secure_mode
                    )
                )

        if self.__receive_mode:
            if self.__multiserver_mode:
                if port is None or not isinstance(port, (tuple, list)):
                    raise ValueError(
                        "[SyncTransport:ERROR] :: Incorrect port value! Kindly provide a list/tuple of Server ports while Multi-Server mode is enabled."
                    )
                else:
                    logger.debug("Enabling Multi-Server Mode at PORTS: {}!".format(port))
                self.__port_buffer = []
            elif self.__multiclient_mode:
                if port is None:
                    raise ValueError(
                        "[SyncTransport:ERROR] :: Kindly provide a unique & valid port value at Client-end."
                    )
                else:
                    logger.debug("Enabling Multi-Client Mode at PORT: {} on this device!".format(port))
                self.__port = port
            else:
                pass

            port = validate_port(port)

            try:
                self.__msg_socket = self.__msg_context.socket(msg_pattern[1])
                self.__pattern == 2 and self.__msg_socket.set_hwm(1)

                apply_socket_security(
                    self.__msg_socket,
                    self.__secure_mode,
                    True,
                    self.__auth_secretkeys_dir,
                    self.__auth_publickeys_dir,
                )

                if self.__pattern == 2:
                    self.__msg_socket.setsockopt_string(zmq.SUBSCRIBE, "")
                    self.__subscriber_timeout and self.__msg_socket.setsockopt(
                        zmq.RCVTIMEO, self.__subscriber_timeout
                    )
                    self.__subscriber_timeout and self.__msg_socket.setsockopt(zmq.LINGER, 0)

                if self.__multiserver_mode:
                    for pt in port:
                        self.__msg_socket.bind(build_connection_string(protocol, address, pt))
                else:
                    self.__msg_socket.bind(build_connection_string(protocol, address, port))

                if self.__pattern < 2:
                    if self.__multiserver_mode:
                        self.__connection_address = [
                            build_connection_string(protocol, address, pt) for pt in port
                        ]
                    else:
                        self.__connection_address = build_connection_string(protocol, address, port)
                    self.__msg_pattern = msg_pattern[1]
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    self.__logging and logger.debug(
                        "Reliable transmission is enabled for this pattern with max-retries: {} and timeout: {} secs.".format(
                            self.__max_retries, self.__request_timeout / 1000
                        )
                    )
                else:
                    self.__logging and self.__subscriber_timeout and logger.debug(
                        "Timeout: {} secs is enabled for this system.".format(self.__subscriber_timeout / 1000)
                    )

            except Exception as e:
                logger.exception(str(e))
                self.__secure_mode and logger.critical(
                    "Failed to activate Secure Mode: `{}` for this connection!".format(
                        VALID_SECURITY_MECHANISMS[self.__secure_mode]
                    )
                )
                if self.__multiserver_mode or self.__multiclient_mode:
                    raise RuntimeError(
                        "[SyncTransport:ERROR] :: Receive Mode failed to activate {} Mode at address: {} with pattern: {}!".format(
                            "Multi-Server" if self.__multiserver_mode else "Multi-Client",
                            build_connection_string(protocol, address, port),
                            self.__pattern,
                        )
                    )
                else:
                    self.__bi_mode and logger.critical(
                        "Failed to activate Bidirectional Mode for this connection!"
                    )
                    raise RuntimeError(
                        "[SyncTransport:ERROR] :: Receive Mode failed to bind address: {} and pattern: {}!".format(
                            build_connection_string(protocol, address, port), self.__pattern
                        )
                    )

            self.__logging and logger.debug(
                "Threaded Queue Mode is enabled by default for this connection."
            )

            self.__queue = deque(maxlen=96)

            self.__thread = Thread(target=self.__recv_handler, name="SyncTransport", args=())
            self.__thread.daemon = True
            self.__thread.start()

            if self.__logging:
                logger.debug(
                    "Successfully Binded to address: {} with pattern: {}.".format(
                        build_connection_string(protocol, address, port), self.__pattern
                    )
                )
                if self.__compression_handler.is_nvidia:
                    logger.debug("GPU-Accelerated encoding is activated for this connection using NVIDIA hardware.")
                elif self.__compression_handler.is_software:
                    logger.debug("Software-Accelerated encoding is activated for this connection using x264.")
                elif self.__jpeg_compression:
                    logger.debug(
                        "JPEG Frame-Compression is activated for this connection with Colorspace:`{}`, Quality:`{}`%, Fastdct:`{}`, and Fastupsample:`{}`.".format(
                            self.__jpeg_compression_colorspace,
                            self.__jpeg_compression_quality,
                            "enabled" if self.__jpeg_compression_fastdct else "disabled",
                            "enabled" if self.__jpeg_compression_fastupsample else "disabled",
                        )
                    )
                self.__secure_mode and logger.debug(
                    "Successfully enabled ZMQ Security Mechanism: `{}` for this connection.".format(
                        VALID_SECURITY_MECHANISMS[self.__secure_mode]
                    )
                )
                logger.debug("Multi-threaded Receive Mode is successfully enabled.")
                logger.debug("Unique System ID is {}.".format(self.__id))
                logger.debug("Receive Mode is now activated.")

        else:
            if self.__multiserver_mode:
                if port is None:
                    raise ValueError(
                        "[SyncTransport:ERROR] :: Kindly provide a unique & valid port value at Server-end."
                    )
                else:
                    logger.debug("Enabling Multi-Server Mode at PORT: {} on this device!".format(port))
                self.__port = port
            elif self.__multiclient_mode:
                if port is None or not isinstance(port, (tuple, list)):
                    raise ValueError(
                        "[SyncTransport:ERROR] :: Incorrect port value! Kindly provide a list/tuple of Client ports while Multi-Client mode is enabled."
                    )
                else:
                    logger.debug("Enabling Multi-Client Mode at PORTS: {}!".format(port))
                self.__port_buffer = []
            else:
                pass

            port = validate_port(port)

            try:
                self.__msg_socket = self.__msg_context.socket(msg_pattern[0])

                if self.__pattern == 1:
                    self.__msg_socket.REQ_RELAXED = True
                    self.__msg_socket.REQ_CORRELATE = True

                if self.__pattern == 2:
                    self.__msg_socket.set_hwm(1)

                apply_socket_security(
                    self.__msg_socket,
                    self.__secure_mode,
                    False,
                    self.__auth_secretkeys_dir,
                    self.__auth_publickeys_dir,
                )

                if self.__multiclient_mode:
                    for pt in port:
                        self.__msg_socket.connect(build_connection_string(protocol, address, pt))
                else:
                    if self.__ssh_tunnel_mode:
                        ssh.tunnel_connection(
                            self.__msg_socket,
                            build_connection_string(protocol, address, port),
                            self.__ssh_tunnel_mode,
                            keyfile=self.__ssh_tunnel_keyfile,
                            password=self.__ssh_tunnel_pwd,
                            paramiko=self.__paramiko_present,
                        )
                    else:
                        self.__msg_socket.connect(build_connection_string(protocol, address, port))

                if self.__pattern < 2:
                    if self.__multiclient_mode:
                        self.__connection_address = [
                            build_connection_string(protocol, address, pt) for pt in port
                        ]
                    else:
                        self.__connection_address = build_connection_string(protocol, address, port)
                    self.__msg_pattern = msg_pattern[0]
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    self.__logging and logger.debug(
                        "Reliable transmission is enabled for this pattern with max-retries: {} and timeout: {} secs.".format(
                            self.__max_retries, self.__request_timeout / 1000
                        )
                    )

            except Exception as e:
                logger.exception(str(e))
                self.__secure_mode and logger.critical(
                    "Failed to activate Secure Mode: `{}` for this connection!".format(
                        VALID_SECURITY_MECHANISMS[self.__secure_mode]
                    )
                )
                if self.__multiserver_mode or self.__multiclient_mode:
                    raise RuntimeError(
                        "[SyncTransport:ERROR] :: Send Mode failed to activate {} Mode at address: {} with pattern: {}!".format(
                            "Multi-Server" if self.__multiserver_mode else "Multi-Client",
                            build_connection_string(protocol, address, port),
                            self.__pattern,
                        )
                    )
                else:
                    self.__bi_mode and logger.critical(
                        "Failed to activate Bidirectional Mode for this connection!"
                    )
                    self.__ssh_tunnel_mode and logger.critical(
                        "Failed to initiate SSH Tunneling Mode for this server with `{}` back-end!".format(
                            "paramiko" if self.__paramiko_present else "pexpect"
                        )
                    )
                    raise RuntimeError(
                        "[SyncTransport:ERROR] :: Send Mode failed to connect address: {} and pattern: {}!".format(
                            build_connection_string(protocol, address, port), self.__pattern
                        )
                    )

            if self.__logging:
                logger.debug(
                    "Successfully connected to address: {} with pattern: {}.".format(
                        build_connection_string(protocol, address, port), self.__pattern
                    )
                )
                if self.__compression_handler.is_nvidia:
                    logger.debug("GPU-Accelerated encoding is activated for this connection using NVIDIA hardware.")
                elif self.__compression_handler.is_software:
                    logger.debug("Software-Accelerated encoding is activated for this connection using x264.")
                elif self.__jpeg_compression:
                    logger.debug(
                        "JPEG Frame-Compression is activated for this connection with Colorspace:`{}`, Quality:`{}`%, Fastdct:`{}`, and Fastupsample:`{}`.".format(
                            self.__jpeg_compression_colorspace,
                            self.__jpeg_compression_quality,
                            "enabled" if self.__jpeg_compression_fastdct else "disabled",
                            "enabled" if self.__jpeg_compression_fastupsample else "disabled",
                        )
                    )
                self.__secure_mode and logger.debug(
                    "Enabled ZMQ Security Mechanism: `{}` for this connection.".format(
                        VALID_SECURITY_MECHANISMS[self.__secure_mode]
                    )
                )
                logger.debug("Unique System ID is {}.".format(self.__id))
                logger.debug("Send Mode is successfully activated and ready to send data.")

    def __recv_handler(self):
        frame = None
        msg_json = None

        while not self.__terminate.is_set():
            if len(self.__queue) >= 96:
                time.sleep(0.000001)
                continue

            if self.__pattern < 2:
                try:
                    socks = dict(self.__poll.poll(self.__request_timeout * 3))
                except zmq.ZMQError:
                    self.__terminate.set()
                    self.__queue.append(None)
                    break
                if socks.get(self.__msg_socket) == zmq.POLLIN:
                    try:
                        msg_json = self.__msg_socket.recv_json(flags=self.__msg_flag | zmq.DONTWAIT)
                    except zmq.error.Again:
                        # Spurious poll wakeup (e.g. CURVE/ZAP handshake
                        # traffic) — nothing was actually readable yet.
                        continue
                else:
                    logger.critical("No response from Server(s), Reconnecting again...")
                    self.__msg_socket.close(linger=0)
                    self.__poll.unregister(self.__msg_socket)
                    self.__max_retries -= 1

                    if not self.__max_retries:
                        if self.__multiserver_mode:
                            logger.error("All Servers seems to be offline, Abandoning!")
                        else:
                            logger.error("Server seems to be offline, Abandoning!")
                        self.__terminate.set()
                        continue

                    try:
                        self.__msg_socket = self.__msg_context.socket(self.__msg_pattern)
                        if isinstance(self.__connection_address, list):
                            for _connection in self.__connection_address:
                                self.__msg_socket.bind(_connection)
                        else:
                            self.__msg_socket.bind(self.__connection_address)
                    except Exception as e:
                        logger.exception(str(e))
                        self.__terminate.set()
                        raise RuntimeError("API failed to restart the Client-end!")
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    continue
            else:
                try:
                    msg_json = self.__msg_socket.recv_json(flags=self.__msg_flag)
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        logger.critical("Connection Timeout. Exiting!")
                        self.__terminate.set()
                        self.__queue.append(None)
                        break

            if msg_json and msg_json["terminate_flag"]:
                if self.__multiserver_mode:
                    if msg_json["port"] in self.__port_buffer:
                        if self.__pattern == 1:
                            self.__msg_socket.send_string(
                                "Termination signal successfully received at client!"
                            )
                        self.__port_buffer.remove(msg_json["port"])
                        self.__logging and logger.warning(
                            "Termination signal received from Server at port: {}!".format(msg_json["port"])
                        )
                    if not self.__port_buffer:
                        logger.critical("Termination signal received from all Servers!!!")
                        self.__terminate.set()
                else:
                    if self.__pattern == 1:
                        self.__msg_socket.send_string(
                            "Termination signal successfully received at Client's end!"
                        )
                    self.__terminate.set()
                    self.__logging and logger.critical("Termination signal received from server!")
                continue

            try:
                msg_data = self.__msg_socket.recv(
                    flags=self.__msg_flag | zmq.DONTWAIT,
                    copy=self.__msg_copy,
                    track=self.__msg_track,
                )
            except zmq.ZMQError:
                logger.critical("Socket Session Expired. Exiting!")
                self.__terminate.set()
                self.__queue.append(None)
                break

            if self.__pattern < 2:
                if self.__bi_mode or self.__multiclient_mode:
                    with self.__return_data_lock:
                        local_return_data = self.__return_data

                    if local_return_data is not None and isinstance(local_return_data, np.ndarray):
                        encoded_data, metadata = self.__compression_handler.encode_frame(local_return_data)

                        return_dict = create_return_message(
                            return_type=type(local_return_data).__name__,
                            compression=metadata if self.__compression_handler.is_enabled else False,
                            array_dtype="" if self.__compression_handler.is_enabled else str(local_return_data.dtype),
                            array_shape="" if self.__compression_handler.is_enabled else local_return_data.shape,
                            port=self.__port if self.__multiclient_mode else None,
                            multiclient_mode=self.__multiclient_mode,
                        )

                        self.__msg_socket.send_json(return_dict, self.__msg_flag | zmq.SNDMORE)
                        self.__msg_socket.send(
                            encoded_data,
                            flags=self.__msg_flag,
                            copy=self.__msg_copy,
                            track=self.__msg_track,
                        )
                    else:
                        return_dict = create_return_message(
                            return_type=type(local_return_data).__name__,
                            return_data=local_return_data,
                            port=self.__port if self.__multiclient_mode else None,
                            multiclient_mode=self.__multiclient_mode,
                        )
                        self.__msg_socket.send_json(return_dict, self.__msg_flag)
                else:
                    self.__msg_socket.send_string("Data received on device: {} !".format(self.__id))
            else:
                with self.__return_data_lock:
                    local_return_data = self.__return_data
                if local_return_data:
                    logger.warning("`return_data` is disabled for this pattern!")

            compression_info = msg_json.get("compression")
            if compression_info:
                frame = decode_sync_frame(
                    bytes(msg_data),
                    compression_info,
                    self.__compression_handler,
                    self.__jpeg_compression_fastdct,
                    self.__jpeg_compression_fastupsample,
                )
                if frame is None:
                    self.__logging and logger.debug("Frame not yet decodable, skipping.")
                    continue
            else:
                frame_buffer = np.frombuffer(msg_data, dtype=msg_json["dtype"])
                frame = frame_buffer.reshape(msg_json["shape"])

            if self.__multiserver_mode:
                if msg_json["port"] not in self.__port_buffer:
                    self.__port_buffer.append(msg_json["port"])
                if msg_json["message"]:
                    self.__queue.append((msg_json["port"], msg_json["message"], frame))
                else:
                    self.__queue.append((msg_json["port"], frame))
            elif self.__bi_mode:
                if msg_json["message"]:
                    self.__queue.append((msg_json["message"], frame))
                else:
                    self.__queue.append((None, frame))
            else:
                self.__queue.append(frame)

    def recv(self, return_data=None) -> Optional[NDArray]:
        if not self.__receive_mode:
            self.__terminate.set()
            raise ValueError(
                "[SyncTransport:ERROR] :: `recv()` function cannot be used while receive_mode is disabled."
            )

        if (self.__bi_mode or self.__multiclient_mode) and return_data is not None:
            with self.__return_data_lock:
                self.__return_data = return_data

        while not self.__terminate.is_set():
            try:
                if len(self.__queue) > 0:
                    return self.__queue.popleft()
                else:
                    time.sleep(0.00001)
                    continue
            except KeyboardInterrupt:
                self.__terminate.set()
                break
        return None

    def send(self, frame: NDArray, message: Any = None) -> Optional[Any]:
        if self.__receive_mode:
            self.__terminate.set()
            raise ValueError(
                "[SyncTransport:ERROR] :: `send()` function cannot be used while receive_mode is enabled."
            )

        if message is not None and isinstance(message, np.ndarray):
            logger.warning("Skipped unsupported `message` of datatype: {}!".format(type(message).__name__))
            message = None

        exit_flag = frame is None or self.__terminate.is_set()

        if exit_flag:
            msg_dict = create_frame_message(
                terminate_flag=True,
                pattern=self.__pattern,
                port=self.__port if self.__multiserver_mode else None,
                multiserver_mode=self.__multiserver_mode,
            )
            self.__msg_socket.send_json(msg_dict, self.__msg_flag | zmq.SNDMORE)
            self.__msg_socket.send(
                b"",
                flags=self.__msg_flag,
                copy=self.__msg_copy,
                track=self.__msg_track,
            )

            if self.__pattern < 2:
                socks = dict(self.__poll.poll(self.__request_timeout))
                if socks.get(self.__msg_socket) == zmq.POLLIN:
                    if self.__bi_mode or self.__multiclient_mode:
                        self.__msg_socket.recv_json(flags=self.__msg_flag)
                    else:
                        self.__msg_socket.recv()
            return None

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame, dtype=frame.dtype)

        original_shape = frame.shape
        original_dtype = frame.dtype

        encoded_data, metadata = self.__compression_handler.encode_frame(frame)

        msg_dict = create_frame_message(
            terminate_flag=False,
            compression=metadata if self.__compression_handler.is_enabled else False,
            message=message,
            pattern=self.__pattern,
            dtype="" if self.__compression_handler.is_enabled else str(original_dtype),
            shape="" if self.__compression_handler.is_enabled else original_shape,
            port=self.__port if self.__multiserver_mode else None,
            multiserver_mode=self.__multiserver_mode,
        )

        self.__msg_socket.send_json(msg_dict, self.__msg_flag | zmq.SNDMORE)
        self.__msg_socket.send(
            encoded_data,
            flags=self.__msg_flag,
            copy=self.__msg_copy,
            track=self.__msg_track,
        )

        if self.__pattern < 2:
            if self.__bi_mode or self.__multiclient_mode:
                recvd_data = None

                socks = dict(self.__poll.poll(self.__request_timeout))
                if socks.get(self.__msg_socket) == zmq.POLLIN:
                    recv_json = self.__msg_socket.recv_json(flags=self.__msg_flag)
                else:
                    logger.critical("No response from Client, Reconnecting again...")
                    self.__msg_socket.setsockopt(zmq.LINGER, 0)
                    self.__msg_socket.close()
                    self.__poll.unregister(self.__msg_socket)
                    self.__max_retries -= 1

                    if not self.__max_retries:
                        if self.__multiclient_mode:
                            logger.error("All Clients failed to respond on multiple attempts.")
                        else:
                            logger.error("Client failed to respond on multiple attempts.")
                        self.__terminate.set()
                        raise RuntimeError("[SyncTransport:ERROR] :: Client(s) seems to be offline, Abandoning.")

                    self.__msg_socket = self.__msg_context.socket(self.__msg_pattern)
                    if isinstance(self.__connection_address, list):
                        for _connection in self.__connection_address:
                            self.__msg_socket.connect(_connection)
                    else:
                        if self.__ssh_tunnel_mode:
                            ssh.tunnel_connection(
                                self.__msg_socket,
                                self.__connection_address,
                                self.__ssh_tunnel_mode,
                                keyfile=self.__ssh_tunnel_keyfile,
                                password=self.__ssh_tunnel_pwd,
                                paramiko=self.__paramiko_present,
                            )
                        else:
                            self.__msg_socket.connect(self.__connection_address)
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    return None

                if self.__multiclient_mode and recv_json["port"] not in self.__port_buffer:
                    self.__port_buffer.append(recv_json["port"])

                if recv_json["return_type"] == "ndarray":
                    recv_array = self.__msg_socket.recv(
                        flags=self.__msg_flag,
                        copy=self.__msg_copy,
                        track=self.__msg_track,
                    )

                    recv_compression = recv_json.get("compression")
                    if recv_compression:
                        recvd_data = decode_sync_frame(
                            bytes(recv_array),
                            recv_compression,
                            self.__compression_handler,
                            self.__jpeg_compression_fastdct,
                            self.__jpeg_compression_fastupsample,
                        )
                        if recvd_data is None:
                            self.__logging and logger.debug("Return frame not yet decodable, skipping.")
                    else:
                        recvd_data = np.frombuffer(
                            recv_array, dtype=recv_json["array_dtype"]
                        ).reshape(recv_json["array_shape"])
                else:
                    recvd_data = recv_json["data"]

                return (recv_json["port"], recvd_data) if self.__multiclient_mode else recvd_data
            else:
                socks = dict(self.__poll.poll(self.__request_timeout))
                if socks.get(self.__msg_socket) == zmq.POLLIN:
                    recv_confirmation = self.__msg_socket.recv()
                else:
                    logger.critical("No response from Client, Reconnecting again...")
                    self.__msg_socket.setsockopt(zmq.LINGER, 0)
                    self.__msg_socket.close()
                    self.__poll.unregister(self.__msg_socket)
                    self.__max_retries -= 1

                    if not self.__max_retries:
                        logger.error("Client failed to respond on repeated attempts.")
                        self.__terminate.set()
                        raise RuntimeError("[SyncTransport:ERROR] :: Client seems to be offline, Abandoning!")

                    self.__msg_socket = self.__msg_context.socket(self.__msg_pattern)
                    if self.__ssh_tunnel_mode:
                        ssh.tunnel_connection(
                            self.__msg_socket,
                            self.__connection_address,
                            self.__ssh_tunnel_mode,
                            keyfile=self.__ssh_tunnel_keyfile,
                            password=self.__ssh_tunnel_pwd,
                            paramiko=self.__paramiko_present,
                        )
                    else:
                        self.__msg_socket.connect(self.__connection_address)
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    return None

                self.__logging and logger.debug(recv_confirmation)

    def close(self, kill: bool = False) -> None:
        self.__logging and logger.debug(
            "Terminating various {} Processes.".format(
                "Receive Mode" if self.__receive_mode else "Send Mode"
            )
        )

        self.__compression_handler.close()

        if self.__receive_mode:
            if self.__queue is not None and self.__queue:
                self.__queue.clear()
            self.__terminate.set()
            self.__logging and logger.debug("Terminating. Please wait...")
            if self.__z_auth:
                self.__logging and logger.debug("Terminating Authenticator Thread.")
                self.__z_auth.stop()
                while self.__z_auth.is_alive():
                    pass
            if self.__thread is not None:
                self.__logging and logger.debug("Terminating Main Thread.")
                if self.__thread.is_alive() and kill:
                    logger.warning("Thread still running...Killing it forcefully!")
                    self.__msg_context.destroy()
                    self.__thread.join()
                else:
                    self.__msg_socket.close(linger=0)
                    self.__thread.join()
                self.__thread = None
            self.__logging and logger.debug("Terminated Successfully!")
        else:
            self.__terminate.set()
            kill and logger.warning("`kill` parameter is only available in the receive mode.")
            if self.__z_auth:
                self.__logging and logger.debug("Terminating Authenticator Thread.")
                self.__z_auth.stop()
                while self.__z_auth.is_alive():
                    pass
            if (self.__pattern < 2 and not self.__max_retries) or (
                self.__multiclient_mode and not self.__port_buffer
            ):
                try:
                    self.__msg_socket.setsockopt(zmq.LINGER, 0)
                    self.__msg_socket.close()
                except ZMQError:
                    pass
                finally:
                    return

            if self.__multiserver_mode:
                term_dict = dict(terminate_flag=True, port=self.__port)
            else:
                term_dict = dict(terminate_flag=True)

            try:
                if self.__multiclient_mode:
                    for _ in self.__port_buffer:
                        self.__msg_socket.send_json(term_dict)
                else:
                    self.__msg_socket.send_json(term_dict)

                if self.__pattern < 2:
                    self.__logging and logger.debug("Terminating. Please wait...")
                    if self.__msg_socket.poll(self.__request_timeout // 5, zmq.POLLIN):
                        self.__msg_socket.recv()
            except Exception as e:
                if not isinstance(e, ZMQError):
                    logger.exception(str(e))
            finally:
                self.__msg_socket.setsockopt(zmq.LINGER, 0)
                self.__msg_socket.close()
                self.__logging and logger.debug("Terminated Successfully!")