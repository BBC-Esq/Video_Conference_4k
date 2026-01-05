import os
import time
import asyncio
import platform
import string
import secrets
import numpy as np
import logging as log
from threading import Thread, Lock, Event
from collections import deque
from os.path import expanduser
from numpy.typing import NDArray
from typing import Optional, Any, Tuple

from ..utils.common import (
    logger_handler,
    generate_auth_certificates,
    check_WriteAccess,
    check_open_port,
    import_dependency_safe,
    log_version,
)

zmq = import_dependency_safe("zmq", pkg_name="pyzmq", error="silent", min_version="4.0")
if not (zmq is None):
    from zmq import ssh
    from zmq import auth
    from zmq.auth.thread import ThreadAuthenticator
    from zmq.error import ZMQError
simplejpeg = import_dependency_safe("simplejpeg", error="silent", min_version="1.6.1")
paramiko = import_dependency_safe("paramiko", error="silent")

logger = log.getLogger("SyncTransport")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


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

        self.__gpu_accelerated = False
        self.__gpu_id = gpu_id
        self.__gpu_resolution = gpu_resolution
        self.__gpu_bitrate = gpu_bitrate
        self.__gpu_codec = gpu_codec
        self.__nvidia_encoder = None
        self.__nvidia_decoder = None

        if gpu_accelerated:
            try:
                from ..utils.nvidia_codec import has_nvidia_codec, NvidiaEncoder, NvidiaDecoder
                if has_nvidia_codec():
                    self.__gpu_accelerated = True
                    self.__logging and logger.info("GPU acceleration enabled with NVIDIA hardware encoding")
                else:
                    logger.warning("GPU acceleration requested but NVIDIA codec not available. Falling back to CPU.")
            except ImportError as e:
                logger.warning("GPU acceleration requested but PyNvVideoCodec not installed: {}. Falling back to CPU.".format(e))

        valid_messaging_patterns = {
            0: (zmq.PAIR, zmq.PAIR),
            1: (zmq.REQ, zmq.REP),
            2: (zmq.PUB, zmq.SUB),
        }

        msg_pattern = None
        if isinstance(pattern, int) and pattern in valid_messaging_patterns.keys():
            msg_pattern = valid_messaging_patterns[pattern]
        else:
            pattern = 0
            msg_pattern = valid_messaging_patterns[pattern]
            self.__logging and logger.warning(
                "Wrong pattern value, Defaulting to `zmq.PAIR`! Kindly refer Docs for more Information."
            )
        self.__pattern = pattern

        if protocol is None or not (protocol in ["tcp", "ipc"]):
            protocol = "tcp"
            self.__logging and logger.warning(
                "Protocol is not supported or not provided. Defaulting to `tcp` protocol!"
            )

        self.__msg_flag = 0
        self.__msg_copy = False
        self.__msg_track = False

        self.__z_auth = None

        self.__ssh_tunnel_mode = None
        self.__ssh_tunnel_pwd = None
        self.__ssh_tunnel_keyfile = None
        self.__paramiko_present = False if paramiko is None else True

        self.__multiserver_mode = False

        self.__multiclient_mode = False

        self.__bi_mode = False

        valid_security_mech = {0: "Grasslands", 1: "StoneHouse", 2: "IronHouse"}
        self.__secure_mode = 0
        auth_cert_dir = ""
        self.__auth_publickeys_dir = ""
        self.__auth_secretkeys_dir = ""
        overwrite_cert = False
        custom_cert_location = ""

        self.__jpeg_compression = (
            True if not (simplejpeg is None) and not self.__gpu_accelerated else False
        )
        self.__jpeg_compression_quality = 90
        self.__jpeg_compression_fastdct = True
        self.__jpeg_compression_fastupsample = False
        self.__jpeg_compression_colorspace = "BGR"

        self.__ex_compression_params = None

        self.__return_data = None
        self.__return_data_lock = Lock()

        self.__id = "".join(
            secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)
        )

        self.__terminate = Event()

        if pattern < 2:
            self.__poll = zmq.Poller()
            self.__max_retries = 3
            self.__request_timeout = 4000
        else:
            self.__subscriber_timeout = None

        options = {str(k).strip(): v for k, v in options.items()}

        for key, value in options.items():
            if key == "multiserver_mode" and isinstance(value, bool):
                if pattern > 0:
                    self.__multiserver_mode = value
                else:
                    self.__multiserver_mode = False
                    logger.critical("Multi-Server Mode is disabled!")
                    raise ValueError(
                        "[SyncTransport:ERROR] :: `{}` pattern is not valid when Multi-Server Mode is enabled. Kindly refer Docs for more Information.".format(
                            pattern
                        )
                    )

            elif key == "multiclient_mode" and isinstance(value, bool):
                if pattern > 0:
                    self.__multiclient_mode = value
                else:
                    self.__multiclient_mode = False
                    logger.critical("Multi-Client Mode is disabled!")
                    raise ValueError(
                        "[SyncTransport:ERROR] :: `{}` pattern is not valid when Multi-Client Mode is enabled. Kindly refer Docs for more Information.".format(
                            pattern
                        )
                    )

            elif key == "bidirectional_mode" and isinstance(value, bool):
                if pattern < 2:
                    self.__bi_mode = value
                else:
                    self.__bi_mode = False
                    logger.warning("Bidirectional data transmission is disabled!")
                    raise ValueError(
                        "[SyncTransport:ERROR] :: `{}` pattern is not valid when Bidirectional Mode is enabled. Kindly refer Docs for more Information!".format(
                            pattern
                        )
                    )

            elif (
                key == "secure_mode"
                and isinstance(value, int)
                and (value in valid_security_mech)
            ):
                self.__secure_mode = value

            elif key == "custom_cert_location" and isinstance(value, str):
                custom_cert_location = os.path.abspath(value)
                assert os.path.isdir(
                    custom_cert_location
                ), "[SyncTransport:ERROR] :: `custom_cert_location` value must be the path to a valid directory!"
                assert check_WriteAccess(
                    custom_cert_location,
                    is_windows=True if os.name == "nt" else False,
                    logging=self.__logging,
                ), "[SyncTransport:ERROR] :: Permission Denied!, cannot write ZMQ authentication certificates to '{}' directory!".format(
                    value
                )
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
                        "Discarded invalid or non-existential SSH Tunnel Key-file at {}!".format(
                            value
                        )
                    )

            elif (
                key == "jpeg_compression"
                and not (simplejpeg is None)
                and isinstance(value, (bool, str))
                and not self.__gpu_accelerated
            ):
                if isinstance(value, str) and value.strip().upper() in [
                    "RGB",
                    "BGR",
                    "RGBX",
                    "BGRX",
                    "XBGR",
                    "XRGB",
                    "GRAY",
                    "RGBA",
                    "BGRA",
                    "ABGR",
                    "ARGB",
                    "CMYK",
                ]:
                    self.__jpeg_compression_colorspace = value.strip().upper()
                    self.__jpeg_compression = True
                else:
                    self.__jpeg_compression = value
            elif key == "jpeg_compression_quality" and isinstance(value, (int, float)):
                if value >= 10 and value <= 100:
                    self.__jpeg_compression_quality = int(value)
                else:
                    logger.warning("Skipped invalid `jpeg_compression_quality` value!")
            elif key == "jpeg_compression_fastdct" and isinstance(value, bool):
                self.__jpeg_compression_fastdct = value
            elif key == "jpeg_compression_fastupsample" and isinstance(value, bool):
                self.__jpeg_compression_fastupsample = value

            elif key == "max_retries" and isinstance(value, int) and pattern < 2:
                if value >= 0:
                    self.__max_retries = value
                else:
                    logger.warning("Invalid `max_retries` value skipped!")

            elif key == "request_timeout" and isinstance(value, int) and pattern < 2:
                if value >= 4:
                    self.__request_timeout = value * 1000
                else:
                    logger.warning("Invalid `request_timeout` value skipped!")

            elif (
                key == "subscriber_timeout" and isinstance(value, int) and pattern == 2
            ):
                if value > 0:
                    self.__subscriber_timeout = value * 1000
                else:
                    logger.warning("Invalid `request_timeout` value skipped!")

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
            else:
                pass

        if not (self.__ssh_tunnel_mode is None):
            if receive_mode:
                logger.error("SSH Tunneling cannot be enabled for Client-end!")
            else:
                ssh_address = self.__ssh_tunnel_mode
                ssh_address, ssh_port = (
                    ssh_address.split(":")
                    if ":" in ssh_address
                    else [ssh_address, "22"]
                )
                if "47" in ssh_port:
                    self.__ssh_tunnel_mode = self.__ssh_tunnel_mode.replace(
                        ":47", ""
                    )
                else:
                    ssh_user, ssh_ip = (
                        ssh_address.split("@")
                        if "@" in ssh_address
                        else ["", ssh_address]
                    )
                    assert check_open_port(
                        ssh_ip, port=int(ssh_port)
                    ), "[SyncTransport:ERROR] :: Host `{}` is not available for SSH Tunneling at port-{}!".format(
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

        if self.__secure_mode > 0:
            if receive_mode:
                overwrite_cert = False
                overwrite_cert and logger.warning(
                    "Overwriting ZMQ Authentication certificates is disabled for Client's end!"
                )
            else:
                overwrite_cert and self.__logging and logger.info(
                    "Overwriting ZMQ Authentication certificates over previous ones!"
                )

            try:
                if custom_cert_location:
                    (
                        auth_cert_dir,
                        self.__auth_secretkeys_dir,
                        self.__auth_publickeys_dir,
                    ) = generate_auth_certificates(
                        custom_cert_location, overwrite=overwrite_cert, logging=logging
                    )
                else:
                    (
                        auth_cert_dir,
                        self.__auth_secretkeys_dir,
                        self.__auth_publickeys_dir,
                    ) = generate_auth_certificates(
                        os.path.join(expanduser("~"), ".videoconference4k"),
                        overwrite=overwrite_cert,
                        logging=logging,
                    )
                self.__logging and logger.debug(
                    "`{}` is the default location for storing ZMQ authentication certificates/keys.".format(
                        auth_cert_dir
                    )
                )

                self.__z_auth = ThreadAuthenticator(self.__msg_context)
                self.__z_auth.start()
                self.__z_auth.allow(str(address))

                if self.__secure_mode == 2:
                    self.__z_auth.configure_curve(
                        domain="*", location=self.__auth_publickeys_dir
                    )
                else:
                    self.__z_auth.configure_curve(
                        domain="*", location=auth.CURVE_ALLOW_ANY
                    )
            except zmq.ZMQError as e:
                if "Address in use" in str(e):
                    logger.info("ZMQ Authenticator already running.")
                else:
                    logger.exception(str(e))
                    self.__secure_mode = 0
                    logger.error(
                        "ZMQ Security Mechanism is disabled for this connection due to errors!"
                    )

        if self.__receive_mode:
            address = "*" if address is None else address

            if self.__multiserver_mode:
                if port is None or not isinstance(port, (tuple, list)):
                    raise ValueError(
                        "[SyncTransport:ERROR] :: Incorrect port value! Kindly provide a list/tuple of Server ports while Multi-Server mode is enabled. For more information refer VideoConference4k docs."
                    )
                else:
                    logger.debug(
                        "Enabling Multi-Server Mode at PORTS: {}!".format(port)
                    )
                self.__port_buffer = []
            elif self.__multiclient_mode:
                if port is None:
                    raise ValueError(
                        "[SyncTransport:ERROR] :: Kindly provide a unique & valid port value at Client-end. For more information refer VideoConference4k docs."
                    )
                else:
                    logger.debug(
                        "Enabling Multi-Client Mode at PORT: {} on this device!".format(
                            port
                        )
                    )
                self.__port = port
            else:
                port = "5555" if port is None else port

            try:
                self.__msg_socket = self.__msg_context.socket(msg_pattern[1])

                self.__pattern == 2 and self.__msg_socket.set_hwm(1)

                if self.__secure_mode > 0:
                    server_secret_file = os.path.join(
                        self.__auth_secretkeys_dir, "server.key_secret"
                    )
                    server_public, server_secret = auth.load_certificate(
                        server_secret_file
                    )
                    self.__msg_socket.curve_secretkey = server_secret
                    self.__msg_socket.curve_publickey = server_public
                    self.__msg_socket.curve_server = True

                if self.__pattern == 2:
                    self.__msg_socket.setsockopt_string(zmq.SUBSCRIBE, "")
                    self.__subscriber_timeout and self.__msg_socket.setsockopt(
                        zmq.RCVTIMEO, self.__subscriber_timeout
                    )
                    self.__subscriber_timeout and self.__msg_socket.setsockopt(
                        zmq.LINGER, 0
                    )

                if self.__multiserver_mode:
                    for pt in port:
                        self.__msg_socket.bind(
                            protocol + "://" + str(address) + ":" + str(pt)
                        )
                else:
                    self.__msg_socket.bind(
                        protocol + "://" + str(address) + ":" + str(port)
                    )

                if pattern < 2:
                    if self.__multiserver_mode:
                        self.__connection_address = []
                        for pt in port:
                            self.__connection_address.append(
                                protocol + "://" + str(address) + ":" + str(pt)
                            )
                    else:
                        self.__connection_address = (
                            protocol + "://" + str(address) + ":" + str(port)
                        )
                    self.__msg_pattern = msg_pattern[1]
                    self.__poll.register(self.__msg_socket, zmq.POLLIN)
                    self.__logging and logger.debug(
                        "Reliable transmission is enabled for this pattern with max-retries: {} and timeout: {} secs.".format(
                            self.__max_retries, self.__request_timeout / 1000
                        )
                    )
                else:
                    self.__logging and self.__subscriber_timeout and logger.debug(
                        "Timeout: {} secs is enabled for this system.".format(
                            self.__subscriber_timeout / 1000
                        )
                    )

            except Exception as e:
                logger.exception(str(e))
                self.__secure_mode and logger.critical(
                    "Failed to activate Secure Mode: `{}` for this connection!".format(
                        valid_security_mech[self.__secure_mode]
                    )
                )
                if self.__multiserver_mode or self.__multiclient_mode:
                    raise RuntimeError(
                        "[SyncTransport:ERROR] :: Receive Mode failed to activate {} Mode at address: {} with pattern: {}! Kindly recheck all parameters.".format(
                            (
                                "Multi-Server"
                                if self.__multiserver_mode
                                else "Multi-Client"
                            ),
                            (protocol + "://" + str(address) + ":" + str(port)),
                            pattern,
                        )
                    )
                else:
                    self.__bi_mode and logger.critical(
                        "Failed to activate Bidirectional Mode for this connection!"
                    )
                    raise RuntimeError(
                        "[SyncTransport:ERROR] :: Receive Mode failed to bind address: {} and pattern: {}! Kindly recheck all parameters.".format(
                            (protocol + "://" + str(address) + ":" + str(port)), pattern
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
                        (protocol + "://" + str(address) + ":" + str(port)), pattern
                    )
                )
                if self.__gpu_accelerated:
                    logger.debug(
                        "GPU-Accelerated encoding is activated for this connection using NVIDIA hardware."
                    )
                elif self.__jpeg_compression:
                    logger.debug(
                        "JPEG Frame-Compression is activated for this connection with Colorspace:`{}`, Quality:`{}`%, Fastdct:`{}`, and Fastupsample:`{}`.".format(
                            self.__jpeg_compression_colorspace,
                            self.__jpeg_compression_quality,
                            ("enabled" if self.__jpeg_compression_fastdct else "disabled"),
                            (
                                "enabled"
                                if self.__jpeg_compression_fastupsample
                                else "disabled"
                            ),
                        )
                    )
                self.__secure_mode and logger.debug(
                    "Successfully enabled ZMQ Security Mechanism: `{}` for this connection.".format(
                        valid_security_mech[self.__secure_mode]
                    )
                )
                logger.debug("Multi-threaded Receive Mode is successfully enabled.")
                logger.debug("Unique System ID is {}.".format(self.__id))
                logger.debug("Receive Mode is now activated.")

        else:
            address = "localhost" if address is None else address

            if self.__multiserver_mode:
                if port is None:
                    raise ValueError(
                        "[SyncTransport:ERROR] :: Kindly provide a unique & valid port value at Server-end. For more information refer VideoConference4k docs."
                    )
                else:
                    logger.debug(
                        "Enabling Multi-Server Mode at PORT: {} on this device!".format(
                            port
                        )
                    )
                self.__port = port
            elif self.__multiclient_mode:
                if port is None or not isinstance(port, (tuple, list)):
                    raise ValueError(
                        "[SyncTransport:ERROR] :: Incorrect port value! Kindly provide a list/tuple of Client ports while Multi-Client mode is enabled. For more information refer VideoConference4k docs."
                    )
                else:
                    logger.debug(
                        "Enabling Multi-Client Mode at PORTS: {}!".format(port)
                    )
                self.__port_buffer = []
            else:
                port = "5555" if port is None else port

            try:
                self.__msg_socket = self.__msg_context.socket(msg_pattern[0])

                if self.__pattern == 1:
                    self.__msg_socket.REQ_RELAXED = True
                    self.__msg_socket.REQ_CORRELATE = True

                if self.__pattern == 2:
                    self.__msg_socket.set_hwm(1)

                if self.__secure_mode > 0:
                    client_secret_file = os.path.join(
                        self.__auth_secretkeys_dir, "client.key_secret"
                    )
                    client_public, client_secret = auth.load_certificate(
                        client_secret_file
                    )
                    self.__msg_socket.curve_secretkey = client_secret
                    self.__msg_socket.curve_publickey = client_public
                    server_public_file = os.path.join(
                        self.__auth_publickeys_dir, "server.key"
                    )
                    server_public, _ = auth.load_certificate(server_public_file)
                    self.__msg_socket.curve_serverkey = server_public

                if self.__multiclient_mode:
                    for pt in port:
                        self.__msg_socket.connect(
                            protocol + "://" + str(address) + ":" + str(pt)
                        )
                else:
                    if self.__ssh_tunnel_mode:
                        ssh.tunnel_connection(
                            self.__msg_socket,
                            protocol + "://" + str(address) + ":" + str(port),
                            self.__ssh_tunnel_mode,
                            keyfile=self.__ssh_tunnel_keyfile,
                            password=self.__ssh_tunnel_pwd,
                            paramiko=self.__paramiko_present,
                        )
                    else:
                        self.__msg_socket.connect(
                            protocol + "://" + str(address) + ":" + str(port)
                        )

                if pattern < 2:
                    if self.__multiclient_mode:
                        self.__connection_address = []
                        for pt in port:
                            self.__connection_address.append(
                                protocol + "://" + str(address) + ":" + str(pt)
                            )
                    else:
                        self.__connection_address = (
                            protocol + "://" + str(address) + ":" + str(port)
                        )
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
                        valid_security_mech[self.__secure_mode]
                    )
                )
                if self.__multiserver_mode or self.__multiclient_mode:
                    raise RuntimeError(
                        "[SyncTransport:ERROR] :: Send Mode failed to activate {} Mode at address: {} with pattern: {}! Kindly recheck all parameters.".format(
                            (
                                "Multi-Server"
                                if self.__multiserver_mode
                                else "Multi-Client"
                            ),
                            (protocol + "://" + str(address) + ":" + str(port)),
                            pattern,
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
                        "[SyncTransport:ERROR] :: Send Mode failed to connect address: {} and pattern: {}! Kindly recheck all parameters.".format(
                            (protocol + "://" + str(address) + ":" + str(port)), pattern
                        )
                    )

            if self.__logging:
                logger.debug(
                    "Successfully connected to address: {} with pattern: {}.".format(
                        (protocol + "://" + str(address) + ":" + str(port)), pattern
                    )
                )
                if self.__gpu_accelerated:
                    logger.debug(
                        "GPU-Accelerated encoding is activated for this connection using NVIDIA hardware."
                    )
                elif self.__jpeg_compression:
                    logger.debug(
                        "JPEG Frame-Compression is activated for this connection with Colorspace:`{}`, Quality:`{}`%, Fastdct:`{}`, and Fastupsample:`{}`.".format(
                            self.__jpeg_compression_colorspace,
                            self.__jpeg_compression_quality,
                            ("enabled" if self.__jpeg_compression_fastdct else "disabled"),
                            (
                                "enabled"
                                if self.__jpeg_compression_fastupsample
                                else "disabled"
                            ),
                        )
                    )
                self.__secure_mode and logger.debug(
                    "Enabled ZMQ Security Mechanism: `{}` for this connection.".format(
                        valid_security_mech[self.__secure_mode]
                    )
                )
                logger.debug("Unique System ID is {}.".format(self.__id))
                logger.debug(
                    "Send Mode is successfully activated and ready to send data."
                )

    def __get_nvidia_encoder(self, width: int, height: int):
        if self.__nvidia_encoder is None:
            from ..utils.nvidia_codec import NvidiaEncoder
            self.__nvidia_encoder = NvidiaEncoder(
                width=width,
                height=height,
                bitrate=self.__gpu_bitrate,
                codec=self.__gpu_codec,
                gpu_id=self.__gpu_id,
                logging=self.__logging,
            )
        return self.__nvidia_encoder

    def __get_nvidia_decoder(self):
        if self.__nvidia_decoder is None:
            from ..utils.nvidia_codec import NvidiaDecoder
            self.__nvidia_decoder = NvidiaDecoder(
                gpu_id=self.__gpu_id,
                logging=self.__logging,
            )
        return self.__nvidia_decoder

    def __recv_handler(self):
        frame = None
        msg_json = None

        while not self.__terminate.is_set():
            if len(self.__queue) >= 96:
                time.sleep(0.000001)
                continue

            if self.__pattern < 2:
                socks = dict(self.__poll.poll(self.__request_timeout * 3))
                if socks.get(self.__msg_socket) == zmq.POLLIN:
                    msg_json = self.__msg_socket.recv_json(
                        flags=self.__msg_flag | zmq.DONTWAIT
                    )
                else:
                    logger.critical("No response from Server(s), Reconnecting again...")
                    self.__msg_socket.close(linger=0)
                    self.__poll.unregister(self.__msg_socket)
                    self.__max_retries -= 1

                    if not (self.__max_retries):
                        if self.__multiserver_mode:
                            logger.error("All Servers seems to be offline, Abandoning!")
                        else:
                            logger.error("Server seems to be offline, Abandoning!")
                        self.__terminate.set()
                        continue

                    try:
                        self.__msg_socket = self.__msg_context.socket(
                            self.__msg_pattern
                        )
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
                            "Termination signal received from Server at port: {}!".format(
                                msg_json["port"]
                            )
                        )
                    if not self.__port_buffer:
                        logger.critical(
                            "Termination signal received from all Servers!!!"
                        )
                        self.__terminate.set()
                else:
                    if self.__pattern == 1:
                        self.__msg_socket.send_string(
                            "Termination signal successfully received at Client's end!"
                        )
                    self.__terminate.set()
                    self.__logging and logger.critical(
                        "Termination signal received from server!"
                    )
                continue

            try:
                msg_data = self.__msg_socket.recv(
                    flags=self.__msg_flag | zmq.DONTWAIT,
                    copy=self.__msg_copy,
                    track=self.__msg_track,
                )
            except zmq.ZMQError as e:
                logger.critical("Socket Session Expired. Exiting!")
                self.__terminate.set()
                self.__queue.append(None)
                break

            if self.__pattern < 2:
                if self.__bi_mode or self.__multiclient_mode:
                    with self.__return_data_lock:
                        local_return_data = self.__return_data

                    if not (local_return_data is None) and isinstance(
                        local_return_data, np.ndarray
                    ):
                        return_data = np.copy(local_return_data)

                        if not (return_data.flags["C_CONTIGUOUS"]):
                            return_data = np.ascontiguousarray(
                                return_data, dtype=return_data.dtype
                            )

                        if self.__gpu_accelerated:
                            encoder = self.__get_nvidia_encoder(
                                return_data.shape[1], return_data.shape[0]
                            )
                            return_data = encoder.encode(return_data)

                            return_dict = (
                                dict(port=self.__port)
                                if self.__multiclient_mode
                                else dict()
                            )

                            return_dict.update(
                                dict(
                                    return_type=(type(local_return_data).__name__),
                                    compression={
                                        "type": "nvenc",
                                        "codec": self.__gpu_codec,
                                        "width": local_return_data.shape[1],
                                        "height": local_return_data.shape[0],
                                    },
                                    array_dtype="",
                                    array_shape="",
                                    data=None,
                                )
                            )

                            self.__msg_socket.send_json(
                                return_dict, self.__msg_flag | zmq.SNDMORE
                            )
                            self.__msg_socket.send(
                                return_data,
                                flags=self.__msg_flag,
                                copy=self.__msg_copy,
                                track=self.__msg_track,
                            )
                        elif self.__jpeg_compression:
                            if self.__jpeg_compression_colorspace == "GRAY":
                                if return_data.ndim == 2:
                                    return_data = return_data[:, :, np.newaxis]
                                return_data = simplejpeg.encode_jpeg(
                                    return_data,
                                    quality=self.__jpeg_compression_quality,
                                    colorspace=self.__jpeg_compression_colorspace,
                                    fastdct=self.__jpeg_compression_fastdct,
                                )
                            else:
                                return_data = simplejpeg.encode_jpeg(
                                    return_data,
                                    quality=self.__jpeg_compression_quality,
                                    colorspace=self.__jpeg_compression_colorspace,
                                    colorsubsampling="422",
                                    fastdct=self.__jpeg_compression_fastdct,
                                )

                            return_dict = (
                                dict(port=self.__port)
                                if self.__multiclient_mode
                                else dict()
                            )

                            return_dict.update(
                                dict(
                                    return_type=(type(local_return_data).__name__),
                                    compression=(
                                        {
                                            "dct": self.__jpeg_compression_fastdct,
                                            "ups": self.__jpeg_compression_fastupsample,
                                            "colorspace": self.__jpeg_compression_colorspace,
                                        }
                                    ),
                                    array_dtype=(
                                        str(local_return_data.dtype)
                                        if not (self.__jpeg_compression)
                                        else ""
                                    ),
                                    array_shape=(
                                        local_return_data.shape
                                        if not (self.__jpeg_compression)
                                        else ""
                                    ),
                                    data=None,
                                )
                            )

                            self.__msg_socket.send_json(
                                return_dict, self.__msg_flag | zmq.SNDMORE
                            )
                            self.__msg_socket.send(
                                return_data,
                                flags=self.__msg_flag,
                                copy=self.__msg_copy,
                                track=self.__msg_track,
                            )
                        else:
                            return_dict = (
                                dict(port=self.__port)
                                if self.__multiclient_mode
                                else dict()
                            )
                            return_dict.update(
                                dict(
                                    return_type=(type(local_return_data).__name__),
                                    compression=False,
                                    array_dtype=str(local_return_data.dtype),
                                    array_shape=local_return_data.shape,
                                    data=None,
                                )
                            )
                            self.__msg_socket.send_json(
                                return_dict, self.__msg_flag | zmq.SNDMORE
                            )
                            self.__msg_socket.send(
                                return_data,
                                flags=self.__msg_flag,
                                copy=self.__msg_copy,
                                track=self.__msg_track,
                            )
                    else:
                        return_dict = (
                            dict(port=self.__port)
                            if self.__multiclient_mode
                            else dict()
                        )
                        return_dict.update(
                            dict(
                                return_type=(type(local_return_data).__name__),
                                data=local_return_data,
                            )
                        )
                        self.__msg_socket.send_json(return_dict, self.__msg_flag)
                else:
                    self.__msg_socket.send_string(
                        "Data received on device: {} !".format(self.__id)
                    )
            else:
                with self.__return_data_lock:
                    local_return_data = self.__return_data
                if local_return_data:
                    logger.warning("`return_data` is disabled for this pattern!")

            if msg_json.get("compression"):
                compression_info = msg_json["compression"]

                if isinstance(compression_info, dict) and compression_info.get("type") == "nvenc":
                    decoder = self.__get_nvidia_decoder()
                    frame = decoder.decode(
                        bytes(msg_data),
                        width=compression_info.get("width"),
                        height=compression_info.get("height")
                    )
                    if frame is None:
                        self.__terminate.set()
                        raise RuntimeError(
                            "[SyncTransport:ERROR] :: Received NVENC frame decoding failed. "
                            "Width: {}, Height: {}".format(
                                compression_info.get("width"),
                                compression_info.get("height")
                            )
                        )
                else:
                    frame = simplejpeg.decode_jpeg(
                        msg_data,
                        colorspace=compression_info["colorspace"],
                        fastdct=self.__jpeg_compression_fastdct
                        or compression_info["dct"],
                        fastupsample=self.__jpeg_compression_fastupsample
                        or compression_info["ups"],
                    )
                    if frame is None:
                        self.__terminate.set()
                        raise RuntimeError(
                            "[SyncTransport:ERROR] :: Received compressed JPEG frame decoding failed"
                        )
                    if compression_info["colorspace"] == "GRAY" and frame.ndim == 3:
                        frame = np.squeeze(frame, axis=2)
            else:
                frame_buffer = np.frombuffer(msg_data, dtype=msg_json["dtype"])
                frame = frame_buffer.reshape(msg_json["shape"])

            if self.__multiserver_mode:
                if not msg_json["port"] in self.__port_buffer:
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
        if not (self.__receive_mode):
            self.__terminate.set()
            raise ValueError(
                "[SyncTransport:ERROR] :: `recv()` function cannot be used while receive_mode is disabled. Kindly refer VideoConference4k docs!"
            )

        if (self.__bi_mode or self.__multiclient_mode) and not (return_data is None):
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
                "[SyncTransport:ERROR] :: `send()` function cannot be used while receive_mode is enabled. Kindly refer VideoConference4k docs!"
            )

        if not (message is None) and isinstance(message, np.ndarray):
            logger.warning(
                "Skipped unsupported `message` of datatype: {}!".format(
                    type(message).__name__
                )
            )
            message = None

        exit_flag = True if (frame is None or self.__terminate.is_set()) else False

        if exit_flag:
            msg_dict = dict(port=self.__port) if self.__multiserver_mode else dict()
            msg_dict.update(
                dict(
                    terminate_flag=True,
                    compression=False,
                    message=None,
                    pattern=str(self.__pattern),
                    dtype="",
                    shape="",
                )
            )
            self.__msg_socket.send_json(msg_dict, self.__msg_flag | zmq.SNDMORE)
            self.__msg_socket.send(
                b"",
                flags=self.__msg_flag,
                copy=self.__msg_copy,
                track=self.__msg_track,
            )

            if self.__pattern < 2:
                if self.__bi_mode or self.__multiclient_mode:
                    socks = dict(self.__poll.poll(self.__request_timeout))
                    if socks.get(self.__msg_socket) == zmq.POLLIN:
                        self.__msg_socket.recv_json(flags=self.__msg_flag)
                else:
                    socks = dict(self.__poll.poll(self.__request_timeout))
                    if socks.get(self.__msg_socket) == zmq.POLLIN:
                        self.__msg_socket.recv()
            return None

        if not (frame.flags["C_CONTIGUOUS"]):
            frame = np.ascontiguousarray(frame, dtype=frame.dtype)

        original_shape = frame.shape
        original_dtype = frame.dtype

        if self.__gpu_accelerated:
            encoder = self.__get_nvidia_encoder(frame.shape[1], frame.shape[0])
            encoded_frame = encoder.encode(frame)

            msg_dict = dict(port=self.__port) if self.__multiserver_mode else dict()

            msg_dict.update(
                dict(
                    terminate_flag=False,
                    compression={
                        "type": "nvenc",
                        "codec": self.__gpu_codec,
                        "width": original_shape[1],
                        "height": original_shape[0],
                    },
                    message=message,
                    pattern=str(self.__pattern),
                    dtype="",
                    shape="",
                )
            )

            self.__msg_socket.send_json(msg_dict, self.__msg_flag | zmq.SNDMORE)
            self.__msg_socket.send(
                encoded_frame, flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track
            )

        elif self.__jpeg_compression:
            if self.__jpeg_compression_colorspace == "GRAY":
                if frame.ndim == 2:
                    frame = np.expand_dims(frame, axis=2)
                frame = simplejpeg.encode_jpeg(
                    frame,
                    quality=self.__jpeg_compression_quality,
                    colorspace=self.__jpeg_compression_colorspace,
                    fastdct=self.__jpeg_compression_fastdct,
                )
            else:
                frame = simplejpeg.encode_jpeg(
                    frame,
                    quality=self.__jpeg_compression_quality,
                    colorspace=self.__jpeg_compression_colorspace,
                    colorsubsampling="422",
                    fastdct=self.__jpeg_compression_fastdct,
                )

            msg_dict = dict(port=self.__port) if self.__multiserver_mode else dict()

            msg_dict.update(
                dict(
                    terminate_flag=False,
                    compression=(
                        {
                            "dct": self.__jpeg_compression_fastdct,
                            "ups": self.__jpeg_compression_fastupsample,
                            "colorspace": self.__jpeg_compression_colorspace,
                        }
                    ),
                    message=message,
                    pattern=str(self.__pattern),
                    dtype=str(original_dtype),
                    shape=original_shape,
                )
            )

            self.__msg_socket.send_json(msg_dict, self.__msg_flag | zmq.SNDMORE)
            self.__msg_socket.send(
                frame, flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track
            )

        else:
            msg_dict = dict(port=self.__port) if self.__multiserver_mode else dict()

            msg_dict.update(
                dict(
                    terminate_flag=False,
                    compression=False,
                    message=message,
                    pattern=str(self.__pattern),
                    dtype=str(frame.dtype),
                    shape=frame.shape,
                )
            )

            self.__msg_socket.send_json(msg_dict, self.__msg_flag | zmq.SNDMORE)
            self.__msg_socket.send(
                frame, flags=self.__msg_flag, copy=self.__msg_copy, track=self.__msg_track
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

                    if not (self.__max_retries):
                        if self.__multiclient_mode:
                            logger.error(
                                "All Clients failed to respond on multiple attempts."
                            )
                        else:
                            logger.error(
                                "Client failed to respond on multiple attempts."
                            )
                        self.__terminate.set()
                        raise RuntimeError(
                            "[SyncTransport:ERROR] :: Client(s) seems to be offline, Abandoning."
                        )

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

                if (
                    self.__multiclient_mode
                    and not recv_json["port"] in self.__port_buffer
                ):
                    self.__port_buffer.append(recv_json["port"])

                if recv_json["return_type"] == "ndarray":
                    recv_array = self.__msg_socket.recv(
                        flags=self.__msg_flag,
                        copy=self.__msg_copy,
                        track=self.__msg_track,
                    )

                    recv_compression = recv_json.get("compression")

                    if isinstance(recv_compression, dict) and recv_compression.get("type") == "nvenc":
                        decoder = self.__get_nvidia_decoder()
                        recvd_data = decoder.decode(
                            bytes(recv_array),
                            width=recv_compression.get("width"),
                            height=recv_compression.get("height")
                        )
                        if recvd_data is None:
                            self.__terminate.set()
                            raise RuntimeError(
                                "[SyncTransport:ERROR] :: Received NVENC frame decoding failed. "
                                "Width: {}, Height: {}".format(
                                    recv_compression.get("width"),
                                    recv_compression.get("height")
                                )
                            )
                    elif recv_compression:
                        recvd_data = simplejpeg.decode_jpeg(
                            recv_array,
                            colorspace=recv_compression["colorspace"],
                            fastdct=self.__jpeg_compression_fastdct
                            or recv_compression["dct"],
                            fastupsample=self.__jpeg_compression_fastupsample
                            or recv_compression["ups"],
                        )
                        if recvd_data is None:
                            self.__terminate.set()
                            raise RuntimeError(
                                "[SyncTransport:ERROR] :: Received compressed frame `{}` decoding failed with flag: {}.".format(
                                    recv_compression,
                                    self.__ex_compression_params,
                                )
                            )

                        if (
                            recv_compression["colorspace"] == "GRAY"
                            and recvd_data.ndim == 3
                        ):
                            recvd_data = np.squeeze(recvd_data, axis=2)
                    else:
                        recvd_data = np.frombuffer(
                            recv_array, dtype=recv_json["array_dtype"]
                        ).reshape(recv_json["array_shape"])
                else:
                    recvd_data = recv_json["data"]

                return (
                    (recv_json["port"], recvd_data)
                    if self.__multiclient_mode
                    else recvd_data
                )
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

                    if not (self.__max_retries):
                        logger.error("Client failed to respond on repeated attempts.")
                        self.__terminate.set()
                        raise RuntimeError(
                            "[SyncTransport:ERROR] :: Client seems to be offline, Abandoning!"
                        )

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

        if self.__nvidia_encoder is not None:
            self.__nvidia_encoder.close()
            self.__nvidia_encoder = None

        if self.__nvidia_decoder is not None:
            self.__nvidia_decoder.close()
            self.__nvidia_decoder = None

        if self.__receive_mode:
            if not (self.__queue is None) and self.__queue:
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
            kill and logger.warning(
                "`kill` parmeter is only available in the receive mode."
            )
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