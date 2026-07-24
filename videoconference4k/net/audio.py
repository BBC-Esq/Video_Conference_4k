import struct
from collections import deque
from threading import Thread, Event
from typing import Optional, Tuple
from numpy.typing import NDArray

from ..utils.common import get_logger, import_dependency_safe
from ..codec import OpusEncoder, OpusDecoder
from .base import (
    validate_address,
    validate_protocol,
    validate_port,
    build_connection_string,
    apply_socket_qos,
    DSCP_AUDIO,
)

zmq = import_dependency_safe("zmq", pkg_name="pyzmq", error="silent", min_version="4.0")

logger = get_logger("AudioTransport")


class AudioTransport:

    def __init__(
        self,
        address: str = None,
        port: str = "5556",
        protocol: str = None,
        receive_mode: bool = False,
        sample_rate: int = 48000,
        channels: int = 1,
        bitrate: int = 32000,
        fec: bool = True,
        packet_loss: int = 10,
        hwm: int = 10,
        dscp: int = DSCP_AUDIO,
        logging: bool = False,
    ):
        import_dependency_safe("zmq" if zmq is None else "", pkg_name="pyzmq", min_version="4.0")

        self.__logging = logging if isinstance(logging, bool) else False
        self.__receive_mode = receive_mode
        self.__sample_rate = sample_rate
        self.__channels = channels

        protocol = validate_protocol(protocol)
        address = validate_address(address, receive_mode)
        port = validate_port(port)
        connection = build_connection_string(protocol, address, port)

        self.__context = zmq.Context.instance()
        self.__socket = self.__context.socket(zmq.PAIR)
        apply_socket_qos(self.__socket, dscp)
        self.__socket.set_hwm(hwm)

        self.__encoder = None
        self.__decoder = None
        self.__queue = None
        self.__terminate = None
        self.__thread = None

        if receive_mode:
            self.__socket.bind(connection)
            self.__decoder = OpusDecoder(
                sample_rate=sample_rate, channels=channels, logging=logging
            )
            self.__queue = deque(maxlen=max(2, hwm * 2))
            self.__terminate = Event()
            self.__thread = Thread(target=self.__recv_handler, name="AudioTransport", daemon=True)
            self.__thread.start()
        else:
            self.__socket.connect(connection)
            self.__encoder = OpusEncoder(
                sample_rate=sample_rate,
                channels=channels,
                bitrate=bitrate,
                fec=fec,
                packet_loss=packet_loss,
                logging=logging,
            )

        self.__logging and logger.debug(
            "AudioTransport {} on {}".format("receiving" if receive_mode else "sending", connection)
        )

    @property
    def sample_rate(self) -> int:
        return self.__sample_rate

    @property
    def channels(self) -> int:
        return self.__channels

    def send(self, pcm: NDArray, pts_ns: int = 0) -> None:
        if self.__receive_mode or self.__encoder is None:
            raise ValueError("[AudioTransport:ERROR] :: send() is only available in send mode.")
        if self.__socket is None:
            return
        data = self.__encoder.encode(pcm)
        if not data:
            return
        try:
            self.__socket.send(struct.pack(">q", int(pts_ns)) + data, flags=zmq.NOBLOCK)
        except zmq.Again:
            self.__logging and logger.debug("Audio send buffer full, dropping chunk.")

    def __recv_handler(self) -> None:
        while not self.__terminate.is_set():
            try:
                if not self.__socket.poll(100):
                    continue
                data = self.__socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            raw = bytes(data)
            if len(raw) < 8:
                continue
            pts_ns = struct.unpack(">q", raw[:8])[0]
            pcm = self.__decoder.decode(raw[8:])
            if pcm is not None:
                self.__queue.append((pcm, pts_ns))

    def recv(self) -> Optional[Tuple[NDArray, int]]:
        if not self.__receive_mode:
            raise ValueError("[AudioTransport:ERROR] :: recv() is only available in receive mode.")
        if self.__queue:
            return self.__queue.popleft()
        return None

    def signal_stop(self) -> None:
        if self.__terminate is not None:
            self.__terminate.set()

    def close(self) -> None:
        if self.__terminate is not None:
            self.__terminate.set()
        if self.__thread is not None:
            self.__thread.join(timeout=2)
            self.__thread = None
        if self.__socket is not None:
            try:
                self.__socket.setsockopt(zmq.LINGER, 0)
                self.__socket.close()
            except Exception:
                pass
            self.__socket = None
        if self.__encoder is not None:
            self.__encoder.close()
            self.__encoder = None
        if self.__decoder is not None:
            self.__decoder.close()
            self.__decoder = None
        self.__logging and logger.debug("AudioTransport closed.")
