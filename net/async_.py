import numpy as np
import asyncio
import inspect
import logging as log
import string
import secrets
import platform
import threading
from typing import Any, Tuple, AsyncGenerator, Union, TypeVar
from numpy.typing import NDArray

from ..utils.common import logger_handler, import_dependency_safe, log_version
from ..stream.video import VideoStream

zmq = import_dependency_safe("zmq", pkg_name="pyzmq", error="silent", min_version="4.0")
if not (zmq is None):
    import zmq.asyncio
msgpack = import_dependency_safe("msgpack", error="silent")
m = import_dependency_safe("msgpack_numpy", error="silent")
uvloop = import_dependency_safe("uvloop", error="silent")

logger = log.getLogger("AsyncTransport")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

T = TypeVar("T", bound="AsyncTransport")


class AsyncTransport:

    def __init__(
        self,
        address: str = None,
        port: str = None,
        protocol: str = "tcp",
        pattern: int = 0,
        receive_mode: bool = False,
        timeout: Union[int, float] = 0.0,
        source: Any = None,
        backend: int = 0,
        colorspace: str = None,
        resolution: Tuple[int, int] = (640, 480),
        framerate: Union[int, float] = 25,
        time_delay: int = 0,
        logging: bool = False,
        **options: dict
    ):
        self.__logging = logging if isinstance(logging, bool) else False

        log_version(logging=self.__logging)

        import_dependency_safe(
            "zmq" if zmq is None else "", min_version="4.0", pkg_name="pyzmq"
        )
        import_dependency_safe("msgpack" if msgpack is None else "")
        import_dependency_safe("msgpack_numpy" if m is None else "")

        valid_messaging_patterns = {
            0: (zmq.PAIR, zmq.PAIR),
            1: (zmq.REQ, zmq.REP),
            2: (zmq.PUB, zmq.SUB),
            3: (zmq.PUSH, zmq.PULL),
        }

        if isinstance(pattern, int) and pattern in valid_messaging_patterns:
            self.__msg_pattern = pattern
            self.__pattern = valid_messaging_patterns[pattern]
        else:
            self.__msg_pattern = 0
            self.__pattern = valid_messaging_patterns[self.__msg_pattern]
            self.__logging and logger.warning(
                "Invalid pattern {pattern}. Defaulting to `zmq.PAIR`!".format(
                    pattern=pattern
                )
            )

        if isinstance(protocol, str) and protocol in ["tcp", "ipc"]:
            self.__protocol = protocol
        else:
            self.__protocol = "tcp"
            self.__logging and logger.warning("Invalid protocol. Defaulting to `tcp`!")

        self.__terminate = False
        self.__receive_mode = receive_mode
        self.__stream = None
        self.__msg_socket = None
        self.config = {}
        self.__queue = None
        self.__bi_mode = False

        if timeout and isinstance(timeout, (int, float)):
            self.__timeout = float(timeout)
        else:
            self.__timeout = 15.0

        self.__id = "".join(
            secrets.choice(string.ascii_uppercase + string.digits) for i in range(8)
        )

        options = {str(k).strip(): v for k, v in options.items()}

        if "bidirectional_mode" in options:
            value = options["bidirectional_mode"]
            if isinstance(value, bool) and pattern < 2 and source is None:
                self.__bi_mode = value
            else:
                self.__bi_mode = False
                logger.warning("Bidirectional data transmission is disabled!")
            if pattern >= 2:
                raise ValueError(
                    "[AsyncTransport:ERROR] :: `{}` pattern is not valid when Bidirectional Mode is enabled. Kindly refer Docs for more Information!".format(
                        pattern
                    )
                )
            elif not (source is None):
                raise ValueError(
                    "[AsyncTransport:ERROR] :: Custom source must be None when Bidirectional Mode is enabled. Kindly refer Docs for more Information!"
                )
            elif isinstance(value, bool) and self.__logging:
                logger.debug(
                    "Bidirectional Data Transmission is {} for this connection!".format(
                        "enabled" if value else "disabled"
                    )
                )
            else:
                logger.error("`bidirectional_mode` value is invalid!")
            del options["bidirectional_mode"]

        self.__loop_thread = None
        self.__owns_loop = False

        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        else:
            if not (uvloop is None):
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            else:
                import_dependency_safe("uvloop", error="log")

        try:
            self.loop = asyncio.get_running_loop()
            self.__owns_loop = False
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            self.__owns_loop = True
            self.__loop_thread = threading.Thread(target=self.__run_loop, daemon=True)
            self.__loop_thread.start()
            self.__logging and logger.debug("Started background event loop thread.")

        self.__logging and logger.info(
            "Using ``{}`` event loop for this process.".format(
                self.loop.__class__.__name__
            )
        )

        self.__msg_context = zmq.asyncio.Context()

        if receive_mode:
            if address is None:
                self.__address = "*"
            else:
                self.__address = address
            if port is None:
                self.__port = "5555"
            else:
                self.__port = port
        else:
            if source is None:
                self.config = {"generator": None}
                self.__logging and logger.warning("Given source is of NoneType!")
            else:
                self.__stream = VideoStream(
                    source=source,
                    backend=backend,
                    colorspace=colorspace,
                    resolution=resolution,
                    framerate=framerate,
                    logging=logging,
                    time_delay=time_delay,
                    **options
                )
                self.config = {"generator": self.__frame_generator()}
            if address is None:
                self.__address = "localhost"
            else:
                self.__address = address
            if port is None:
                self.__port = "5555"
            else:
                self.__port = port
            self.task = None

        self.__queue = asyncio.Queue() if self.__bi_mode else None

    def __run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def launch(self) -> T:
        if self.__receive_mode:
            self.__logging and logger.debug(
                "Launching AsyncTransport asynchronous generator!"
            )
        else:
            self.__logging and logger.debug(
                "Creating AsyncTransport asynchronous server handler!"
            )
            if self.__owns_loop:
                self.task = asyncio.run_coroutine_threadsafe(
                    self.__server_handler(), self.loop
                )
            else:
                self.task = self.loop.create_task(self.__server_handler())
        return self

    async def __server_handler(self):
        if isinstance(self.config, dict) and "generator" in self.config:
            if self.config["generator"] is None or not inspect.isasyncgen(
                self.config["generator"]
            ):
                raise ValueError(
                    "[AsyncTransport:ERROR] :: Invalid configuration. Assigned generator must be a asynchronous generator function/method only!"
                )
        else:
            raise RuntimeError(
                "[AsyncTransport:ERROR] :: Assigned AsyncTransport configuration is invalid!"
            )

        self.__msg_socket = self.__msg_context.socket(self.__pattern[0])

        if self.__msg_pattern == 1:
            self.__msg_socket.REQ_RELAXED = True
            self.__msg_socket.REQ_CORRELATE = True

        if self.__msg_pattern == 2:
            self.__msg_socket.set_hwm(1)

        try:
            self.__msg_socket.connect(
                self.__protocol + "://" + str(self.__address) + ":" + str(self.__port)
            )
            self.__logging and logger.debug(
                "Successfully connected to address: {} with pattern: {}.".format(
                    (
                        self.__protocol
                        + "://"
                        + str(self.__address)
                        + ":"
                        + str(self.__port)
                    ),
                    self.__msg_pattern,
                )
            )
            logger.critical(
                "Send Mode is successfully activated and ready to send data!"
            )
        except Exception as e:
            logger.exception(str(e))
            if self.__bi_mode:
                logger.error(
                    "Failed to activate Bidirectional Mode for this connection!"
                )
            raise ValueError(
                "[AsyncTransport:ERROR] :: Failed to connect address: {} and pattern: {}!".format(
                    (
                        self.__protocol
                        + "://"
                        + str(self.__address)
                        + ":"
                        + str(self.__port)
                    ),
                    self.__msg_pattern,
                )
            )

        async for dataframe in self.config["generator"]:
            if self.__bi_mode and len(dataframe) == 2:
                (data, frame) = dataframe
                if not (data is None) and isinstance(data, np.ndarray):
                    logger.warning(
                        "Skipped unsupported `data` of datatype: {}!".format(
                            type(data).__name__
                        )
                    )
                    data = None
                assert isinstance(
                    frame, np.ndarray
                ), "[AsyncTransport:ERROR] :: Invalid data received from server end!"
            elif self.__bi_mode:
                raise ValueError(
                    "[AsyncTransport:ERROR] :: Send Mode only accepts tuple(data, frame) as input in Bidirectional Mode. \
                    Kindly refer VideoConference4k docs!"
                )
            else:
                frame = np.copy(dataframe)
                data = None

            if not (frame.flags["C_CONTIGUOUS"]):
                frame = np.ascontiguousarray(frame, dtype=frame.dtype)

            data_dict = dict(
                terminate=False,
                bi_mode=self.__bi_mode,
                data=data if not (data is None) else "",
            )
            data_enc = msgpack.packb(data_dict)
            await self.__msg_socket.send(data_enc, flags=zmq.SNDMORE)

            frame_enc = msgpack.packb(frame, default=m.encode)
            await self.__msg_socket.send_multipart([frame_enc])

            if self.__msg_pattern < 2:
                if self.__bi_mode:
                    recvdmsg_encoded = await asyncio.wait_for(
                        self.__msg_socket.recv(), timeout=self.__timeout
                    )
                    recvd_data = msgpack.unpackb(recvdmsg_encoded, use_list=False)
                    if recvd_data.get("return_type") == "ndarray":
                        recvdframe_encoded = await asyncio.wait_for(
                            self.__msg_socket.recv_multipart(), timeout=self.__timeout
                        )
                        await self.__queue.put(
                            msgpack.unpackb(
                                recvdframe_encoded[0],
                                use_list=False,
                                object_hook=m.decode,
                            )
                        )
                    else:
                        await self.__queue.put(
                            recvd_data.get("return_data")
                            if recvd_data.get("return_data")
                            else None
                        )
                else:
                    recv_confirmation = await asyncio.wait_for(
                        self.__msg_socket.recv(), timeout=self.__timeout
                    )
                    self.__logging and logger.debug(recv_confirmation)

    async def recv_generator(self) -> AsyncGenerator[Tuple[Any, NDArray], NDArray]:
        if not (self.__receive_mode):
            self.__terminate = True
            raise ValueError(
                "[AsyncTransport:ERROR] :: `recv_generator()` function cannot be accessed while `receive_mode` is disabled. Kindly refer VideoConference4k docs!"
            )

        self.__msg_socket = self.__msg_context.socket(self.__pattern[1])

        if self.__msg_pattern == 2:
            self.__msg_socket.set_hwm(1)
            self.__msg_socket.setsockopt(zmq.SUBSCRIBE, b"")

        try:
            self.__msg_socket.bind(
                self.__protocol + "://" + str(self.__address) + ":" + str(self.__port)
            )
            self.__logging and logger.debug(
                "Successfully binded to address: {} with pattern: {}.".format(
                    (
                        self.__protocol
                        + "://"
                        + str(self.__address)
                        + ":"
                        + str(self.__port)
                    ),
                    self.__msg_pattern,
                )
            )
            logger.critical("Receive Mode is activated successfully!")
        except Exception as e:
            logger.exception(str(e))
            raise RuntimeError(
                "[AsyncTransport:ERROR] :: Failed to bind address: {} and pattern: {}{}!".format(
                    (
                        self.__protocol
                        + "://"
                        + str(self.__address)
                        + ":"
                        + str(self.__port)
                    ),
                    self.__msg_pattern,
                    " and Bidirectional Mode enabled" if self.__bi_mode else "",
                )
            )

        while not self.__terminate:
            datamsg_encoded = await asyncio.wait_for(
                self.__msg_socket.recv(), timeout=self.__timeout
            )
            data = msgpack.unpackb(datamsg_encoded, use_list=False)
            if data.get("terminate", False):
                if self.__msg_pattern < 2:
                    return_dict = dict(
                        terminated="Client-`{}` successfully terminated!".format(
                            self.__id
                        ),
                    )
                    retdata_enc = msgpack.packb(return_dict)
                    await self.__msg_socket.send(retdata_enc)
                self.__logging and logger.info(
                    "Termination signal received from server!"
                )
                self.__terminate = True
                break
            framemsg_encoded = await asyncio.wait_for(
                self.__msg_socket.recv_multipart(), timeout=self.__timeout
            )
            frame = msgpack.unpackb(
                framemsg_encoded[0], use_list=False, object_hook=m.decode
            )

            if self.__msg_pattern < 2:
                if self.__bi_mode and data.get("bi_mode", False):
                    if not self.__queue.empty():
                        return_data = await self.__queue.get()
                        self.__queue.task_done()
                    else:
                        return_data = None
                    if not (return_data is None) and isinstance(
                        return_data, np.ndarray
                    ):
                        if not (return_data.flags["C_CONTIGUOUS"]):
                            return_data = np.ascontiguousarray(
                                return_data, dtype=return_data.dtype
                            )

                        rettype_dict = dict(
                            return_type=(type(return_data).__name__),
                            return_data=None,
                        )
                        rettype_enc = msgpack.packb(rettype_dict)
                        await self.__msg_socket.send(rettype_enc, flags=zmq.SNDMORE)

                        retframe_enc = msgpack.packb(return_data, default=m.encode)
                        await self.__msg_socket.send_multipart([retframe_enc])
                    else:
                        return_dict = dict(
                            return_type=(type(return_data).__name__),
                            return_data=(
                                return_data if not (return_data is None) else ""
                            ),
                        )
                        retdata_enc = msgpack.packb(return_dict)
                        await self.__msg_socket.send(retdata_enc)
                elif self.__bi_mode or data.get("bi_mode", False):
                    raise RuntimeError(
                        "[AsyncTransport:ERROR] :: Invalid configuration! Bidirectional Mode is not activate on {} end.".format(
                            "client" if self.__bi_mode else "server"
                        )
                    )
                else:
                    await self.__msg_socket.send(
                        bytes(
                            "Data received on client: {} !".format(self.__id), "utf-8"
                        )
                    )
            if self.__bi_mode:
                yield (data.get("data"), frame) if data.get("data") else (None, frame)
            else:
                yield frame
            await asyncio.sleep(0)

    async def __frame_generator(self):
        self.__stream.start()
        while not self.__terminate:
            frame = self.__stream.read()
            if frame is None:
                break
            yield frame
            await asyncio.sleep(0)

    async def transceive_data(self, data: Any = None) -> Any:
        recvd_data = None
        if not self.__terminate:
            if self.__bi_mode:
                if self.__receive_mode:
                    await self.__queue.put(data)
                else:
                    if not self.__queue.empty():
                        recvd_data = await self.__queue.get()
                        self.__queue.task_done()
            else:
                logger.error(
                    "`transceive_data()` function cannot be used when Bidirectional Mode is disabled."
                )
        return recvd_data

    async def __terminate_connection(self, disable_confirmation=False):
        self.__logging and logger.debug(
            "Terminating various {} Processes. Please wait.".format(
                "Receive Mode" if self.__receive_mode else "Send Mode"
            )
        )

        if self.__receive_mode:
            self.__terminate = True
        else:
            self.__terminate = True
            if not (self.__stream is None):
                self.__stream.stop()
            data_dict = dict(terminate=True)
            data_enc = msgpack.packb(data_dict)
            await self.__msg_socket.send(data_enc)
            if self.__msg_pattern < 2 and not disable_confirmation:
                recv_confirmation = await self.__msg_socket.recv()
                recvd_conf = msgpack.unpackb(recv_confirmation, use_list=False)
                self.__logging and "terminated" in recvd_conf and logger.debug(
                    recvd_conf["terminated"]
                )
        self.__msg_socket.setsockopt(zmq.LINGER, 0)
        self.__msg_socket.close()
        if self.__bi_mode:
            while not self.__queue.empty():
                try:
                    self.__queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue
                self.__queue.task_done()
            await self.__queue.join()

        logger.critical(
            "{} successfully terminated!".format(
                "Receive Mode" if self.__receive_mode else "Send Mode"
            )
        )

    def close(self, skip_loop: bool = False) -> None:
        if not (skip_loop):
            if self.__owns_loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.__terminate_connection(), self.loop
                )
                try:
                    future.result(timeout=10)
                except Exception as e:
                    logger.error("Error during close: {}".format(e))
                self.loop.call_soon_threadsafe(self.loop.stop)
                if self.__loop_thread is not None:
                    self.__loop_thread.join(timeout=2)
            else:
                if self.loop is not None and self.loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self.__terminate_connection(), self.loop
                    )
                    try:
                        future.result(timeout=10)
                    except Exception as e:
                        logger.error("Error during close: {}".format(e))
                else:
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(self.__terminate_connection())
                        new_loop.close()
                    except Exception as e:
                        logger.error("Error during close: {}".format(e))
        else:
            if self.loop is not None and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.__terminate_connection(disable_confirmation=True),
                    self.loop
                )
            else:
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(
                        self.__terminate_connection(disable_confirmation=True)
                    )
                    new_loop.close()
                except Exception as e:
                    logger.error("Error during async close: {}".format(e))