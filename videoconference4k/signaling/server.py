import asyncio
import json
from typing import Any, Dict, Set

from ..utils.common import get_logger, import_dependency_safe

websockets = import_dependency_safe("websockets", error="silent")

logger = get_logger("SignalingServer")


class SignalingServer:

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_peers_per_room: int = 8,
        logging: bool = False,
    ):
        import_dependency_safe("websockets" if websockets is None else "", error="raise")
        self.__host = host
        self.__port = port
        self.__max_peers_per_room = max_peers_per_room
        self.__logging = logging if isinstance(logging, bool) else False
        self.__rooms: Dict[str, Set[Any]] = {}

    async def __broadcast(self, room: str, sender: Any, message: dict) -> None:
        payload = json.dumps(message)
        for peer in list(self.__rooms.get(room, ())):
            if peer is not sender:
                try:
                    await peer.send(payload)
                except Exception:
                    pass

    async def __handler(self, websocket, *args) -> None:
        room = None
        try:
            async for raw in websocket:
                try:
                    message = json.loads(raw)
                except (ValueError, TypeError):
                    continue

                msg_type = message.get("type")
                room = message.get("room", room)
                if not room:
                    continue

                if msg_type == "join":
                    members = self.__rooms.setdefault(room, set())
                    if len(members) >= self.__max_peers_per_room:
                        await websocket.send(json.dumps({
                            "type": "room_full",
                            "room": room,
                        }))
                        continue
                    peers_before = len(members)
                    members.add(websocket)
                    await websocket.send(json.dumps({
                        "type": "joined",
                        "room": room,
                        "peers": peers_before,
                    }))
                    await self.__broadcast(room, websocket, {
                        "type": "peer_joined",
                        "room": room,
                    })
                    self.__logging and logger.debug(
                        "Peer joined room '{}' ({} present).".format(room, peers_before + 1)
                    )
                else:
                    await self.__broadcast(room, websocket, message)
        finally:
            if room:
                members = self.__rooms.get(room)
                if members is not None:
                    members.discard(websocket)
                    await self.__broadcast(room, websocket, {
                        "type": "peer_left",
                        "room": room,
                    })
                    if not members:
                        self.__rooms.pop(room, None)
                    self.__logging and logger.debug("Peer left room '{}'.".format(room))

    async def serve_forever(self) -> None:
        self.__logging and logger.info(
            "Signaling server listening on ws://{}:{}".format(self.__host, self.__port)
        )
        async with websockets.serve(self.__handler, self.__host, self.__port):
            await asyncio.Future()

    def run(self) -> None:
        try:
            asyncio.run(self.serve_forever())
        except KeyboardInterrupt:
            logger.info("Signaling server stopped.")
