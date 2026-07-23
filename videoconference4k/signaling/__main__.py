import argparse

from .server import SignalingServer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m videoconference4k.signaling",
        description="Lightweight WebSocket signaling server for VideoConference4k peers.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--max-peers-per-room",
        type=int,
        default=8,
        help="Maximum peers allowed in one room (default: 8)",
    )
    parser.add_argument("--logging", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    server = SignalingServer(
        host=args.host,
        port=args.port,
        max_peers_per_room=args.max_peers_per_room,
        logging=args.logging,
    )
    server.run()


if __name__ == "__main__":
    main()
