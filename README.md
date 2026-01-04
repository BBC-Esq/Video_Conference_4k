# 🎥 VideoConference4k

> **Lightweight peer-to-peer video conferencing for Python — up to 4K60 streaming**

---

## ✨ Features

- 📹 **High-quality video streaming** — Support for up to 4K resolution at 60fps
- 🔗 **Peer-to-peer conferencing** — Direct P2P connections without a central server
- 👥 **Multi-peer support** — Connect with multiple participants simultaneously
- 🎙️ **Audio capture and playback** — Full-duplex audio communication
- 🌐 **Flexible networking** — Works over LAN or the internet via WebRTC
- 🔒 **Secure connections** — Optional ZMQ authentication with CurveZMQ
- 💻 **Cross-platform** — Works on Windows, macOS, and Linux

---

## 🚀 Installation
```bash
pip install git+https://github.com/BBC-Esq/Video_Conference_4k.git@main
```

---

## 📖 Quick Start

### Simple Peer-to-Peer Video Call

**Person A (creates the invite):**
```python
from videoconference4k import PeerConference

conference = PeerConference(resolution=(1920, 1080), framerate=30)

# Create and share this invite code with Person B
invite_code = conference.create_invite()
print(f"Share this code: {invite_code}")

# After receiving response from Person B
response_code = input("Enter response code: ")
conference.complete_connection(response_code)

# Wait for connection
if conference.wait_for_connection(timeout=30):
    print("Connected!")
    while conference.is_connected:
        local_frame = conference.get_local_frame()
        remote_frame = conference.get_remote_frame()
        # Display frames with OpenCV, etc.

conference.stop()
```

**Person B (joins with the invite):**
```python
from videoconference4k import PeerConference

conference = PeerConference(resolution=(1920, 1080), framerate=30)

invite_code = input("Enter invite code: ")
response_code = conference.accept_invite(invite_code)
print(f"Share this response: {response_code}")

# Connection starts automatically after accept_invite
if conference.wait_for_connection(timeout=30):
    print("Connected!")
    while conference.is_connected:
        local_frame = conference.get_local_frame()
        remote_frame = conference.get_remote_frame()
        # Display frames with OpenCV, etc.

conference.stop()
```

### Multi-Peer Conference
```python
from videoconference4k import MultiPeerConference

conference = MultiPeerConference(
    resolution=(1280, 720),
    framerate=30,
    max_peers=3
)

# Create invites for each peer
invite_alice = conference.create_invite_for_peer("Alice")
invite_bob = conference.create_invite_for_peer("Bob")

# Complete connections as responses come in
conference.complete_connection_with_peer("Alice", alice_response)
conference.complete_connection_with_peer("Bob", bob_response)

# Access frames from each peer
alice_frame = conference.get_peer_frame("Alice")
bob_frame = conference.get_peer_frame("Bob")

conference.stop()
```

---

## 🧩 Components

| Component | Description |
|-----------|-------------|
| `PeerConference` | Simple two-party video conferencing |
| `MultiPeerConference` | Multi-party video conferencing |
| `VideoCapture` | Capture video from cameras or files |
| `AudioCapture` | Capture and playback audio |
| `VideoStream` | High-level video streaming with resolution/framerate control |
| `RTCConnection` | Low-level WebRTC connection management |
| `SyncTransport` | Synchronous ZMQ-based video transport |
| `AsyncTransport` | Asynchronous ZMQ-based video transport |

---

## 📋 Requirements

- Python 3.10+

---

## 📄 License

Apache-2.0

---

## 👤 Author

**Blair Chintella** — vici0549@gmail.com

---

<p align="center">
  <i>Built with ❤️ for seamless video communication</i>
</p>
