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
- 🚀 **GPU Acceleration** — Optional NVIDIA hardware encoding via PyNvVideoCodec

---

## 🚀 Installation

**Standard installation:**
```bash
pip install git+https://github.com/BBC-Esq/Video_Conference_4k.git@main
```

**With NVIDIA GPU acceleration (requires NVIDIA GPU and CUDA):**
```bash
pip install PyNvVideoCodec
```

---

## 📖 Quick Start

### Simple Peer-to-Peer Video Call

**Person A (creates the invite):**
```python
from videoconference4k import PeerConference

conference = PeerConference(resolution=(1920, 1080), framerate=30)

invite_code = conference.create_invite()
print(f"Share this code: {invite_code}")

response_code = input("Enter response code: ")
conference.complete_connection(response_code)

if conference.wait_for_connection(timeout=30):
    print("Connected!")
    while conference.is_connected:
        local_frame = conference.get_local_frame()
        remote_frame = conference.get_remote_frame()

conference.stop()
```

**Person B (joins with the invite):**
```python
from videoconference4k import PeerConference

conference = PeerConference(resolution=(1920, 1080), framerate=30)

invite_code = input("Enter invite code: ")
response_code = conference.accept_invite(invite_code)
print(f"Share this response: {response_code}")

if conference.wait_for_connection(timeout=30):
    print("Connected!")
    while conference.is_connected:
        local_frame = conference.get_local_frame()
        remote_frame = conference.get_remote_frame()

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

invite_alice = conference.create_invite_for_peer("Alice")
invite_bob = conference.create_invite_for_peer("Bob")

conference.complete_connection_with_peer("Alice", alice_response)
conference.complete_connection_with_peer("Bob", bob_response)

alice_frame = conference.get_peer_frame("Alice")
bob_frame = conference.get_peer_frame("Bob")

conference.stop()
```

---

## 🚀 GPU-Accelerated Transport (ZMQ)

For high-performance LAN streaming with NVIDIA hardware encoding:

**Sender:**
```python
from videoconference4k import SyncTransport, VideoCapture

capture = VideoCapture(source=0).start()

transport = SyncTransport(
    address="192.168.1.100",
    port="5555",
    gpu_accelerated=True,
    gpu_codec="h264",
    gpu_bitrate=8000000,
    logging=True,
)

while True:
    frame = capture.read()
    if frame is None:
        break
    transport.send(frame)

transport.close()
capture.stop()
```

**Receiver:**
```python
from videoconference4k import SyncTransport
import cv2

transport = SyncTransport(
    address="*",
    port="5555",
    receive_mode=True,
    gpu_accelerated=True,
    logging=True,
)

while True:
    frame = transport.recv()
    if frame is None:
        break
    cv2.imshow("Received", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

transport.close()
cv2.destroyAllWindows()
```

### GPU Acceleration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_accelerated` | bool | False | Enable NVIDIA hardware encoding |
| `gpu_id` | int | 0 | GPU device ID for multi-GPU systems |
| `gpu_codec` | str | "h264" | Video codec (h264, hevc) |
| `gpu_bitrate` | int | 8000000 | Target bitrate in bits per second |

### Check GPU Availability
```python
from videoconference4k.utils import has_nvidia_codec, get_nvidia_info

if has_nvidia_codec():
    print("NVIDIA hardware encoding available!")
    print(get_nvidia_info())
else:
    print("GPU acceleration not available, using CPU encoding")
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
| `SyncTransport` | Synchronous ZMQ-based video transport (supports GPU) |
| `AsyncTransport` | Asynchronous ZMQ-based video transport (supports GPU) |

---

## 📋 Requirements

- Python 3.10+
- For GPU acceleration: NVIDIA GPU with NVENC support + CUDA Toolkit

---

## 📄 License

Apache-2.0

---

## 👤 Author

**Blair Chintella** — vici0549@gmail.com
