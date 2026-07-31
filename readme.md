# 🎥 VideoConference4k

> **Lightweight peer-to-peer video conferencing for Python — up to 4K60 streaming**

---

## ✨ Features

- 📹 **High-quality video streaming** — up to 4K resolution at 60fps
- 🔗 **Peer-to-peer conferencing** — direct connections without a central server
- 👥 **Multi-peer support** — connect with several participants at once
- 🎙️ **Audio that keeps up** — Opus over its own path, a jitter buffer for uneven networks, and lip-sync that slews video to the audio clock
- 🔇 **Echo cancellation** — the other person stops hearing themselves when you are not on headphones
- 🎬 **Codec negotiation** — both ends agree on H.264, HEVC or AV1 per direction, in hardware where available
- 📶 **Adaptive bitrate** — the picture backs off under congestion and probes carefully on the way back up
- 🌐 **Flexible networking** — your own LAN, or the internet via WebRTC with room codes, TURN and optional UPnP
- 💻 **Cross-platform** — Windows, macOS and Linux
- 🚀 **Hardware encoding** — NVIDIA NVENC, Intel Quick Sync, or CPU, whichever the machine has

---

## 🚀 Installation

**Standard installation:**
```bash
pip install git+https://github.com/BBC-Esq/Video_Conference_4k.git@main
```

**ffmpeg** — needed for Intel Quick Sync, for CPU encoding, and for software
decoding. Without it on your PATH, only NVIDIA encoding and motion-JPEG are
available, so a machine with no NVIDIA card quietly falls back to JPEG:

```bash
winget install Gyan.FFmpeg
```

On macOS use `brew install ffmpeg`; on Linux install `ffmpeg` from your package
manager. `examples/two_machine_call.py --preflight` will tell you what your
build of ffmpeg can actually do.

**With NVIDIA GPU acceleration (requires an NVIDIA GPU and a recent driver):**
```bash
pip install "videoconference4k[gpu] @ git+https://github.com/BBC-Esq/Video_Conference_4k.git@main"
```

**With room-code signaling for calls over the internet:**
```bash
pip install "videoconference4k[signaling] @ git+https://github.com/BBC-Esq/Video_Conference_4k.git@main"
```

**Better echo cancellation (optional):**
```bash
pip install livekit
```

---

## 📖 Quick Start

### 4K Call Over Your Own Network

`DirectConference` is the path this library is built around: hardware encoding,
per-direction codec negotiation, a jitter buffer and lip-sync. Both machines run
the same script, each pointing at the other.

```python
from videoconference4k import DirectConference

conference = DirectConference(
    peer_address="192.168.1.42",          # the other machine
    resolution=(3840, 2160),
    framerate=30,
    gpu_accelerated=True,
    codec_priority=("h264", "hevc", "av1"),
    enable_audio=True,
)

conference.start()

while conference.is_running:
    local_frame = conference.get_local_frame()
    remote_frame = conference.get_remote_frame()
    # hand them to whatever you are drawing with

conference.stop()
```

Ask it what it settled on at any point:

```python
s = conference.stats()
print(s["send_codec"], s["send_impl"])   # e.g. h264 nvenc
print(s["recv_codec"], s["recv_impl"])   # what is arriving, and what decodes it
```

`examples/two_machine_call.py` wraps this in a window with a preflight check
(`--preflight`) that reports what your camera, GPU and ffmpeg build can do.

### Echo Cancellation

Without headphones, your microphone picks up the other person's voice from your
speakers and sends it back. `DirectConference` cancels that automatically,
choosing the best canceller present on the machine:

| backend | echo removed | needs |
|---|---|---|
| `localvqe` | ~58 dB | a compiled library, pointed at by `VIDEOCONFERENCE4K_LOCALVQE` |
| `webrtc` | ~27 dB | `pip install livekit` |
| `numpy` | ~13 dB | nothing, always present |

Measured on identical signals with the echo arriving 165 ms late and a second
person talking over it. The first two run at 16 kHz, so they do not cancel above
8 kHz; the third is the gentlest on your own voice when both people speak at
once. Pick one explicitly if you would rather not have it chosen for you:

```python
conference = DirectConference(peer_address="192.168.1.42", echo_backend="webrtc")
```

`examples/two_machine_call.py --preflight` prints which of the three this
machine can use.

### Peer-to-Peer Call Over the Internet

`PeerConference` uses WebRTC, so it traverses NAT without port forwarding.

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
from videoconference4k.codec import has_nvidia_codec, get_nvidia_info

if has_nvidia_codec():
    print("NVIDIA hardware encoding available!")
    print(get_nvidia_info())
else:
    print("GPU acceleration not available, using CPU encoding")
```

---

## 🛠️ Checking a machine before you call

`examples/two_machine_call.py` is a working call in a window, and carries the
diagnostics worth running first:

```bash
python examples/two_machine_call.py --preflight
```

Reports your camera's real modes, which encoders and decoders actually run here
(by running them, not by reading a list), your audio devices, and which echo
canceller is available.

```bash
python examples/two_machine_call.py --loopback a
```

Calls this machine from itself, so the whole pipeline can be checked without a
second computer. Run `--loopback b` in another terminal for the far end.

```bash
python examples/two_machine_call.py 192.168.1.42 --preset 1080p60
```

The call itself. Presets run from `safe` (720p30, works between any two
machines) up to `4k30`.

---

## 🧩 Components

| Component | Description |
|-----------|-------------|
| `DirectConference` | Two-party calling over your own network, up to 4K with hardware codecs |
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
- **ffmpeg on your PATH** — required for Intel Quick Sync, CPU encoding and
  software decoding; without it the encoder ladder falls through to motion-JPEG
- For NVIDIA hardware encoding: an NVENC-capable GPU and a driver new enough for
  CUDA 12 (527.41 or later on Windows)
- A camera and, for calls, a microphone and speakers or a headset

Security note: CurveZMQ authentication exists on `SyncTransport` and is set
through its options, but `DirectConference` does not currently expose it and the
audio transport has none, so calls on the direct path are unencrypted. Treat it
as a local-network feature until that is wired through.

---

## 📄 License

Apache-2.0

---

## 👤 Author

**Blair Chintella** — vici0549@gmail.com
