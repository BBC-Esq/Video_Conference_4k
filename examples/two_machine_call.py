"""Two-machine LAN call test for DirectConference.

Run --preflight on both machines first, then start the call on both with each
pointed at the other's LAN address.

    python examples/two_machine_call.py --preflight
    python examples/two_machine_call.py 192.168.1.42
    python examples/two_machine_call.py 192.168.1.42 --preset hd60
"""

import argparse
import socket
import sys
import time

import numpy as np

PRESETS = {
    "safe": dict(resolution=(1280, 720), framerate=30, gpu_accelerated=False,
                 gpu_bitrate=8_000_000,
                 note="720p30 JPEG - works between any two machines, proves the link"),
    "hd60": dict(resolution=(1920, 1080), framerate=60, gpu_accelerated=True,
                 gpu_bitrate=12_000_000,
                 note="1080p60 hardware H264 - needs a capable camera on both ends"),
    "uhd30": dict(resolution=(3840, 2160), framerate=30, gpu_accelerated=True,
                  gpu_bitrate=25_000_000,
                  note="4K30 hardware H264 - the headline path"),
}


def lan_address():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def preflight(args):
    from videoconference4k.capture import probe_camera, AudioCapture
    from videoconference4k.codec import get_available_codecs

    print("=" * 68)
    print("PREFLIGHT  host {}  address {}".format(socket.gethostname(), lan_address()))
    print("=" * 68)

    print("\nGive the other machine this address: {}".format(lan_address()))

    codecs = get_available_codecs()
    print("\nCodecs")
    for name, ok in codecs.items():
        print("  {:<16} {}".format(name, "yes" if ok else "no"))
    if not codecs.get("nvidia"):
        print("  NOTE: no NVENC here. Both machines must agree on --preset;")
        print("        a machine without NVENC cannot decode a hardware H264 stream.")

    print("\nAudio devices")
    try:
        devices = AudioCapture.get_devices()
        for kind in ("input", "output"):
            entries = devices.get(kind, [])
            print("  {}: {}".format(kind, len(entries)))
            for dev in entries[:3]:
                print("    [{}] {}".format(dev["index"], dev["name"][:52]))
        if not devices.get("input"):
            print("  WARNING: no microphone found - run the call with --no-audio")
    except Exception as exc:
        print("  audio probe failed: {}".format(exc))

    print("\nCamera (measured, not advertised)")
    presets = [(1280, 720, 30), (1920, 1080, 60), (3840, 2160, 30)]
    try:
        for entry in probe_camera(source=args.camera, presets=presets, sample=20):
            req_w, req_h, req_fps = entry["requested"]
            if not entry["opened"]:
                print("  {}x{}@{:<3} could not open camera".format(req_w, req_h, req_fps))
                continue
            print("  {}x{}@{:<3} -> delivered {} at {:.1f} fps  "
                  "(worst frame gap {:.0f} ms, p99 {:.0f} ms, {})".format(
                      req_w, req_h, req_fps, entry["delivered"], entry["measured_fps"],
                      entry["worst_interval_ms"], entry["p99_interval_ms"],
                      entry["fourcc"] or "n/a"))
    except Exception as exc:
        print("  camera probe failed: {}".format(exc))

    print("\nPorts {} and {} must accept inbound TCP.".format(args.video_port, args.audio_port))
    print("If no video appears during the call, run this once as Administrator:")
    print('  netsh advfirewall firewall add rule name="VideoConference4k" '
          'dir=in action=allow protocol=TCP localport={},{}'.format(args.video_port, args.audio_port))
    print("\nPresets")
    for name, cfg in PRESETS.items():
        print("  {:<6} {}".format(name, cfg["note"]))


def portcheck(args):
    for label, port in (("video", args.video_port), ("audio", args.audio_port)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        try:
            sock.connect((args.peer, int(port)))
            print("  {} port {} on {}: REACHABLE".format(label, port, args.peer))
        except Exception as exc:
            print("  {} port {} on {}: BLOCKED ({})".format(label, port, args.peer, exc))
            print("      the peer must already be running the call, and its firewall must allow this port")
        finally:
            sock.close()


def run_call(args):
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
    )
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtCore import Qt, QTimer
    import cv2

    from videoconference4k import DirectConference

    cfg = dict(PRESETS[args.preset])
    cfg.pop("note")
    if args.no_gpu:
        cfg["gpu_accelerated"] = False
    if args.fixed_bitrate:
        cfg["adaptive_bitrate"] = False

    class CallWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("VideoConference4k  {}  ->  {}".format(lan_address(), args.peer))
            self.resize(1280, 620)

            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            videos = QHBoxLayout()
            self.local_label = self._panel(videos, "Local")
            self.remote_label = self._panel(videos, "Remote")
            layout.addLayout(videos)

            self.stats_label = QLabel("starting...")
            self.stats_label.setStyleSheet(
                "font-family: Consolas, monospace; font-size: 12px; padding: 8px;")
            self.stats_label.setTextFormat(Qt.PlainText)
            layout.addWidget(self.stats_label)

            self.conf = DirectConference(
                peer_address=args.peer,
                video_port=str(args.video_port),
                audio_port=str(args.audio_port),
                camera_id=args.camera,
                enable_audio=not args.no_audio,
                logging=args.verbose,
                **cfg
            )
            self.conf.start()

            self.started = time.perf_counter()
            self.last_remote = None
            self.remote_frames = 0
            self.prev_remote_frames = 0
            self.prev_sent = 0
            self.prev_stats_time = self.started
            self.first_remote_at = None

            self.video_timer = QTimer(self)
            self.video_timer.timeout.connect(self.tick_video)
            self.video_timer.start(8)

            self.stats_timer = QTimer(self)
            self.stats_timer.timeout.connect(self.tick_stats)
            self.stats_timer.start(500)

        def _panel(self, parent, title):
            column = QVBoxLayout()
            caption = QLabel(title)
            caption.setAlignment(Qt.AlignCenter)
            view = QLabel()
            view.setMinimumSize(620, 400)
            view.setAlignment(Qt.AlignCenter)
            view.setStyleSheet("background-color: #101010; border: 1px solid #303030;")
            column.addWidget(caption)
            column.addWidget(view)
            parent.addLayout(column)
            return view

        def tick_video(self):
            local = self.conf.get_local_frame()
            if local is not None:
                self.show_frame(local, self.local_label)
            remote = self.conf.get_remote_frame()
            if remote is not None and remote is not self.last_remote:
                self.last_remote = remote
                self.remote_frames += 1
                if self.first_remote_at is None:
                    self.first_remote_at = time.perf_counter() - self.started
                self.show_frame(remote, self.remote_label)

        def show_frame(self, frame, label):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(image).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        def tick_stats(self):
            stats = self.conf.stats()
            now = time.perf_counter()
            window = max(1e-6, now - self.prev_stats_time)
            recv_fps = (self.remote_frames - self.prev_remote_frames) / window
            send_fps = (stats["frames_sent"] - self.prev_sent) / window
            self.prev_stats_time = now
            self.prev_remote_frames = self.remote_frames
            self.prev_sent = stats["frames_sent"]

            if self.first_remote_at is None:
                first = "waiting for peer..."
            else:
                first = "{:.1f} s after start".format(self.first_remote_at)

            lines = [
                "preset {}   {}x{}@{}   {}   audio {}".format(
                    args.preset, cfg["resolution"][0], cfg["resolution"][1], cfg["framerate"],
                    "hardware" if cfg["gpu_accelerated"] else "jpeg",
                    "off" if args.no_audio else "on"),
                "send  {:6.1f} fps   {:8.0f} kbps   target {:5.1f} Mbps   first remote frame: {}".format(
                    send_fps, stats["send_kbps"], stats["target_bitrate"] / 1e6, first),
                "recv  {:6.1f} fps   hold {:3d}   lipsync {}   camera {}".format(
                    recv_fps, stats["video_hold_depth"],
                    "on" if stats["lipsync"] else "off",
                    "FAILED" if stats["capture_failed"] else "ok"),
                "loss  pipe {:5d}   shed lvl {} ({} frames)   lagged {:5d}   "
                "late/stale {:5d}   dup {:5d}   reconnects {}".format(
                    stats["pipe_dropped"], stats["shed_level"], stats["frames_source_shed"],
                    stats["frames_lagged"], stats["frames_dropped"], stats["frames_skipped"],
                    stats["reconnects"]),
            ]
            self.stats_label.setText("\n".join(lines))

        def closeEvent(self, event):
            self.video_timer.stop()
            self.stats_timer.stop()
            self.conf.stop()
            event.accept()

    print("Starting call to {} on ports {}/{} using preset '{}'".format(
        args.peer, args.video_port, args.audio_port, args.preset))
    print("Use headphones on both machines or the microphones will echo.")
    app = QApplication(sys.argv)
    window = CallWindow()
    window.show()
    sys.exit(app.exec())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("peer", nargs="?", help="LAN address of the other machine")
    parser.add_argument("--preflight", action="store_true", help="check this machine and exit")
    parser.add_argument("--portcheck", action="store_true",
                        help="test whether the peer's ports are reachable (peer must be running)")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="safe")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video-port", default="5555")
    parser.add_argument("--audio-port", default="5556")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--fixed-bitrate", action="store_true",
                        help="disable adaptive bitrate, to tell encoder limits from link limits")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        preflight(args)
        return
    if not args.peer:
        parser.error("peer address is required (or use --preflight)")
    if args.portcheck:
        portcheck(args)
        return
    run_call(args)


if __name__ == "__main__":
    main()
