"""Two-machine LAN call test for DirectConference.

Run --preflight on both machines first, then start the call on both with each
pointed at the other's LAN address.

    python examples/two_machine_call.py --preflight
    python examples/two_machine_call.py 192.168.1.42
    python examples/two_machine_call.py 192.168.1.42 --preset hd60
"""

import argparse
import re
import socket
import sys
import time

import numpy as np

PRESETS = {
    "safe": dict(resolution=(1280, 720), framerate=30, gpu_accelerated=False,
                 gpu_bitrate=4_000_000,
                 note="720p30 JPEG - works between any two machines, proves the link"),
    "720p30": dict(resolution=(1280, 720), framerate=30, gpu_accelerated=True,
                   gpu_bitrate=4_000_000,
                   note="720p30 hardware H264"),
    "720p60": dict(resolution=(1280, 720), framerate=60, gpu_accelerated=True,
                   gpu_bitrate=6_000_000,
                   note="720p60 hardware H264 - smoothest motion at modest bandwidth"),
    "1080p30": dict(resolution=(1920, 1080), framerate=30, gpu_accelerated=True,
                    gpu_bitrate=8_000_000,
                    note="1080p30 hardware H264"),
    "1080p60": dict(resolution=(1920, 1080), framerate=60, gpu_accelerated=True,
                    gpu_bitrate=12_000_000,
                    note="1080p60 hardware H264"),
    "4k30": dict(resolution=(3840, 2160), framerate=30, gpu_accelerated=True,
                 gpu_bitrate=25_000_000,
                 note="4K30 hardware H264 - the headline path"),
}

PRESET_ALIASES = {"hd60": "1080p60", "uhd30": "4k30"}


def lan_address():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def nvenc_diagnosis():
    """Explain why hardware encoding is unavailable instead of only reporting that it is."""
    import os
    from pathlib import Path

    lines = []
    venv = Path(sys.executable).parent.parent
    nvidia_dir = venv / "Lib" / "site-packages" / "nvidia"
    needed = ("cuda_runtime", "cublas", "cuda_nvrtc")
    missing = [n for n in needed if not (nvidia_dir / n / "bin").exists()]

    install = ('pip install PyNvVideoCodec nvidia-cuda-runtime-cu12 '
               'nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12')

    try:
        import PyNvVideoCodec as nvc
    except Exception as exc:
        lines.append("PyNvVideoCodec is not importable in this environment:")
        lines.append("  {}".format(exc))
        lines.append("Install the hardware packages INTO THIS venv, then re-run preflight:")
        lines.append("  " + install)
        return lines

    if missing:
        lines.append("PyNvVideoCodec is installed, but these CUDA runtime packages are missing: "
                     + ", ".join(missing))
        lines.append("  " + install)

    try:
        encoder = nvc.CreateEncoder(256, 256, "NV12", True, codec="h264")
        del encoder
        lines.append("PyNvVideoCodec is installed and the driver created a test encoder.")
    except Exception as exc:
        lines.append("PyNvVideoCodec is installed but the driver refused to create an encoder:")
        lines.append("  {}".format(exc))
        if not missing:
            lines.append("This is usually an out-of-date GPU driver. Check the driver version")
            lines.append("reported above and update it from NVIDIA, then re-run preflight.")
    return lines


CUDA12_WINDOWS_DRIVER_FLOOR = 527.41


def gpu_hardware():
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        text = (out.stdout or "").strip()
        return text if text else "nvidia-smi returned nothing (no NVIDIA driver?)"
    except FileNotFoundError:
        return "nvidia-smi not found - no NVIDIA driver installed, or not on PATH"
    except Exception as exc:
        return "nvidia-smi failed: {}".format(exc)


def driver_floor_warning(gpu_line):
    """Compare the installed driver against the CUDA 12.x minimum for Windows."""
    parts = [p.strip() for p in gpu_line.split(",")]
    if len(parts) < 2:
        return None
    try:
        version = float(".".join(parts[-1].split(".")[:2]))
    except ValueError:
        return None
    if version < CUDA12_WINDOWS_DRIVER_FLOOR:
        return ("Driver {} is below the CUDA 12.x minimum of {} for Windows - "
                "update the GPU driver before the CUDA packages can load."
                .format(parts[-1], CUDA12_WINDOWS_DRIVER_FLOOR))
    return None


MODE_PATTERN = re.compile(
    r"(?:vcodec|pixel_format)=(\S+)\s+min s=\d+x\d+ fps=[\d.]+\s+"
    r"max s=(\d+)x(\d+) fps=([\d.]+)")


def _run_ffmpeg(arguments, timeout=25):
    import subprocess
    try:
        done = subprocess.run(["ffmpeg", "-hide_banner"] + arguments,
                              capture_output=True, text=True, timeout=timeout)
        return (done.stderr or "") + (done.stdout or "")
    except Exception:
        return ""


NVENC_CODECS = [("h264", "H.264"), ("hevc", "HEVC/H.265"), ("av1", "AV1")]
NVENC_VIA_FFMPEG = [("h264_nvenc", "H.264"), ("hevc_nvenc", "HEVC/H.265"),
                    ("av1_nvenc", "AV1")]
QSV_CODECS = [("h264_qsv", "H.264"), ("hevc_qsv", "HEVC/H.265"), ("av1_qsv", "AV1")]
CPU_CODECS = [("libx264", "H.264 (x264)"), ("libx265", "HEVC/H.265 (x265)"),
              ("libsvtav1", "AV1 (SVT-AV1)"), ("libaom-av1", "AV1 (AOM)")]

WORKS = "works"
NO_HARDWARE = "hardware cannot do it"
NO_BUILD = "not in this ffmpeg build"
NO_FFMPEG = "this ffmpeg cannot"
UNTESTED = "not testable here"
NO_ROUTE = "could not run here"


def ffmpeg_encoder_works(name):
    """Actually encode a few frames. Being listed by ffmpeg is not the same as working."""
    import subprocess
    try:
        done = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30",
             "-frames:v", "5", "-c:v", name, "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        if done.returncode == 0:
            return True, ""
        detail = " ".join((done.stderr or "").split())
        return False, detail[:88]
    except FileNotFoundError:
        return False, "ffmpeg not installed"
    except Exception as exc:
        return False, str(exc)[:88]


def graphics_adapters():
    """Every display adapter, so an Intel iGPU is visible even if ffmpeg knows nothing about it."""
    import json
    import subprocess
    if not sys.platform.startswith("win"):
        return []
    try:
        done = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,DriverVersion | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=40)
        data = json.loads((done.stdout or "").strip() or "[]")
        if isinstance(data, dict):
            data = [data]
        return [(str(d.get("Name", "")), str(d.get("DriverVersion", ""))) for d in data]
    except Exception:
        return []


QSV_RUNTIMES = [
    ("libmfxhw64.dll", "legacy Media SDK"),
    ("libvpl.dll", "oneVPL"),
    ("libmfx64.dll", "Media SDK dispatcher"),
]


def qsv_runtime_dll():
    """List every Quick Sync runtime present.

    Which one is installed decides what ffmpeg can reach: the legacy Media SDK
    exposes less than oneVPL, so a codec can fail here while the silicon is fine.
    """
    import os
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    return [(dll, kind) for dll, kind in QSV_RUNTIMES
            if os.path.exists(os.path.join(system32, dll))]


def report_graphics():
    adapters = graphics_adapters()
    intel = [name for name, _ in adapters if "intel" in name.lower()]
    if adapters:
        print("\nGraphics hardware")
        for name, driver in adapters:
            print("  {:<38} driver {}".format(name[:38], driver))
    runtime = qsv_runtime_dll() if intel else []
    if intel:
        if runtime:
            print("  Quick Sync runtimes: {}".format(
                ", ".join("{} ({})".format(dll, kind) for dll, kind in runtime)))
        else:
            print("  Quick Sync runtimes: NONE found - update the Intel graphics driver")
    return bool(intel), runtime


class quiet_codec_logs:
    """A probe deliberately tries codecs that may fail; that is data, not an error.

    The encoder logs rejections at ERROR unconditionally, which is right during a
    call and pure noise here, where it also breaks the table it prints into.
    """

    NAMES = ("NvidiaCodec", "IntelCodec", "SoftwareCodec", "CodecFactory", "Compression")

    def __enter__(self):
        import logging as pylog
        self._saved = []
        for name in self.NAMES:
            logger = pylog.getLogger(name)
            self._saved.append((logger, logger.level))
            logger.setLevel(pylog.CRITICAL + 1)
        return self

    def __exit__(self, *exc_info):
        for logger, level in self._saved:
            logger.setLevel(level)
        return False


def nvidia_roundtrip(codec):
    """Encode then decode with the classes a call actually uses.

    Indirect probes lie: a stream from libx264 fails to decode on hardware that
    decodes its own NVENC output perfectly, and this ffmpeg lists cuvid decoders
    it cannot run. Only a round trip through the real path is trustworthy.
    """
    import numpy as np
    try:
        from videoconference4k.codec.nvidia import NvidiaEncoder, NvidiaDecoder
    except Exception as exc:
        return False, False, str(exc)[:64]

    encoder = decoder = None
    try:
        encoder = NvidiaEncoder(width=320, height=240, framerate=30,
                                bitrate=2_000_000, codec=codec)
    except Exception as exc:
        return False, False, " ".join(str(exc).split())[:64]

    try:
        decoder = NvidiaDecoder(codec=codec)
        decoded = 0
        for index in range(14):
            frame = np.full((240, 320, 3), 20 + index * 12, np.uint8)
            packet = encoder.encode(frame)
            if packet and decoder.decode(packet, width=320, height=240) is not None:
                decoded += 1
        tail = encoder.flush()
        if tail and decoder.decode(tail, width=320, height=240) is not None:
            decoded += 1
        return True, decoded > 0, "" if decoded else "encoder ran but nothing decoded"
    except Exception as exc:
        return True, False, " ".join(str(exc).split())[:64]
    finally:
        for handle in (encoder, decoder):
            try:
                handle is not None and handle.close()
            except Exception:
                pass


def nvenc_codec_support():
    """Encode and decode verdicts per codec, or None when PyNvVideoCodec is absent."""
    try:
        import PyNvVideoCodec  # noqa: F401
    except Exception:
        return None
    results = []
    for codec, label in NVENC_CODECS:
        with quiet_codec_logs():
            can_encode, can_decode, detail = nvidia_roundtrip(codec)
        results.append((label,
                        WORKS if can_encode else NO_HARDWARE,
                        WORKS if can_decode else (UNTESTED if not can_encode else NO_HARDWARE),
                        detail))
    return results


def ffmpeg_codec_support(codec_list, listed):
    results = []
    for name, label in codec_list:
        if name not in listed:
            results.append((label, NO_BUILD, ""))
            continue
        ok, detail = ffmpeg_encoder_works(name)
        results.append((label, WORKS if ok else NO_HARDWARE, "" if ok else detail))
    return results


def _print_group(title, subtitle, rows):
    print("\n  {}".format(title))
    print("    ({})".format(subtitle))
    if rows is None:
        print("    unavailable on this machine")
        return
    for label, verdict, detail in rows:
        line = "    {:<20} {}".format(label, verdict)
        if verdict == NO_HARDWARE and detail:
            line += "  [{}]".format(detail[:52])
        print(line)


def report_nvenc(nvidia_name, listed):
    """NVENC two ways: the route this program uses, and ffmpeg's, so a mismatch is visible."""
    print("\n  NVIDIA NVENC / NVDEC - {}".format(nvidia_name[:40]))
    print("    (encode and decode are proven together by encoding frames and decoding")
    print("     them back through the very classes a call uses - ffmpeg is NOT required")
    print("     for this path. ffmpeg's own column is shown only to expose a mismatch.)")

    direct = nvenc_codec_support()
    via_ffmpeg = ffmpeg_codec_support(NVENC_VIA_FFMPEG, listed) if listed else None

    print("    {:<14} {:<23} {:<23} {}".format("", "encode", "decode", "ffmpeg's own encode"))
    for position, (_, label) in enumerate(NVENC_CODECS):
        if direct:
            own_enc, own_dec = direct[position][1], direct[position][2]
        else:
            own_enc = own_dec = "PyNvVideoCodec missing"
        if via_ffmpeg is None:
            other = "ffmpeg not found"
        else:
            verdict = via_ffmpeg[position][1]
            other = NO_FFMPEG if verdict == NO_HARDWARE else verdict
        print("    {:<14} {:<23} {:<23} {}".format(label, own_enc, own_dec, other))

    if direct is None:
        if via_ffmpeg and any(v == WORKS for _, v, _ in via_ffmpeg):
            print("\n    MISMATCH: the GPU encodes fine through ffmpeg, but PyNvVideoCodec is")
            print("    missing and that is the ONLY route this program uses. The GPU and")
            print("    driver are healthy - install the four hardware packages to use it.")
        return

    for position, (_, label) in enumerate(NVENC_CODECS):
        own = direct[position][1]
        other = via_ffmpeg[position][1] if via_ffmpeg else None
        if own == WORKS and direct[position][2] != WORKS:
            print("\n    {}: encoding works but decoding the result back did not, so this".format(label))
            print("    codec is not usable end to end here. The silicon may well support")
            print("    decoding it - this test only proves the program's own path failed.")
        if own != WORKS and other == WORKS:
            print("\n    MISMATCH on {}: ffmpeg drives the GPU but PyNvVideoCodec cannot."
                  .format(label))
            print("    The silicon supports it; the Python binding or its CUDA libraries do not.")
        elif own == WORKS and other in (NO_HARDWARE, NO_BUILD):
            print("\n    {} works for this program. Your ffmpeg cannot drive it, but that is"
                  .format(label))
            print("    harmless: the program never uses ffmpeg for NVENC. The GPU is proven")
            print("    capable by the direct test, so treat the ffmpeg column as ffmpeg's limit.")


def report_encoders(has_intel_gpu, qsv_runtime, gpu_line):
    from videoconference4k.codec.base import get_ffmpeg_encoders
    listed = get_ffmpeg_encoders()

    print("\nCodec capability - every entry below was tested by actually running it")
    print("  {:<26} usable right now".format(WORKS))
    print("  {:<26} the codec exists in software, this chip cannot run it".format(NO_HARDWARE))
    print("  {:<26} a fuller ffmpeg would be needed; hardware not testable this way"
          .format(NO_BUILD))
    print("  {:<26} could not be tested, because encoding it failed first".format(UNTESTED))
    print("  {:<26} refused by the only route available; chip vs driver unresolved"
          .format(NO_ROUTE))

    nvidia_name = gpu_line.split(",")[0] if "," in gpu_line else "NVIDIA"
    report_nvenc(nvidia_name, listed)

    if listed:
        qsv_rows = [(label, NO_ROUTE if verdict == NO_HARDWARE else verdict, detail)
                    for label, verdict, detail in ffmpeg_codec_support(QSV_CODECS, listed)]
        _print_group("Intel Quick Sync{}".format(
            "" if has_intel_gpu else " - no Intel GPU detected"),
            "tested through ffmpeg, the only route this program has to it",
            qsv_rows)
        if any(verdict == NO_ROUTE for _, verdict, _ in qsv_rows):
            print("      A Quick Sync failure cannot separate the chip from the driver or")
            print("      runtime, because ffmpeg is the only way this program reaches it.")
            print("      Errors mentioning querying or runtime versions usually mean the")
            print("      installed runtime is too old rather than that the silicon lacks it.")
        _print_group("CPU / software", "tested through ffmpeg",
                     ffmpeg_codec_support(CPU_CODECS, listed))
    else:
        print("\n  ffmpeg not found - Quick Sync and CPU encoding cannot be used or tested")
        if has_intel_gpu:
            print("    This machine HAS an Intel GPU, so installing ffmpeg would unlock Quick Sync.")
        return

    unbuilt_qsv = [label for label, verdict, _ in
                   ffmpeg_codec_support(QSV_CODECS, listed) if verdict == NO_BUILD]
    if unbuilt_qsv and has_intel_gpu:
        print("\n  This machine has an Intel GPU{} but this ffmpeg build lacks"
              .format(" with the Quick Sync runtime installed" if qsv_runtime else ""))
        print("  Quick Sync for: {}. The hardware may well support them - install a".format(
            ", ".join(unbuilt_qsv)))
        print("  full ffmpeg build (gyan.dev or BtbN on Windows) to find out.")


def dshow_video_devices():
    text = _run_ffmpeg(["-list_devices", "true", "-f", "dshow", "-i", "dummy"])
    return re.findall(r'"([^"]+)"\s*\(video\)', text)


def camera_modes(device_name):
    """Every capture mode the camera itself advertises, newest DirectShow data."""
    text = _run_ffmpeg(["-list_options", "true", "-f", "dshow",
                        "-i", "video={}".format(device_name)])
    modes = {}
    for fmt, width, height, fps in MODE_PATTERN.findall(text):
        key = (int(width), int(height))
        modes.setdefault(key, {})
        best = modes[key].get(fmt, 0.0)
        modes[key][fmt] = max(best, float(fps))
    return modes


def report_camera_modes(args):
    devices = dshow_video_devices()
    if not devices:
        return None
    name = devices[args.camera] if args.camera < len(devices) else devices[0]
    print("\nCamera modes advertised by \"{}\"".format(name))
    modes = camera_modes(name)
    if not modes:
        print("  could not read the mode list")
        return None

    for (width, height) in sorted(modes, key=lambda wh: (-wh[0] * wh[1])):
        formats = modes[(width, height)]
        rendered = "   ".join(
            "{} up to {:g} fps".format(fmt, fps)
            for fmt, fps in sorted(formats.items(), key=lambda kv: -kv[1]))
        print("  {:>9}   {}".format("{}x{}".format(width, height), rendered))

    print("\n  Presets this camera can deliver")
    for preset_name, cfg in PRESETS.items():
        width, height = cfg["resolution"]
        wanted = cfg["framerate"]
        best = max(modes.get((width, height), {}).values(), default=0.0)
        verdict = "yes" if best + 0.5 >= wanted else "NO  (camera tops out at {:g} fps)".format(best)
        print("    --preset {:<8} {}x{}@{:<3} {}".format(
            preset_name, width, height, wanted, verdict))
    return modes


def preflight(args):
    from videoconference4k.capture import probe_camera, AudioCapture
    from videoconference4k.codec import get_available_codecs

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 68)
    print("PREFLIGHT  host {}  address {}".format(socket.gethostname(), lan_address()))
    print("=" * 68)

    print("\nGive the other machine this address: {}".format(lan_address()))

    gpu_line = gpu_hardware()
    print("\nGPU  {}".format(gpu_line))
    driver_warning = driver_floor_warning(gpu_line)
    if driver_warning:
        print("     {}".format(driver_warning))

    codecs = get_available_codecs()
    print("\nCodecs")
    for name, ok in codecs.items():
        print("  {:<16} {}".format(name, "yes" if ok else "no"))

    print("  {:<16} {}".format("(intel_qsv above is only what ffmpeg lists;", "see the tested list below)"))

    if not codecs.get("nvidia"):
        print("\nWhy hardware encoding is unavailable")
        for line in nvenc_diagnosis():
            print("  " + line)
        print("  Until this is fixed, use --preset safe; a machine without NVENC")
        print("  cannot decode the other machine's hardware H264 stream.")

    try:
        has_intel_gpu, qsv_runtime = report_graphics()
    except Exception as exc:
        has_intel_gpu, qsv_runtime = False, None
        print("\nGraphics adapter probe failed: {}".format(exc))

    try:
        report_encoders(has_intel_gpu, qsv_runtime, gpu_line)
    except Exception as exc:
        print("\nEncoder test failed: {}".format(exc))

    print("\n  A call currently picks: NVENC if present, else software x264, else JPEG.")
    print("  Quick Sync is NOT yet used by a call even when it works here.")

    print("\nAudio devices   (choose with --mic N and --speaker N)")
    try:
        import sounddevice as sd
        all_devices = sd.query_devices()
        hostapis = [h["name"] for h in sd.query_hostapis()]
        default_in, default_out = sd.default.device

        for kind, channel_key, default_index in (
            ("MICROPHONES", "max_input_channels", default_in),
            ("SPEAKERS", "max_output_channels", default_out),
        ):
            print("  {}".format(kind))
            found = False
            for index, dev in enumerate(all_devices):
                if dev[channel_key] <= 0:
                    continue
                found = True
                marker = "  <-- default (used when you pass no flag)" \
                    if index == default_index else ""
                print("    [{:>2}] {:<44} {:<12} {}ch{}".format(
                    index, dev["name"][:44],
                    hostapis[dev["hostapi"]][:12],
                    dev[channel_key], marker))
            if not found:
                print("    none found")
        if default_in is None or default_in < 0:
            print("  WARNING: no microphone available - run the call with --no-audio")
        print("  The same headset appears once per sound system; any copy works.")
        print("  Windows WASAPI entries generally give the lowest latency.")
    except Exception as exc:
        print("  audio probe failed: {}".format(exc))

    try:
        report_camera_modes(args)
    except Exception as exc:
        print("\nCamera mode list unavailable: {}".format(exc))

    if args.measure_camera:
        presets = []
        for cfg in PRESETS.values():
            entry = (cfg["resolution"][0], cfg["resolution"][1], cfg["framerate"])
            if entry not in presets:
                presets.append(entry)
        print("\nCamera (measured by capturing from each mode - about a minute each)")
        try:
            for want in presets:
                print("  {}x{}@{} ...".format(*want), end=" ", flush=True)
                entry = probe_camera(source=args.camera, presets=[want], sample=20)[0]
                if not entry["opened"]:
                    print("could not open camera")
                    continue
                print("delivered {} at {:.1f} fps  (worst gap {:.0f} ms, p99 {:.0f} ms)".format(
                    entry["delivered"], entry["measured_fps"],
                    entry["worst_interval_ms"], entry["p99_interval_ms"]))
        except Exception as exc:
            print("  camera probe failed: {}".format(exc))
    else:
        print("\nCamera (measured) skipped - the advertised list above comes from the")
        print("  camera itself and is authoritative. Add --measure-camera to also capture")
        print("  from every mode and time the frames; that takes several minutes.")

    print("\nPorts {} and {} must accept inbound TCP.".format(args.video_port, args.audio_port))
    print("If no video appears during the call, run this once as Administrator:")
    print('  netsh advfirewall firewall add rule name="VideoConference4k" '
          'dir=in action=allow protocol=TCP localport={},{}'.format(args.video_port, args.audio_port))
    print("\nPresets")
    for name, cfg in PRESETS.items():
        print("  {:<9} {}".format(name, cfg["note"]))
    for alias, target in PRESET_ALIASES.items():
        print("  {:<9} same as {}".format(alias, target))


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
                microphone_id=args.mic,
                speaker_id=args.speaker,
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
    parser.add_argument("--measure-camera", action="store_true",
                        help="also capture from every camera mode (slow: about a minute per mode)")
    parser.add_argument("--portcheck", action="store_true",
                        help="test whether the peer's ports are reachable (peer must be running)")
    parser.add_argument("--preset", choices=sorted(PRESETS) + sorted(PRESET_ALIASES),
                        default="safe", metavar="NAME")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mic", type=int, default=None,
                        help="microphone index from --preflight (default: system default)")
    parser.add_argument("--speaker", type=int, default=None,
                        help="speaker index from --preflight (default: system default)")
    parser.add_argument("--video-port", default="5555")
    parser.add_argument("--audio-port", default="5556")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--fixed-bitrate", action="store_true",
                        help="disable adaptive bitrate, to tell encoder limits from link limits")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.preset = PRESET_ALIASES.get(args.preset, args.preset)

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
