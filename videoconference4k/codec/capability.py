from typing import Dict, Optional, Sequence

from .base import get_ffmpeg_decoders, normalize_codec
from .nvidia import has_nvidia_codec
from .intel import has_intel_codec
from .software import has_x264, has_x265, has_software_codec
from ..utils.common import get_logger

logger = get_logger("Capability")

KNOWN_CODECS = ("h264", "hevc", "av1")

DEFAULT_PRIORITY = ("h264",)

FFMPEG_DECODER_NAMES = {"h264": "h264", "hevc": "hevc", "av1": "av1"}


def can_encode(codec: str) -> bool:
    """Whether anything on this machine can encode this codec."""
    codec = normalize_codec(codec)
    if has_nvidia_codec(codec):
        return True
    if has_intel_codec(codec):
        return True
    if codec == "h264" and has_x264():
        return True
    if codec == "hevc" and has_x265():
        return True
    return False


def can_decode(codec: str) -> bool:
    """Whether anything on this machine can decode this codec.

    Decoding is the wider capability: hardware gains it a generation before it
    gains the matching encoder, and a software decoder is available for these
    codecs in any ordinary ffmpeg, so this is deliberately generous.
    """
    codec = normalize_codec(codec)
    if has_nvidia_codec(codec):
        return True
    name = FFMPEG_DECODER_NAMES.get(codec)
    if name and has_software_codec():
        decoders = get_ffmpeg_decoders()
        return bool(decoders) and name in decoders
    return False


def local_capabilities(codecs: Sequence[str] = KNOWN_CODECS) -> Dict[str, Dict[str, bool]]:
    """What this machine can send and receive, per codec.

    Deliberately says nothing about which implementation would be used. A peer
    only needs to know whether a codec can be handled at all; how is a private
    matter for whichever side is doing the work.
    """
    result = {}
    for codec in codecs:
        codec = normalize_codec(codec)
        result[codec] = {"encode": can_encode(codec), "decode": can_decode(codec)}
    return result


def choose_send_codec(
    local: Dict[str, Dict[str, bool]],
    remote: Optional[Dict[str, Dict[str, bool]]],
    priority: Sequence[str] = DEFAULT_PRIORITY,
    fallback: str = "h264",
) -> str:
    """Pick the codec for one direction: what I encode and the far end decodes.

    Negotiation is per direction, never symmetric. A peer that can decode a codec
    it cannot produce is common, so the two directions of a call may legitimately
    settle on different codecs.
    """
    for candidate in priority:
        candidate = normalize_codec(candidate)
        mine = local.get(candidate) or {}
        if not mine.get("encode"):
            continue
        if remote is None:
            continue
        theirs = remote.get(candidate) or {}
        if theirs.get("decode"):
            return candidate
    return normalize_codec(fallback)


def describe(capabilities: Dict[str, Dict[str, bool]]) -> str:
    parts = []
    for codec, entry in capabilities.items():
        flags = "".join((
            "e" if entry.get("encode") else "-",
            "d" if entry.get("decode") else "-",
        ))
        parts.append("{}:{}".format(codec, flags))
    return " ".join(parts)
