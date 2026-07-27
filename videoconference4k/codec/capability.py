from typing import Dict, Optional, Sequence

from .base import CODEC_ALIASES, get_ffmpeg_decoders, normalize_codec
from .nvidia import has_nvidia_codec
from .intel import has_intel_codec
from .software import has_x264, has_x265, has_software_codec
from ..utils.common import get_logger

logger = get_logger("Capability")

KNOWN_CODECS = ("h264", "hevc", "av1")

# The order a codec is tried in, most preferred first. H.264 leads because it is
# the most widely supported and the cheapest to decode; HEVC and AV1 follow for
# the cases where both ends can do better. This is a default, not a rule: it is
# meant to be replaced wholesale by whatever a user chooses in settings.
DEFAULT_CODEC_PRIORITY = ("h264", "hevc", "av1")

# Names for each rank, so a settings screen can talk about "first choice"
# without having to know that the order happens to be stored as a sequence.
PRIORITY_LABELS = ("first", "second", "third", "fourth", "fifth")

DEFAULT_PRIORITY = DEFAULT_CODEC_PRIORITY

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


def normalize_priority(order: Optional[Sequence[str]]) -> tuple:
    """Turn whatever a caller or a settings screen supplies into a usable order.

    Accepts any spelling of a codec, drops names this build knows nothing about,
    removes repeats, and appends whatever was left out so the result is always a
    complete ranking. A user reordering two entries should never quietly delete
    the third.
    """
    ranked = []
    for name in (order or ()):
        # match the alias table directly: normalize_codec answers h264 for anything
        # it does not recognise, which would silently promote a typo to first place
        key = str(name or "").strip().lower()
        codec = CODEC_ALIASES.get(key)
        if codec in KNOWN_CODECS and codec not in ranked:
            ranked.append(codec)
    for codec in DEFAULT_CODEC_PRIORITY:
        if codec not in ranked:
            ranked.append(codec)
    return tuple(ranked)


def describe_priority(order: Optional[Sequence[str]] = None) -> str:
    """Render an order the way a settings screen would label it."""
    ranked = normalize_priority(order if order is not None else DEFAULT_CODEC_PRIORITY)
    return ", ".join(
        "{}={}".format(PRIORITY_LABELS[index] if index < len(PRIORITY_LABELS) else index + 1, codec)
        for index, codec in enumerate(ranked)
    )


def choose_send_codec(
    local: Dict[str, Dict[str, bool]],
    remote: Optional[Dict[str, Dict[str, bool]]],
    priority: Sequence[str] = DEFAULT_CODEC_PRIORITY,
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
