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


# Layer 2 vocabulary: how a codec gets encoded here. Deliberately the same
# strings the wire uses for its type tag, so nothing has to translate.
IMPL_NVENC = "nvenc"
IMPL_QSV = "intel_qsv"
IMPL_SOFTWARE = "software"

HARDWARE_IMPLEMENTATIONS = (IMPL_NVENC, IMPL_QSV)


def software_encoder_available(codec: str) -> bool:
    """Whether a CPU encoder exists for this codec. There is no software AV1 path."""
    codec = normalize_codec(codec)
    if codec == "h264":
        return has_x264()
    if codec == "hevc":
        return has_x265()
    return False


def encoder_implementation(codec: str) -> Optional[str]:
    """The best local way to encode this codec, or None if there is no way at all.

    This is the whole of layer two. Which codec to use is agreed with the far end;
    how to produce it is nobody else's business, so this is never announced and
    never negotiated. Hardware first, then the CPU.
    """
    codec = normalize_codec(codec)
    if has_nvidia_codec(codec):
        return IMPL_NVENC
    if has_intel_codec(codec):
        return IMPL_QSV
    if software_encoder_available(codec):
        return IMPL_SOFTWARE
    return None


def encoder_is_hardware(codec: str) -> bool:
    return encoder_implementation(codec) in HARDWARE_IMPLEMENTATIONS


def can_encode(codec: str) -> bool:
    """Whether anything on this machine can encode this codec."""
    return encoder_implementation(codec) is not None


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


def _peer_list(remote) -> list:
    """Accept one peer or many, so the same rule serves a call and a conference."""
    if not remote:
        return []
    if isinstance(remote, dict):
        return [remote]
    return [peer for peer in remote if peer]


def choose_send_codec(
    local: Dict[str, Dict[str, bool]],
    remote=None,
    priority: Sequence[str] = DEFAULT_CODEC_PRIORITY,
    fallback: str = "h264",
    prefer_hardware: bool = False,
) -> str:
    """Pick the codec I will send, given everyone who has to decode it.

    One rule covers every case: I must be able to encode it and every receiver
    must be able to decode it. What the receivers can encode, and what I can
    decode, are beside the point, which is why hardware that decodes a codec it
    cannot produce causes no trouble - such a peer simply receives that codec
    while sending something else back.

    Adding a participant can only narrow the choice, never widen it, so a third
    caller who cannot decode HEVC quietly forces everyone down to H.264.
    """
    peers = _peer_list(remote)
    if not peers:
        return normalize_codec(fallback)

    usable = []
    for candidate in normalize_priority(priority):
        if not (local.get(candidate) or {}).get("encode"):
            continue
        if all((peer.get(candidate) or {}).get("decode") for peer in peers):
            usable.append(candidate)

    if not usable:
        return normalize_codec(fallback)

    if prefer_hardware:
        accelerated = [codec for codec in usable if encoder_is_hardware(codec)]
        if accelerated:
            return accelerated[0]

    return usable[0]


def describe(capabilities: Dict[str, Dict[str, bool]]) -> str:
    parts = []
    for codec, entry in capabilities.items():
        flags = "".join((
            "e" if entry.get("encode") else "-",
            "d" if entry.get("decode") else "-",
        ))
        parts.append("{}:{}".format(codec, flags))
    return " ".join(parts)
