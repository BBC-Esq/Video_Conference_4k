"""
VideoConference4k
=================

A lightweight peer-to-peer video conferencing library for 4K60 streaming.

Author: Blair Chintella (vici0549@gmail.com)
License: Apache-2.0
"""

from .version import __version__
from .capture import VideoCapture, AudioCapture
from .stream import VideoStream
from .net import SyncTransport, AsyncTransport
from .rtc import RTCConnection
from .conference import PeerConference, MultiPeerConference

__author__ = "Blair Chintella <vici0549@gmail.com>"

__all__ = [
    "VideoCapture",
    "AudioCapture",
    "VideoStream",
    "SyncTransport",
    "AsyncTransport",
    "RTCConnection",
    "PeerConference",
    "MultiPeerConference",
    "__version__",
]