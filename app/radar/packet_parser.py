"""TI mmWave UART data-port packet parser.

Supports the frame format shared by the mmWave SDK 3.x Out-of-Box demo and
the Radar Toolbox tracking demos:

    magic word   8 bytes   02 01 04 03 06 05 08 07
    version      u32
    totalLen     u32       whole packet, including header
    platform     u32       e.g. 0xA6843
    frameNumber  u32
    timeCpu      u32
    numDetObj    u32
    numTLVs      u32
    subFrame     u32
    TLVs...                type u32, length u32 (payload bytes), payload

TLVs decoded:
    1     detected points           float32 x, y, z, doppler           (OOB demo)
    7     side info per point       int16 snr, noise  (0.1 dB units)  (OOB demo)
    4     range-azimuth heatmap     zero-Doppler cmplx16 per range bin x azimuth antenna (OOB demo)
    6     stats                     (ignored, counted)
    1020  compressed spherical pts  (3D people tracking demo)
    1010  target list               (3D people tracking demo)
Unknown TLVs are skipped and listed in RadarFrame.tlv_types.
"""
from __future__ import annotations

import struct
import time

import numpy as np

from app.models.radar_frame import RadarFrame
from app.models.radar_object import RadarTrack
from app.models.radar_point import POINT_DTYPE, empty_points  # pylint: disable=unused-import

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_FMT = "<8s8I"
HEADER_LEN = struct.calcsize(HEADER_FMT)  # 40
TLV_HDR_FMT = "<2I"
TLV_HDR_LEN = 8
MAX_PACKET_LEN = 1 << 20

TLV_DETECTED_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_AZIMUTH_STATIC_HEATMAP = 4
TLV_STATS = 6
TLV_SIDE_INFO = 7
TLV_COMPRESSED_POINTS = 1020
TLV_TARGET_LIST = 1010
TLV_TARGET_INDEX = 1011

TLV_NAMES = {
    1: "detected points",
    2: "range profile",
    3: "noise profile",
    4: "azimuth static heatmap",
    5: "range-Doppler heatmap",
    6: "stats",
    7: "side info",
    8: "azimuth/elevation static heatmap",
    9: "temperature stats",
    1010: "target list",
    1011: "target index",
    1012: "target height",
    1020: "compressed point cloud",
    1021: "presence indication",
}


class PacketParser:
    """Incremental parser: feed() arbitrary byte chunks, get complete frames back."""

    def __init__(self):
        self._buf = bytearray()
        self.bytes_discarded = 0
        self.bad_packets = 0  # packets damaged in transit (partially salvaged or dropped)
        self.frames_salvaged = 0  # damaged packets whose leading, verified TLVs were kept
        self.frames_parsed = 0

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes, timestamp: float | None = None) -> list[RadarFrame]:
        self._buf.extend(data)
        frames = []
        ts = time.time() if timestamp is None else timestamp
        while True:
            idx = self._buf.find(MAGIC)
            if idx < 0:
                # keep the tail in case it holds a partial magic word
                keep = len(MAGIC) - 1
                if len(self._buf) > keep:
                    self.bytes_discarded += len(self._buf) - keep
                    del self._buf[:-keep]
                break
            if idx > 0:
                self.bytes_discarded += idx
                del self._buf[:idx]
            if len(self._buf) < HEADER_LEN:
                break
            total_len = struct.unpack_from("<I", self._buf, 12)[0]
            if total_len < HEADER_LEN or total_len > MAX_PACKET_LEN:
                self.bad_packets += 1
                del self._buf[: len(MAGIC)]
                continue
            # A packet that lost bytes in transit runs into the next packet's magic word.  Keep the
            # TLVs that can be verified as intact (normally the point cloud, which comes first),
            # then resync at the next packet instead of swallowing it too.
            nxt = self._buf.find(MAGIC, len(MAGIC), min(total_len, len(self._buf)))
            if nxt > 0:
                self.bad_packets += 1
                damaged = bytes(self._buf[:nxt])
                del self._buf[:nxt]
                try:
                    fr = parse_packet(damaged, ts, partial=True)
                except (struct.error, ValueError):
                    fr = None
                if fr is not None and fr.tlv_types:
                    frames.append(fr)
                    self.frames_salvaged += 1
                else:
                    self.bytes_discarded += len(damaged)
                continue
            if len(self._buf) < total_len:
                break
            packet = bytes(self._buf[:total_len])
            del self._buf[:total_len]
            try:
                frames.append(parse_packet(packet, ts))
                self.frames_parsed += 1
            except (struct.error, ValueError):
                self.bad_packets += 1
        return frames


def _tlv_plausible(tlv_type: int, tlv_len: int, num_obj: int, remaining: int) -> bool:
    """Sanity check of a TLV header read from a damaged packet."""
    if tlv_type not in TLV_NAMES or tlv_len > remaining:
        return False
    if tlv_type == TLV_DETECTED_POINTS:
        return tlv_len == 16 * num_obj
    if tlv_type == TLV_SIDE_INFO:
        return tlv_len == 4 * num_obj
    return True


def parse_packet(packet: bytes, timestamp: float = 0.0, partial: bool = False) -> RadarFrame:
    """Decode one packet.  partial=True: the packet lost bytes somewhere; keep only TLVs whose own
    header and the FOLLOWING TLV header (or the exact packet end) are consistent, and stop at the
    first one that is not."""
    (_, version, total_len, platform, frame_num, _cpu, num_obj, num_tlvs, _sub) = struct.unpack_from(
        HEADER_FMT, packet, 0
    )
    frame = RadarFrame(frame_number=frame_num, timestamp=timestamp, platform=platform, version=version, raw=packet,
                       damaged=partial)
    off = HEADER_LEN
    xyzv = None
    side = None
    for k in range(num_tlvs):
        if off + TLV_HDR_LEN > len(packet):
            if partial:
                break
            raise ValueError("truncated TLV header")
        tlv_type, tlv_len = struct.unpack_from(TLV_HDR_FMT, packet, off)
        if partial:
            end = off + TLV_HDR_LEN + tlv_len
            if not _tlv_plausible(tlv_type, tlv_len, num_obj, len(packet) - off - TLV_HDR_LEN):
                break
            if k + 1 < num_tlvs:  # the next header must also look right, else the loss is inside this TLV
                if end + TLV_HDR_LEN > len(packet):
                    break
                nt, nl = struct.unpack_from(TLV_HDR_FMT, packet, end)
                if not _tlv_plausible(nt, nl, num_obj, len(packet) - end - TLV_HDR_LEN + 10**9):
                    break
        off += TLV_HDR_LEN
        payload = packet[off : off + tlv_len]
        if len(payload) < tlv_len:
            if partial:
                break
            raise ValueError("truncated TLV payload")
        off += tlv_len
        frame.tlv_types.append(tlv_type)
        if tlv_type == TLV_DETECTED_POINTS:
            n = tlv_len // 16
            xyzv = np.frombuffer(payload, dtype="<f4", count=n * 4).reshape(n, 4)
        elif tlv_type == TLV_SIDE_INFO:
            n = tlv_len // 4
            side = np.frombuffer(payload, dtype="<i2", count=n * 2).reshape(n, 2).astype(np.float32) * 0.1
        elif tlv_type == TLV_AZIMUTH_STATIC_HEATMAP:
            # zero-Doppler range x azimuth-virtual-antenna samples, cmplx16 (imag, real) per entry;
            # reshaped by the consumer, which knows the number of range bins from the config
            iq = np.frombuffer(payload, dtype="<i2", count=tlv_len // 2).reshape(-1, 2).astype(np.float32)
            frame.az_heatmap = (iq[:, 1] + 1j * iq[:, 0]).astype(np.complex64)
        elif tlv_type == TLV_COMPRESSED_POINTS:
            frame.points = _parse_compressed_points(payload)
        elif tlv_type == TLV_TARGET_LIST:
            frame.tracks = _parse_target_list(payload)
    if xyzv is not None:
        pts = empty_points(len(xyzv))
        pts["x"], pts["y"], pts["z"], pts["doppler"] = xyzv.T
        if side is not None and len(side) == len(xyzv):
            pts["snr"], pts["noise"] = side.T
        frame.points = pts
    return frame


def _parse_compressed_points(payload: bytes) -> np.ndarray:
    if len(payload) < 20:
        return empty_points()
    el_u, az_u, dop_u, rng_u, snr_u = struct.unpack_from("<5f", payload, 0)
    n = (len(payload) - 20) // 8
    raw = np.frombuffer(
        payload,
        dtype=np.dtype([("el", "i1"), ("az", "i1"), ("dop", "<i2"), ("rng", "<u2"), ("snr", "<u2")]),
        count=n,
        offset=20,
    )
    el = raw["el"] * el_u
    az = raw["az"] * az_u
    r = raw["rng"] * rng_u
    pts = empty_points(n)
    pts["x"] = r * np.cos(el) * np.sin(az)
    pts["y"] = r * np.cos(el) * np.cos(az)
    pts["z"] = r * np.sin(el)
    pts["doppler"] = raw["dop"] * dop_u
    # snrUnit gives linear SNR; express in dB for consistency with the OOB demo
    pts["snr"] = 10 * np.log10(np.maximum(raw["snr"] * snr_u, 1e-6))
    return pts


_TARGET_FMT = "<I27f"
_TARGET_LEN = struct.calcsize(_TARGET_FMT)  # 112


def _parse_target_list(payload: bytes) -> list[RadarTrack]:
    tracks = []
    for i in range(len(payload) // _TARGET_LEN):
        v = struct.unpack_from(_TARGET_FMT, payload, i * _TARGET_LEN)
        tracks.append(RadarTrack(track_id=v[0], x=v[1], y=v[2], z=v[3], vx=v[4], vy=v[5], vz=v[6], confidence=v[27]))
    return tracks


# ---------------------------------------------------------------- encoding
def build_packet(frame_number: int, points: np.ndarray, platform: int = 0xA6843, include_side_info: bool = True) -> bytes:
    """Encode points as an OOB-demo packet (used by the simulator and tests)."""
    n = len(points)
    tlvs = []
    xyzv = np.column_stack([points["x"], points["y"], points["z"], points["doppler"]]).astype("<f4")
    tlvs.append((TLV_DETECTED_POINTS, xyzv.tobytes()))
    if include_side_info:
        snr = np.nan_to_num(points["snr"], nan=0.0)
        noise = np.nan_to_num(points["noise"], nan=0.0)
        side = np.round(np.column_stack([snr, noise]) * 10).astype("<i2")
        tlvs.append((TLV_SIDE_INFO, side.tobytes()))
    body = b"".join(struct.pack(TLV_HDR_FMT, t, len(p)) + p for t, p in tlvs)
    total = HEADER_LEN + len(body)
    pad = (-total) % 32  # the OOB demo pads packets to 32-byte multiples
    total += pad
    header = struct.pack(HEADER_FMT, MAGIC, 0x03060000, total, platform, frame_number, 0, n, len(tlvs), 0)
    return header + body + b"\x00" * pad
