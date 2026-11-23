"""Lossless chunked storage for a source session's camera.bin (2026-11-22).

A recorded session keeps `camera_index.jsonl` unchanged (offsets into the RAW camera.bin stream).  A packed
session replaces `camera.bin` with:
    camera.bin.zc        concatenated zlib streams, one per RAW chunk of CHUNK bytes
    camera_chunks.json   {"format": "skjutbana-camera-zc", "version": 1, "chunk_size": .., "raw_size": ..,
                          "raw_sha256": .., "chunks": [[packed_offset, packed_length], ...]}
`open_camera_bin(dir)` returns a seekable read-only file object over the RAW stream for either layout, so
readers keep using seek(offset)/read(length).  Packing verifies the full RAW sha256 by decompressing the
packed file before camera.bin is removed; `unpack` restores the byte-identical original.

    python3 -m src.engine.camera_bin_store pack SESSION_DIR [...]
    python3 -m src.engine.camera_bin_store unpack SESSION_DIR
    python3 -m src.engine.camera_bin_store verify SESSION_DIR
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zlib
from pathlib import Path

CHUNK = 4 * 1024 * 1024
FORMAT = "skjutbana-camera-zc"


class PackedCameraBin(io.RawIOBase):
    def __init__(self, directory: Path):
        self.meta = json.loads((directory / "camera_chunks.json").read_text())
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"unknown camera pack format in {directory}")
        self.size = int(self.meta["raw_size"])
        self.chunk = int(self.meta["chunk_size"])
        self.chunks = self.meta["chunks"]
        self.fh = open(directory / "camera.bin.zc", "rb")
        self.pos = 0
        self._cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + int(offset))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _chunk(self, k: int) -> bytes:
        data = self._cache.get(k)
        if data is None:
            off, length = self.chunks[k]
            self.fh.seek(off)
            data = zlib.decompress(self.fh.read(length))
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[k] = data
        return data

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            k, inner = divmod(self.pos, self.chunk)
            data = self._chunk(k)
            take = data[inner:inner + (end - self.pos)]
            out += take
            self.pos += len(take)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def close(self) -> None:
        self.fh.close()
        super().close()


def is_packed(directory: Path) -> bool:
    directory = Path(directory)
    return not (directory / "camera.bin").exists() and (directory / "camera_chunks.json").exists()


def open_camera_bin(directory) -> io.RawIOBase:
    directory = Path(directory)
    if (directory / "camera.bin").exists():
        return open(directory / "camera.bin", "rb")
    return PackedCameraBin(directory)


def _iter_unpacked(directory: Path):
    with PackedCameraBin(directory) as src:
        for off, length in src.chunks:
            src.fh.seek(off)
            yield zlib.decompress(src.fh.read(length))


def verify(directory: Path) -> bool:
    meta = json.loads((Path(directory) / "camera_chunks.json").read_text())
    h, n = hashlib.sha256(), 0
    for data in _iter_unpacked(Path(directory)):
        h.update(data); n += len(data)
    return n == meta["raw_size"] and h.hexdigest() == meta["raw_sha256"]


def pack(directory: Path, level: int = 6) -> dict:
    """Pack camera.bin -> camera.bin.zc, verify byte-identity, then remove camera.bin."""
    directory = Path(directory)
    raw = directory / "camera.bin"
    if not raw.exists():
        return dict(session=directory.name, status="skip_no_raw")
    tmp_zc, tmp_meta = directory / "camera.bin.zc.tmp", directory / "camera_chunks.json.tmp"
    h, chunks, raw_size, packed = hashlib.sha256(), [], 0, 0
    with open(raw, "rb") as src, open(tmp_zc, "wb") as dst:
        while True:
            data = src.read(CHUNK)
            if not data:
                break
            h.update(data); raw_size += len(data)
            z = zlib.compress(data, level)
            dst.write(z)
            chunks.append([packed, len(z)]); packed += len(z)
        dst.flush(); os.fsync(dst.fileno())
    if raw_size != raw.stat().st_size:
        tmp_zc.unlink()
        return dict(session=directory.name, status="error_size_changed")
    meta = dict(format=FORMAT, version=1, codec="zlib", level=level, chunk_size=CHUNK, raw_size=raw_size,
                raw_sha256=h.hexdigest(), chunks=chunks,
                note="camera.bin packed losslessly; camera_index.jsonl offsets refer to the RAW stream. "
                     "Restore: python3 -m src.engine.camera_bin_store unpack <session>")
    with open(tmp_meta, "w") as fh:
        json.dump(meta, fh); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp_zc, directory / "camera.bin.zc")
    os.replace(tmp_meta, directory / "camera_chunks.json")
    if not verify(directory):
        (directory / "camera.bin.zc").unlink(); (directory / "camera_chunks.json").unlink()
        return dict(session=directory.name, status="error_verify")
    raw.unlink()
    return dict(session=directory.name, status="packed", raw=raw_size, packed=packed)


def unpack(directory: Path) -> dict:
    directory = Path(directory)
    if (directory / "camera.bin").exists():
        return dict(session=directory.name, status="skip_raw_exists")
    meta = json.loads((directory / "camera_chunks.json").read_text())
    tmp = directory / "camera.bin.tmp"
    h = hashlib.sha256()
    with open(tmp, "wb") as dst:
        for data in _iter_unpacked(directory):
            h.update(data); dst.write(data)
        dst.flush(); os.fsync(dst.fileno())
    if h.hexdigest() != meta["raw_sha256"]:
        tmp.unlink()
        return dict(session=directory.name, status="error_verify")
    os.replace(tmp, directory / "camera.bin")
    (directory / "camera.bin.zc").unlink(); (directory / "camera_chunks.json").unlink()
    return dict(session=directory.name, status="unpacked")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, dirs = argv[0], argv[1:]
    ok = True
    for d in dirs:
        if cmd == "pack":
            r = pack(Path(d))
        elif cmd == "unpack":
            r = unpack(Path(d))
        else:
            r = dict(session=Path(d).name, status="ok" if verify(Path(d)) else "error_verify")
        ok &= not r["status"].startswith("error")
        print(json.dumps(r), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
