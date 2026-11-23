"""Selftest: camera.bin pack/unpack is byte-identical and random seek/read matches the raw stream."""
import hashlib, json, os, random, tempfile
from pathlib import Path

from src.engine import camera_bin_store as cbs


def main():
    rnd = random.Random(1)
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        raw = bytes(rnd.getrandbits(8) for _ in range(300_000)) + b"\xff\xd8" * 2_000_000 + os.urandom(5_000_000)
        (d / "camera.bin").write_bytes(raw)
        cbs.CHUNK = 1 << 20
        r = cbs.pack(d)
        assert r["status"] == "packed" and not (d / "camera.bin").exists(), r
        assert cbs.is_packed(d) and cbs.verify(d)
        with cbs.open_camera_bin(d) as fh:
            for _ in range(300):
                a = rnd.randrange(len(raw)); n = rnd.randrange(1, 3_000_000)
                fh.seek(a); assert fh.read(n) == raw[a:a + n]
            fh.seek(0); assert fh.read() == raw
        assert cbs.unpack(d)["status"] == "unpacked"
        assert hashlib.sha256((d / "camera.bin").read_bytes()).digest() == hashlib.sha256(raw).digest()
        assert not (d / "camera.bin.zc").exists() and not (d / "camera_chunks.json").exists()
        with cbs.open_camera_bin(d) as fh:
            fh.seek(123); assert fh.read(10) == raw[123:133]
    print("camera_bin_store selftest OK")


if __name__ == "__main__":
    main()
