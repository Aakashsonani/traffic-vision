"""Download immutable, checksummed inference models. Uses only the standard library."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected:
        print(f"Verified {destination.name}", flush=True)
        return
    temp = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            print(f"Downloading {destination.name} ...", flush=True)
            if url.startswith("https://github.com/Aakashsonani/traffic-vision/releases/download/"):
                if not shutil.which("gh"):
                    raise RuntimeError("Install GitHub CLI and run gh auth login for private model assets.")
                subprocess.run(["gh", "release", "download", "models-v1",
                                "--repo", "Aakashsonani/traffic-vision", "--pattern", destination.name,
                                "--output", str(temp), "--clobber"], check=True,
                               capture_output=True, timeout=300)
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "TrafficVision/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
            if sha256(temp) != expected:
                raise ValueError(f"Checksum mismatch for {destination.name}; file not installed")
            os.replace(temp, destination)
            return
        except Exception:
            temp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    manifest = json.loads((ROOT / "model-manifest.json").read_text(encoding="utf-8"))
    for name, spec in manifest.items():
        download(spec.get("url", spec["source"]), ROOT / "models" / spec["file"], spec["sha256"])
        if "labels" in spec:
            (ROOT / "models" / f"{name}.labels.json").write_text(json.dumps(spec["labels"]))
    print("All models verified. Run start.bat (Windows) or python -m traffic_vision.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
