"""Build only the OpenCV modules needed by the image/camera application."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz"
parser = argparse.ArgumentParser()
parser.add_argument("--record", action="store_true")
args = parser.parse_args()
archive = ROOT / ".cache/downloads/opencv-4.12.0.tar.gz"
archive.parent.mkdir(parents=True, exist_ok=True)
if not archive.exists():
    with urlopen(URL, timeout=120) as response, archive.open("wb") as target:
        while block := response.read(1024 * 1024):
            target.write(block)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
manifest = ROOT / "cmake/opencv-dep.json"
if args.record:
    manifest.write_text(json.dumps({"url": URL, "sha256": digest}, indent=2) + "\n")
if json.loads(manifest.read_text())["sha256"] != digest:
    raise RuntimeError("OpenCV checksum mismatch")
source = ROOT / ".cache/native/opencv-4.12.0"
if not source.exists():
    with tarfile.open(archive) as package:
        package.extractall(source.parent, filter="data")
build = ROOT / ".cache/opencv-build"
prefix = ROOT / ".cache/opencv"
flags = [
    "cmake",
    "-S",
    str(source),
    "-B",
    str(build),
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    f"-DCMAKE_INSTALL_PREFIX={prefix}",
    "-DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_opencv_python3=OFF",
    "-DWITH_QT=OFF",
    "-DWITH_FFMPEG=OFF",
    "-DWITH_OPENCL=OFF",
    "-DWITH_OPENEXR=OFF",
    "-DWITH_TIFF=OFF",
    "-DWITH_WEBP=OFF",
    "-DWITH_JASPER=OFF",
    "-DWITH_IPP=OFF",
    "-DBUILD_JAVA=OFF",
    "-DWITH_VTK=OFF",
    "-DBUILD_opencv_apps=OFF",
]
subprocess.run(flags, check=True)
subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
subprocess.run(["cmake", "--install", str(build)], check=True)
print(f"OpenCV_DIR={prefix}/lib/cmake/opencv4")
