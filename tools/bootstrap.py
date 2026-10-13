"""Fetch pinned native dependencies; repeat runs verify the cached bytes."""

import argparse
import hashlib
import json
import platform
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEPS = {
    "eigen": "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
    "json": "https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz",
    "httplib": "https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.60.1.tar.gz",
    "catch2": "https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.1.tar.gz",
    "utf8proc": "https://github.com/JuliaStrings/utf8proc/archive/refs/tags/v2.12.0.tar.gz",
}


def download(name, url, expected=None):
    directory = ROOT / ".cache" / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".tar.gz")
    if not path.exists():
        temporary = path.with_suffix(".tmp")
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and expected != digest:
        raise RuntimeError(
            f"Checksum mismatch for {name}; remove the invalid cache file."
        )
    return {"url": url, "sha256": digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record hashes for a reviewed dependency update.",
    )
    args = parser.parse_args()
    manifest = ROOT / "cmake" / "native-deps.json"
    pinned = json.loads(manifest.read_text()) if manifest.exists() else {}
    if not pinned and not args.record:
        parser.error(
            "Missing dependency manifest; use --record for an initial reviewed lock."
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            name: executor.submit(
                download, name, url, pinned.get(name, {}).get("sha256")
            )
            for name, url in DEPS.items()
        }
        records = {name: future.result() for name, future in futures.items()}
    if args.record:
        manifest.write_text(json.dumps(records, indent=2) + "\n")
    native = []
    for name, record in records.items():
        native.append(
            f'set({name.upper()}_URL "{record["url"]}")\nset({name.upper()}_SHA256 "{record["sha256"]}")'
        )
    (ROOT / "cmake" / "native-deps.cmake").write_text("\n".join(native) + "\n")
    system, machine = platform.system(), platform.machine()
    names = {
        ("Darwin", "arm64"): "osx-arm64",
        ("Linux", "x86_64"): "linux-x64",
        ("Linux", "aarch64"): "linux-aarch64",
    }
    key = names.get((system, machine))
    if not key:
        raise RuntimeError(f"Unsupported native prebuilt platform: {system}/{machine}")
    version = "1.31.0"
    name = f"onnxruntime-{key}-{version}"
    runtime_manifest = ROOT / "cmake" / "onnxruntime-deps.json"
    runtime_records = (
        json.loads(runtime_manifest.read_text()) if runtime_manifest.exists() else {}
    )
    if key not in runtime_records and not args.record:
        raise RuntimeError(f"Unpinned native runtime platform: {key}")
    record = download(
        name,
        f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/{name}.tgz",
        runtime_records.get(key, {}).get("sha256"),
    )
    if args.record:
        runtime_records[key] = record
        runtime_manifest.write_text(json.dumps(runtime_records, indent=2) + "\n")
    destination = ROOT / ".cache" / "native"
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / name).exists():
        with tarfile.open(
            ROOT / ".cache" / "downloads" / (name + ".tar.gz")
        ) as archive:
            archive.extractall(destination, filter="data")
    (ROOT / "cmake" / "runtime-local.cmake").write_text(
        f'set(ONNXRUNTIME_ROOT "{destination / name}")\n'
    )
    print(
        json.dumps(
            {"onnxruntime_root": str(destination / name), "dependencies": list(records)}
        )
    )


if __name__ == "__main__":
    main()
