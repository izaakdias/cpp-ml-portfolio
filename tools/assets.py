"""Verified downloads and deterministic conversion of public datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "wine": (
        "https://archive.ics.uci.edu/static/public/109/wine.zip",
        "https://archive.ics.uci.edu/dataset/109/wine",
        "CC BY 4.0",
    ),
    "mpg": (
        "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip",
        "https://archive.ics.uci.edu/dataset/9/auto+mpg",
        "CC BY 4.0",
    ),
    "seeds": (
        "https://archive.ics.uci.edu/static/public/236/seeds.zip",
        "https://archive.ics.uci.edu/dataset/236/seeds",
        "CC BY 4.0",
    ),
    "movielens": (
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "https://grouplens.org/datasets/movielens/100k/",
        "GroupLens research-use terms; see archive README",
    ),
    "nab": (
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv",
        "https://github.com/numenta/NAB",
        "See NAB repository license and attribution",
    ),
    "nab-labels": (
        "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json",
        "https://github.com/numenta/NAB",
        "See NAB repository license and attribution",
    ),
}
MNIST = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]
BERT = "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/714eb0fa89d2f80546fda750413ed43d93601a13/onnx/"
for name in MNIST:
    SOURCES[name] = (
        "https://ossci-datasets.s3.amazonaws.com/mnist/" + name + ".gz",
        "https://yann.lecun.com/exdb/mnist/",
        "See original MNIST provider terms",
    )
for name in ["model.onnx", "vocab.txt", "tokenizer.json", "config.json"]:
    SOURCES["bert-" + name] = (
        BERT + name,
        "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "Apache-2.0 model card",
    )
SOURCES["ssd"] = (
    "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx",
    "https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1",
    "See ONNX model zoo model license",
)


def fetch(name, record=False):
    manifest = ROOT / "docs/assets.lock.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    if name not in records and not record:
        raise RuntimeError(f"Unpinned asset {name}; record a reviewed download first")
    url, source, license = SOURCES[name]
    directory = ROOT / ".cache/assets"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if not path.exists():
        temporary = path.with_suffix(".part")
        with (
            requests.get(url, stream=True, timeout=(20, 180)) as response,
            temporary.open("wb") as output,
        ):
            response.raise_for_status()
            for block in response.iter_content(1024 * 1024):
                output.write(block)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = records.get(name, {}).get("sha256")
    if expected and expected != digest:
        raise RuntimeError(f"Checksum mismatch: {name}")
    return path, {
        "url": url,
        "source": source,
        "license": license,
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def prepare(files):
    data = ROOT / "data"
    models = ROOT / "models"
    data.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    if "wine" in files:
        with zipfile.ZipFile(files["wine"]) as archive:
            rows = list(csv.reader(io.StringIO(archive.read("wine.data").decode())))
        header = [
            "alcohol",
            "malic_acid",
            "ash",
            "alkalinity",
            "magnesium",
            "phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280",
            "proline",
            "target",
        ]
        write_csv(
            data / "wine.csv",
            header,
            [row[1:] + [int(int(row[0]) == 1)] for row in rows],
        )
    if "mpg" in files:
        with zipfile.ZipFile(files["mpg"]) as archive:
            raw = archive.read("auto-mpg.data").decode()
        rows = []
        for line in raw.splitlines():
            values = line.split()[:8]
            if len(values) == 8 and "?" not in values:
                rows.append(values[1:] + [values[0]])
        write_csv(
            data / "auto-mpg.csv",
            [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "model_year",
                "origin",
                "mpg",
            ],
            rows,
        )
    if "seeds" in files:
        with zipfile.ZipFile(files["seeds"]) as archive:
            raw = archive.read("seeds_dataset.txt").decode()
        write_csv(
            data / "seeds.csv",
            [
                "area",
                "perimeter",
                "compactness",
                "length",
                "width",
                "asymmetry",
                "groove",
                "variety",
            ],
            [line.split() for line in raw.splitlines() if line.strip()],
        )
    if "movielens" in files:
        with zipfile.ZipFile(files["movielens"]) as archive:
            raw = archive.read("ml-100k/u.data").decode()
        write_csv(
            data / "movielens.csv",
            ["user_id", "item_id", "rating", "timestamp"],
            [line.split() for line in raw.splitlines()],
        )
    if "nab" in files:
        labels = json.loads(files["nab-labels"].read_text())[
            "realKnownCause/machine_temperature_system_failure.csv"
        ]
        windows = [
            (datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in labels
        ]
        # Sort and deduplicate original timestamps before constructing any windows.
        readings = {
            row["timestamp"]: float(row["value"])
            for row in csv.DictReader(files["nab"].open())
        }
        rows = []
        stream = []
        for timestamp, value in sorted(readings.items()):
            moment = datetime.fromisoformat(timestamp)
            label = int(any(a <= moment <= b for a, b in windows))
            rows.append([value, label])
            stream.append(
                {
                    "device": "machine-1",
                    "timestamp_ms": int(moment.replace(tzinfo=UTC).timestamp() * 1000),
                    "values": [value],
                }
            )
        write_csv(data / "nab.csv", ["temperature", "anomaly"], rows)
        (data / "nab-stream.jsonl").write_text(
            "\n".join(json.dumps(row) for row in stream) + "\n"
        )
    for name in MNIST:
        if name in files:
            destination = data / "mnist"
            destination.mkdir(exist_ok=True)
            (destination / name).write_bytes(gzip.decompress(files[name].read_bytes()))
    for name in ["model.onnx", "vocab.txt", "tokenizer.json", "config.json"]:
        if "bert-" + name in files:
            destination = models / "bert"
            destination.mkdir(exist_ok=True)
            (destination / name).write_bytes(files["bert-" + name].read_bytes())
    if "ssd" in files:
        destination = models / "mobile"
        destination.mkdir(exist_ok=True)
        (destination / "ssd.onnx").write_bytes(files["ssd"].read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["core", "mnist", "bert", "mobile", "all"])
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    groups = {
        "core": ["wine", "mpg", "seeds", "movielens", "nab", "nab-labels"],
        "mnist": MNIST,
        "bert": [
            "bert-" + name
            for name in ["model.onnx", "vocab.txt", "tokenizer.json", "config.json"]
        ],
        "mobile": ["ssd"],
    }
    names = list(SOURCES) if args.group == "all" else groups[args.group]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = {
            name: future.result()
            for name, future in {
                name: executor.submit(fetch, name, args.record) for name in names
            }.items()
        }
    if args.record:
        manifest = ROOT / "docs/assets.lock.json"
        records = json.loads(manifest.read_text()) if manifest.exists() else {}
        records.update({name: record for name, (_, record) in results.items()})
        manifest.write_text(json.dumps(records, indent=2) + "\n")
    prepare({name: path for name, (path, _) in results.items()})
    print(json.dumps({"prepared": names}))


if __name__ == "__main__":
    main()
