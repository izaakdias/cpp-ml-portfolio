"""Fit a normal-image reference and export a CNN feature anomaly scorer."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
MIRROR = "https://huggingface.co/datasets/foersben/mvtec-ad"
WEIGHTS = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def download(record=False):
    manifest = ROOT / "docs/vision-assets.lock.json"
    locked = json.loads(manifest.read_text()) if manifest.exists() else {}
    if not locked and not record:
        raise RuntimeError("Missing reviewed vision asset lock")
    listing = requests.get(
        f"https://huggingface.co/api/datasets/foersben/mvtec-ad/revision/{REVISION}",
        timeout=60,
    )
    listing.raise_for_status()
    paths = [
        x["rfilename"]
        for x in listing.json()["siblings"]
        if x["rfilename"].startswith(("bottle/train/", "bottle/test/"))
        and x["rfilename"].endswith(".png")
    ]
    assets = {
        path: (f"{MIRROR}/resolve/{REVISION}/{path}", ROOT / "data/mvtec" / path)
        for path in paths
    }
    assets["resnet18"] = (WEIGHTS, ROOT / ".cache/assets/resnet18.pth")

    def fetch(item):
        name, (url, path) = item
        if not record and name not in locked:
            raise RuntimeError(f"Unpinned vision asset: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            response = requests.get(url, timeout=(30, 180))
            response.raise_for_status()
            temporary = path.with_suffix(".part")
            temporary.write_bytes(response.content)
            temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in locked and digest != locked[name]["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {name}")
        return name, {"url": url, "sha256": digest, "bytes": path.stat().st_size}

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = dict(pool.map(fetch, assets.items()))
    if record:
        manifest.write_text(json.dumps(records, indent=2) + "\n")
    return assets["resnet18"][1]


def image_tensor(path):
    # Match native OpenCV INTER_LINEAR at uint8 resolution, including rounding.
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(
        cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB
    )
    array = rgb.astype(np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class Scorer(torch.nn.Module):
    def __init__(self, encoder, mean, scale):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, image):
        embedding = self.encoder(image)
        return ((embedding - self.mean) / self.scale).square().mean(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    weights = download(args.record)
    torch.set_num_threads(2)
    torch.manual_seed(41)
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(weights, weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    train = sorted((ROOT / "data/mvtec/bottle/train/good").glob("*.png"))
    rng = np.random.default_rng(41)
    rng.shuffle(train)
    boundary = int(0.8 * len(train))
    fit, validation = train[:boundary], train[boundary:]

    def features(paths):
        batches = []
        with torch.no_grad():
            for start in range(0, len(paths), 16):
                batches.append(
                    encoder(
                        torch.stack(
                            [image_tensor(p) for p in paths[start : start + 16]]
                        )
                    )
                )
        return torch.cat(batches)

    reference = features(fit)
    model = Scorer(
        encoder, reference.mean(0), reference.std(0, unbiased=False).clamp_min(0.05)
    ).eval()
    v = features(validation)
    threshold = float(
        np.quantile(((v - model.mean) / model.scale).square().mean(1).numpy(), 0.98)
    )
    test = sorted((ROOT / "data/mvtec/bottle/test").glob("*/*.png"))
    labels = np.array([int(p.parent.name != "good") for p in test])
    f = features(test)
    scores = ((f - model.mean) / model.scale).square().mean(1).numpy()
    predicted = scores >= threshold
    tp, fp, fn = (
        int(np.sum(predicted & (labels == 1))),
        int(np.sum(predicted & (labels == 0))),
        int(np.sum(~predicted & (labels == 1))),
    )
    metrics = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "train_normal": len(fit),
        "validation_normal": len(validation),
        "test_images": len(test),
    }
    destination = ROOT / "models/vision"
    destination.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        image_tensor(test[0]).unsqueeze(0),
        str(destination / "model.onnx"),
        input_names=["image"],
        output_names=["score"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    config = {
        "threshold": threshold,
        "sha256": hashlib.sha256((destination / "model.onnx").read_bytes()).hexdigest(),
        "metrics": metrics,
        "architecture": "ImageNet ResNet18 + diagonal standardized embedding distance",
        "scope": "image-level anomaly detection; no defect localization",
        "license": "MVTec AD CC BY-NC-SA 4.0 data; ImageNet pretrained weights terms apply",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (ROOT / "docs/results").mkdir(exist_ok=True)
    (ROOT / "docs/results/vision.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config))


if __name__ == "__main__":
    main()
