"""Reconstruct native-trained LeNet weights and export interoperable ONNX."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]


class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(256, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.nn.functional.avg_pool2d(torch.relu(self.conv2(x)), 2).flatten(1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def main():
    project = ROOT / "artifacts/cnn"
    run = json.loads((project / "latest.json").read_text())["run_id"]
    snapshot = json.loads((project / run / "model.json").read_text())
    model = LeNet()
    model.load_state_dict(
        {
            name: torch.tensor(p["data"], dtype=torch.float32).reshape(p["shape"])
            for name, p in snapshot["parameters"].items()
        }
    )
    model.eval()
    images = (
        np.frombuffer(
            (ROOT / "data/mnist/t10k-images-idx3-ubyte").read_bytes()[16:],
            dtype=np.uint8,
        )[: 3 * 784]
        .reshape(3, 1, 28, 28)
        .astype(np.float32)
        / 255
    )
    destination = ROOT / "models/cnn"
    destination.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.from_numpy(images[:1]),
        str(destination / "model.onnx"),
        input_names=["image"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}},
    )
    with torch.no_grad():
        python = model(torch.from_numpy(images)).numpy()
    native = np.array(
        json.loads((ROOT / "artifacts/cnn-native-logits.json").read_text())
    ).reshape(3, 10)
    runtime = ort.InferenceSession(str(destination / "model.onnx")).run(
        None, {"image": images}
    )[0]
    report = {
        "native_python_max_abs": float(np.max(np.abs(native - python))),
        "native_onnx_max_abs": float(np.max(np.abs(native - runtime))),
        "predicted_digits": runtime.argmax(1).tolist(),
    }
    (destination / "input.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "image": {
                        "dtype": "float32",
                        "shape": list(images.shape),
                        "data": images.ravel().tolist(),
                    }
                }
            }
        )
    )
    (ROOT / "docs/results/cnn-parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
