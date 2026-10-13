"""Export the real native-trained regression, including its preprocessing."""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]


def main():
    project = ROOT / "artifacts/tabular-regression"
    run = json.loads((project / "latest.json").read_text())["run_id"]
    model = json.loads((project / run / "model.json").read_text())
    weights = np.array(model["weights"], dtype=np.float32)
    mean = np.array(model["normalizer"]["mean"], dtype=np.float32)
    scale = np.array(model["normalizer"]["scale"], dtype=np.float32)
    nodes = [
        helper.make_node("Sub", ["features", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "scale"], ["normalized"]),
        helper.make_node("MatMul", ["normalized", "weights"], ["linear"]),
        helper.make_node("Add", ["linear", "bias"], ["prediction"]),
    ]
    tensors = [
        numpy_helper.from_array(value, name)
        for name, value in [
            ("mean", mean),
            ("scale", scale),
            ("weights", weights[1:].reshape(-1, 1)),
            ("bias", weights[:1]),
        ]
    ]
    graph = helper.make_graph(
        nodes,
        "native_auto_mpg",
        [
            helper.make_tensor_value_info(
                "features", TensorProto.FLOAT, ["batch", len(mean)]
            )
        ],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, ["batch", 1])],
        tensors,
    )
    exported = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
    )
    onnx.checker.check_model(exported)
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    onnx.save(exported, str(directory / "linear.onnx"))
    rows = np.loadtxt(ROOT / "data/auto-mpg.csv", delimiter=",", skiprows=1)[
        :3, :-1
    ].astype(np.float32)
    (directory / "linear-input.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "features": {
                        "dtype": "float32",
                        "shape": list(rows.shape),
                        "data": rows.ravel().tolist(),
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )
    (directory / "linear-schema.json").write_text(
        json.dumps(
            {"features": model["features"], "source_run": run, "dtype": "float32"},
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"model": "models/linear.onnx", "features": len(mean), "source_run": run}
        )
    )


if __name__ == "__main__":
    main()
