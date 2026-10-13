"""Record parity measurements for real models and native preprocessing."""

import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from vision_model import image_tensor

ROOT = Path(__file__).resolve().parents[1]


def native(app, *args):
    result = subprocess.run(
        [str(ROOT / "build" / app), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main():
    report = {}
    texts = [
        "This movie was excellent.",
        "This film was terrible and boring.",
        "Café naïve résumé!",
        "Hello\tworld\nnext",
        "你好世界",
        "İSTANBUL Straße Ωμέγα",
        "[CLS] a [MASK] word [SEP]",
        "🙂🚀 punctuation—test",
        "x" * 101,
        "  padded   text  ",
        "can't won't don't",
        "",
        "zero\u200bwidth",
        "ＡＢＣ fullwidth",
        "가나다 한글",
        "some words " * 100,
    ]
    tokenizer = Tokenizer.from_file(str(ROOT / "models/bert/tokenizer.json"))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(length=128)
    mismatches = []
    for text in texts:
        actual = native("sentiment_service", "--mode", "tokenize", "--text", text)
        expected = tokenizer.encode(text)
        if (
            actual["input_ids"] != expected.ids
            or actual["attention_mask"] != expected.attention_mask
        ):
            mismatches.append(
                {
                    "text": text,
                    "native": actual,
                    "expected_ids": expected.ids,
                    "expected_mask": expected.attention_mask,
                }
            )
    report["wordpiece"] = {
        "cases": len(texts),
        "mismatches": len(mismatches),
        "details": mismatches,
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sentiment = ort.InferenceSession(str(ROOT / "models/bert/model.onnx"), options)
    errors = []
    for text in texts[:2]:
        encoded = tokenizer.encode(text)
        inputs = {
            "input_ids": np.array([encoded.ids], np.int64),
            "attention_mask": np.array([encoded.attention_mask], np.int64),
        }
        logits = sentiment.run(None, inputs)[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        actual = native("sentiment_service", "--text", text)
        errors.append(abs(actual["positive_probability"] - float(probabilities[1])))
    report["bert"] = {
        "examples": len(errors),
        "max_probability_abs_error": max(errors),
        "scope": "native WordPiece+C++ ORT versus reference tokenizer+Python ORT",
    }
    vision = ort.InferenceSession(str(ROOT / "models/vision/model.onnx"), options)
    errors = []
    for category in ["good", "broken_large", "contamination"]:
        path = next((ROOT / "data/mvtec/bottle/test" / category).glob("*.png"))
        expected = float(
            vision.run(None, {"image": image_tensor(path).unsqueeze(0).numpy()})[0][0]
        )
        actual = native(
            "vision_inspector",
            "--image",
            str(path),
            "--output",
            str(ROOT / "artifacts/parity-inspection.jpg"),
        )
        errors.append(abs(expected - actual["score"]))
    report["vision"] = {
        "images": len(errors),
        "max_score_abs_error": max(errors),
        "scope": "OpenCV resize/RGB/ImageNet normalization and ORT across native/Python",
    }
    model = ROOT / "artifacts/tabular-regression"
    run = json.loads((model / "latest.json").read_text())["run_id"]
    parameters = json.loads((model / run / "model.json").read_text())
    inputs = json.loads((ROOT / "models/linear-input.json").read_text())["inputs"][
        "features"
    ]
    values = np.array(inputs["data"]).reshape(inputs["shape"])
    normalized = (values - np.array(parameters["normalizer"]["mean"])) / np.array(
        parameters["normalizer"]["scale"]
    )
    weights = np.array(parameters["weights"])
    expected = normalized @ weights[1:] + weights[0]
    result = native("inference_engine", "--mode", "infer")["outputs"]["prediction"][
        "data"
    ]
    report["tabular"] = {
        "rows": len(expected),
        "float64_native_weights_vs_float32_onnx_max_abs": float(
            np.max(np.abs(expected - result))
        ),
    }
    destination = ROOT / "docs/results/runtime-parity.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
