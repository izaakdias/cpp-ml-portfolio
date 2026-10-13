# Data, models, and licenses

The implementation's MIT license does not change the licenses of datasets, weights, or libraries. No dataset, book PDF, or model weights are included in Git. Scripts verify recorded hashes before preparing local files.

| Material | Source and terms |
|---|---|
| Book | [Publisher's official repository](https://github.com/PacktPublishing/Hands-on-Machine-learning-with-C-Second-Edition); conceptual inspiration, without reproducing the book's text/code |
| Wine, Auto MPG, and Seeds | [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), [Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg), [Seeds](https://archive.ics.uci.edu/dataset/236/seeds); CC BY 4.0 as listed on the UCI pages |
| MovieLens 100k | [GroupLens](https://grouplens.org/datasets/movielens/100k/); research-use and attribution terms in the archive's README. Do not assume the ratings are MIT-licensed. |
| NAB | [Numenta Anomaly Benchmark](https://github.com/numenta/NAB); `realKnownCause/machine_temperature_system_failure.csv` and intervals from `combined_windows.json`; consult the original repository's licenses/attribution |
| MNIST | [Original page](https://yann.lecun.com/exdb/mnist/); files served by the OSSCI PyTorch mirror. Attribution: LeCun, Cortes, and Burges. |
| DistilBERT SST-2 | [Official model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english); Apache-2.0 according to the card; revision and ONNX/vocab/tokenizer assets pinned in `assets.lock.json` |
| ResNet18 | [TorchVision weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html); official ImageNet1K V1 weights, with full hash in `vision-assets.lock.json`. ImageNet data and weight terms must be considered separately from the TorchVision code license. |
| MVTec AD bottle | [Original source](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), **CC BY-NC-SA 4.0**, noncommercial use; [mirror used](https://huggingface.co/datasets/foersben/mvtec-ad), revision `c75b39616f84db43677bcc8228caaafaf5096d7f`, with per-image hashes. Reference: Bergmann et al., MVTec AD, CVPR 2019 / IJCV 2021. The mirror grants no additional rights. |
| SSD MobileNet v1 | [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1); opset 12 model derived from TensorFlow Object Detection, with COCO classes. Consult the model card and original weight/data terms. |

## Dependencies

- Eigen 3.4.0: MPL-2.0 and per-file notices.
- nlohmann/json: MIT.
- cpp-httplib: MIT.
- Catch2: BSL-1.0.
- utf8proc: MIT and the Unicode data license included in the package.
- OpenSSL: Apache-2.0 for the version used locally; consult your system distribution.
- ONNX Runtime: MIT; the distribution includes third-party notices.
- OpenCV 4.12: Apache-2.0 and optional-component notices.
- PyTorch/LibTorch and TorchVision: BSD and third-party notices in their packages.
- Gradle wrapper: Apache-2.0; scripts/JAR are the originals from release 8.9, with hashes in `cmake/gradle-wrapper.lock.json`.
- AndroidX/CameraX: Apache-2.0.

Manifests record URLs, sizes, and SHA hashes. Version changes require reviewing provenance/terms and updating lock files. Weights prepared using MVTec AD are intended for the stated academic/portfolio context, subject to the dataset's terms.
