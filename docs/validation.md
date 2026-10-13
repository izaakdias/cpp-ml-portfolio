# Methodology and evidence

## Reference run

macOS Apple Silicon, Apple Clang 21, CMake 4.4.1, Ninja 1.13.2, ONNX Runtime 1.31.0, LibTorch/PyTorch 2.14.1, and OpenCV 4.12.0. Seeds are fixed; bitwise determinism across compilers, BLAS implementations, and platforms is not guaranteed. Files in [results](results/) record the run's measurements; downloaded versions and checksums are recorded in the manifests and `uv.lock`.

All nine desktop applications were compiled locally. The initial C++ suite passed **26/26 test cases** in Release mode. It covers malformed and quoted CSV, finite values, tied metrics, constant-feature scaling, splits, models, persistence, corruption, path traversal, concurrent writes, real ONNX tensors, tokenization, and the MLflow REST protocol. CI rebuilds on Linux in Release and ASan/UBSan configurations; leak detection is disabled because of interactions with external libraries that are not instrumented.

## Data protocols

| Dataset | Split / selection | Metrics and limitations |
|---|---|---|
| Wine | Class 1 versus the others; 106 training, 34 validation, and 38 test examples. Class-stratified split; scaler fitted on training data; L2, k, and threshold selected on validation data. | F1/precision/recall/ROC-AUC/AP with a majority-class baseline. kNN classified the entire small test set correctly; this does not establish universal generalization. |
| Auto MPG | 392 complete rows; 235/78/79 split, seed 41. Rows with missing horsepower are excluded. Scaling uses training data only; ridge is selected by validation MAE. | MAE/RMSE/R² versus the training mean. This is neither a price prediction nor a causal estimate of fuel consumption. |
| Seeds | 210 rows; exploration uses the entire dataset, with scaler/PCA fitted on that dataset. KMeans/GMM use the known k=3; DBSCAN uses eps=1.2/min_pts=5. | ARI compares assignments with actual varieties. Default DBSCAN has ARI near zero because density connects the groups. Epsilon was not selected using test ARI. |
| NAB | Sorted, deduplicated series; chronological 60/20/20 splits **before** forming 24-point windows. Features: mean/std/last/range. Only normal training windows fit the forest. | Threshold maximizes validation F1 when both classes are present; fallback is the 98th percentile of normal scores. The independent test split has 4,514 windows. A window is positive if any point falls within an official NAB anomaly interval. These metrics are not the official NAB score and do not measure incident lead time. Overlapping windows produce correlated samples. |
| MovieLens 100k | Last interaction per user held out; 99,057 training ratings and 943 test ratings. 16 factors, seed 61, 20 epochs. | Rating RMSE/MAE/R²; HitRate and NDCG@10 on 486 holdouts with rating ≥4, using the full catalog and excluding seen items. One holdout per user means recall=hit per query. This is a per-user holdout rather than a global temporal cutoff; another user's training data may occur after a user's holdout. |
| MNIST | Permutation with seed 41; 54,000 training/6,000 validation examples. 5 epochs, batch size 64, Adam 0.001. Best checkpoint selected by validation accuracy. | Official 10,000-image test set, evaluated once after selection. 98.52% accuracy. LeNet is a compact baseline, with no state-of-the-art claim. |
| MVTec AD bottle | 209 normal training images split 167/42 with seed 41. Frozen ImageNet ResNet18 encoder. Per-feature mean/std fitted on training data; minimum std 0.05. Threshold is the 98th percentile of normal validation scores. | 83 official test images: 63 defective and 20 normal. ROC-AUC/AP and F1 at the frozen threshold. The method labels entire images; it does not evaluate mask segmentation/IoU. |
| SST-2 pretrained | Frozen public DistilBERT weights; native tokenizer and exported model. No fine-tuning in this project. | Demonstration and numerical parity. Neither a full SST-2 evaluation nor a calibration evaluation was performed. English only. |
| SSD COCO pretrained | Frozen public model; RGB uint8 NHWC and Android CPU backend. | A successful build/APK does not measure detection quality/mAP. No physical device/camera evaluation was performed during this run. |

## Parity and performance

`tools/export_cnn.py` compares three MNIST images and records maximum error: **0** between the reconstructed Python weights and C++, and approximately **1.9×10⁻⁶** with ONNX. This result covers those images, architecture, and versions; it is not proof for every exported model.

`tools/evaluate_runtime.py` records tokenization on Unicode/truncation examples, DistilBERT probabilities, three inspection images, and exported regression with its scaler. Reports include sample counts and differences; parity does not replace model-quality evaluation.

The CLI benchmark includes validation of already loaded JSON, tensor allocation, inference, and output copying/serialization into a JSON object. It excludes file reads, session initialization, HTTP/network transport, and the final JSON dump. `p50/p95/p99` use nearest-rank percentiles after warmup. The Auto MPG demonstration uses a batch of three rows, CPU/1 thread, 10 warmups, and 200 repetitions. **This alone is not a fair C++ versus Python comparison**, nor an SLA promise.

## Complete systems

- Real local MLflow 3.17.0: experiment created, run opened, metrics/parameters logged, and run finished. Weights stayed in the local registry, as specified by the client contract.
- Streaming: the entire NAB series was processed as JSONL, with warmup and per-observation scores. Final state preserves readings, timestamp, and model run. This is a replay, not new sensor data.
- Vision: a real defective image was processed by the C++ binary and saved with annotations. Neither the macOS camera nor the Android application was exercised on physical hardware.
- APK and Linux builds: check the [workflow](https://github.com/izaakdias/cpp-ml-portfolio/actions/workflows/build.yml) for the exact commit's outcome. A previous successful run does not establish that a newer commit passes.

## Improvements guided by measurements

The popularity baseline achieved **HitRate@10 of 0.1111**, compared with matrix factorization's **0.0514**. For ratings, matrix factorization achieved RMSE **1.0092**, compared with the mean baseline's **1.2127**. In this configuration, better rating estimates did not improve ranking. Ranking uses continuous scores before clipping to 1..5; recommendations display the raw score and bounded rating separately.

The recommender provides a rating baseline, but its full-catalog ranking remains weak. Compare against popularity, adopt a global temporal protocol, and evaluate BPR/implicit ALS before commercial recommendation use. The NAB detector favors recall and produces false alerts; validate the threshold, incident cost, and cooldown policy on real sensors. DBSCAN needs a density/epsilon study and may require HDBSCAN. These limitations are part of the published results.
