# Running the projects

Start with the README quickstart and run these commands from the repository root. All binaries accept `--help`; failures emit JSON to stderr and exit with code 2.

## 1. ONNX inference engine

```bash
build/ml_baselines --task regression
uv run python tools/export_tabular.py
build/inference_engine --mode inspect
build/inference_engine --mode infer
build/inference_engine --mode benchmark --warmup 10 --runs 200
build/inference_engine --mode serve --workers 4
```

The demo model is Auto MPG regression trained in C++, with normalization included in the graph. It accepts `[batch,7]` inputs. HTTP contract: `{"inputs":{"features":{"dtype":"float32","shape":[1,7],"data":[...]}}}`. See `models/linear-schema.json` for feature order. The runtime also accepts int64/uint8 tensors, used by BERT/SSD.

## 2. Industrial vision inspector

```bash
uv run python tools/vision_model.py
build/vision_inspector --image data/mvtec/bottle/test/broken_large/000.png \
  --output artifacts/inspection.jpg
build/vision_inspector --camera 0 --frames 300
# Use a video format supported by your OpenCV build:
build/vision_inspector --video sample.mp4 --headless --frames 100
```

The script downloads only the bottle category from a pinned mirror and verifies each image. It fits the mean/scale of ResNet18 embeddings on 167 normal images; 42 normal images calibrate the threshold. Test defects are excluded from fitting. The score applies to the whole image; defect localization is not provided. The minimal OpenCV build excludes FFmpeg; video codecs depend on the system backend.

## 3. Native sentiment service

```bash
uv run python tools/assets.py bert
build/sentiment_service --mode tokenize --text 'Café, excellent movie!'
build/sentiment_service --text 'This movie was excellent.'
build/sentiment_service --mode serve --port 8081
```

DistilBERT fine-tuned on SST-2, two classes, **English**, sequence length 128. UTF-8 handling, lowercase, NFD, accent stripping, punctuation/CJK splitting, subwords, truncation, and padding run in C++. Softmax probabilities do not guarantee calibration or explain predictions, and do not establish quality in another language.

## 4. Streaming anomaly monitor

```bash
build/anomaly_monitor
build/anomaly_monitor --mode replay
build/anomaly_monitor --mode stream --stream data/nab-stream.jsonl \
  --state artifacts/device-state > artifacts/events.jsonl
```

Each line: `{"device":"machine-1","timestamp_ms":123456789,"values":[83.2]}`. After 24 readings, it emits a score/alert and reference distance. Each device has an independent window. The checkpoint retains the model version and recent values. To resume, supply **only timestamps after** the checkpoint; duplicates and decreasing timestamps are rejected. A new `--state` directory starts a clean run. CSV replay may include training observations; it demonstrates model application rather than providing a new independent evaluation.

## 5. Native recommender

```bash
build/recommender --epochs 20 --user 1
build/recommender --mode recommend --user 1 --limit 10
```

MovieLens 100k, 16 factors, SGD, user/item biases, and regularization. The second command restores the snapshot without retraining. Previously seen movies are excluded. Evaluation holds out each user's latest interaction; RMSE uses all 943 holdouts and ranking uses the 486 with rating ≥4. Movie IDs can be joined to the dataset's `u.item` file. Catalog serving and online training are outside this application's scope.

## 6. Clustering explorer

```bash
build/clustering_explorer
build/clustering_explorer --epsilon 0.7 --minimum-points 5 --output artifacts/seeds-eps07.csv
uv run python tools/plot_clusters.py --output artifacts/clustering.png
```

The output CSV contains pc1, pc2, and assignments from all three methods; normalization precedes distance calculations. `--clusters` controls KMeans/GMM and `--epsilon` affects DBSCAN. UCI labels are used only for ARI. The default DBSCAN configuration poorly recovers the varieties; the published result shows the method's sensitivity. Artifacts contain exploratory assignments, not a general prediction model for new points.

## 7. Tabular ML library

```bash
build/ml_baselines --task binary
build/ml_baselines --task regression
build/ml_baselines --data other.csv --task regression --seed 41
```

Input is numeric CSV with unique headers; the last column is the target. The default training/validation/test split is 60/20/20, with rounding and class stratification for binary classification. Wine's target is class 1 versus the others; normalization uses training data only. Ridge and L2/k are selected on validation data. Selected parameters are neither tuned nor refitted on the test set.

## 8. Native CNN trainer

```bash
uv run python tools/assets.py mnist
build/native_cnn --epochs 5 --limit 60000
uv run python tools/export_cnn.py
build/inference_engine --model models/cnn/model.onnx --input models/cnn/input.json --mode infer
```

LeNet: conv 1→6→16 with size-5 kernels, average pooling, FC 256→120→84→10, ReLU, cross-entropy, and Adam at 0.001. Training uses 54,000 images; 6,000 select the checkpoint; the official test set contains 10,000. IDX parsing validates magic values, dimensions, payload sizes, and classes. The exporter reconstructs named weights from C++ training and records numerical differences between LibTorch, PyTorch, and ONNX.

## 9. Native experiment tracker

```bash
uv sync --frozen --extra tracking
uv run mlflow server --backend-store-uri sqlite:///.cache/mlflow.db \
  --artifacts-destination .cache/mlflow-artifacts --host 127.0.0.1 --port 5000
# In another terminal:
build/experiment_tracker --uri http://127.0.0.1:5000
```

The C++ client creates/finds an experiment, starts a run, logs parameters/metrics, and finishes with FINISHED/FAILED. The local artifact records the remote run ID. `MLFLOW_TRACKING_TOKEN` enables a Bearer header. This client keeps weights in the local registry and does not upload them to MLflow. It implements a subset of the MLflow SDK's capabilities.

## 10. Android

[Android guide: prepare, build, and install](../projects/android-detector/README.md).
