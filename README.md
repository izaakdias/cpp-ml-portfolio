# C++ ML Portfolio

[![Build portfolio](https://github.com/izaakdias/cpp-ml-portfolio/actions/workflows/build.yml/badge.svg)](https://github.com/izaakdias/cpp-ml-portfolio/actions/workflows/build.yml)

Dez projetos originais de machine learning e engenharia de sistemas, inspirados nos temas de **Hands-On Machine Learning with C++, Second Edition**, de Kirill Kolodiazhnyi. Implementações próprias, dados públicos reais, modelos persistidos e comandos reproduzíveis. O livro e seus exemplos não são redistribuídos.

O núcleo e as aplicações desktop são C++20. Python prepara dados e exporta modelos; o app Android usa Kotlin/CameraX, ONNX Runtime e pré/pós-processamento próprio em C++ via JNI.

## Projetos

| # | Projeto | Modelo e entrega | Guia |
|---|---|---|---|
| 1 | ONNX inference engine | Biblioteca, CLI, HTTP, lotes dinâmicos, percentis de latência | [Executar](docs/projects.md#1-onnx-inference-engine) |
| 2 | Industrial vision inspector | ResNet18 + distância de embeddings; imagens, vídeo e câmera | [Executar](docs/projects.md#2-industrial-vision-inspector) |
| 3 | Native sentiment service | DistilBERT SST-2 + WordPiece Unicode nativo; CLI e HTTP | [Executar](docs/projects.md#3-native-sentiment-service) |
| 4 | Streaming anomaly monitor | Isolation Forest, janelas, estado por dispositivo e checkpoints | [Executar](docs/projects.md#4-streaming-anomaly-monitor) |
| 5 | Native recommender | Fatoração matricial com vieses, ranking e recuperação do modelo | [Executar](docs/projects.md#5-native-recommender) |
| 6 | Clustering explorer | KMeans++, DBSCAN, GMM diagonal e PCA | [Executar](docs/projects.md#6-clustering-explorer) |
| 7 | Tabular ML library | Ridge via QR, regressão logística e kNN; seleção em validação | [Executar](docs/projects.md#7-tabular-ml-library) |
| 8 | Native CNN trainer | LeNet/MNIST treinada em C++ com LibTorch; checkpoint e ONNX | [Executar](docs/projects.md#8-native-cnn-trainer) |
| 9 | Native experiment tracker | Cliente REST MLflow integrado ao treinamento de regressão | [Executar](docs/projects.md#9-native-experiment-tracker) |
| 10 | Android offline detector | SSD MobileNet v1, CameraX, YUV/rotação/NMS em C++ | [Executar](projects/android-detector/README.md) |

## Resultados medidos

Resultados de uma execução local no macOS arm64, CPU. As políticas de divisão e seleção estão em [metodologia](docs/validation.md). Estes resultados descrevem os datasets usados; não comprovam desempenho em uma fábrica ou em qualquer celular.

| Aplicação | Resultado |
|---|---|
| LeNet / MNIST | **98,52%** de acurácia, 10.000 imagens de teste |
| Inspeção / MVTec AD bottle | ROC-AUC **0,9833**, F1 **0,9421**, 83 imagens de teste |
| Isolation Forest / NAB | ROC-AUC **0,9655**, recall **0,9644**, precisão **0,6459** |
| Regressão / Auto MPG | MAE **2,7466 mpg**; baseline da média **6,7666 mpg** |
| Logística / Wine binário | F1 **0,8696**; kNN de referência **1,0** em apenas 38 amostras de teste |
| KMeans / Seeds | Adjusted Rand Index **0,7733** |
| Recomendação / MovieLens 100k | RMSE **1,0092**; ranking completo e limitações registrados no relatório |

[Resultados em JSON](docs/results/) · [Arquitetura e trade-offs](docs/architecture.md) · [Modelos e avaliação](docs/validation.md) · [Licenças dos dados/modelos](docs/data-and-licenses.md)

## Começar

Requisitos: CMake ≥3.25, Ninja, compilador C++20, OpenSSL e [uv](https://docs.astral.sh/uv/). Plataformas de runtime pré-compilado: macOS Apple Silicon, Linux x64 e Linux arm64. O treinamento nativo usa a instalação de LibTorch fornecida pelo wheel de PyTorch.

```bash
# macOS: se ainda não instalados
brew install cmake ninja openssl@3
# Linux: apt install cmake ninja-build libssl-dev

uv sync --frozen
uv run python tools/bootstrap.py
uv run python tools/assets.py core
cmake --preset release -DPORTFOLIO_VISION=OFF -DPORTFOLIO_TORCH=OFF
cmake --build --preset release

build/ml_baselines
build/ml_baselines --task regression
build/anomaly_monitor
build/recommender
build/clustering_explorer
uv run python tools/export_tabular.py
build/inference_engine --mode infer
```

Os downloads são verificados por SHA-256. Dados, pesos e artefatos grandes ficam fora do Git. `--record` é reservado à atualização revisada dos manifestos; a reprodução normal usa os hashes já publicados.

### Build com os nove executáveis desktop

```bash
uv run python tools/build_opencv.py
cmake --preset release -DPORTFOLIO_VISION=ON -DPORTFOLIO_TORCH=ON \
  -DOpenCV_DIR="$PWD/.cache/opencv/lib/cmake/opencv4" \
  -DCMAKE_PREFIX_PATH="$(uv run python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build --preset release
```

No Linux, `libopencv-dev` instalado pelo gerenciador também funciona, dispensando `build_opencv.py` e `OpenCV_DIR`.

### Serviços locais

```bash
build/inference_engine --mode serve --port 8080
# Em outro terminal:
curl http://127.0.0.1:8080/health
curl -H 'Content-Type: application/json' --data-binary @models/linear-input.json \
  http://127.0.0.1:8080/infer

uv run python tools/assets.py bert
build/sentiment_service --mode serve --port 8081
curl -H 'Content-Type: application/json' -d '{"text":"This movie was excellent."}' \
  http://127.0.0.1:8081/sentiment
```

Os serviços são configurados para loopback. Um serviço exposto precisa de autenticação, limites e TLS no gateway de implantação. DistilBERT SST-2 avalia **inglês**; não é um classificador de sentimento em português.

## Organização

```text
include/portfolio/   contratos públicos
src/                algoritmos, persistência, tokenização e adaptadores
apps/               composição dos casos de uso e CLIs
tools/              dados, exportação e preparação de dependências
tests/              testes C++ de contratos, algoritmos e integrações
projects/           aplicação Android com Kotlin e JNI
docs/               decisões, comandos, métricas e licenças
```

A CI compila os executáveis Linux, executa a suíte existente em Release e ASan/UBSan e compila o APK Android. A execução de câmera física e do app em aparelho precisa de equipamento; não há números inventados de FPS, bateria ou latência móvel.

Código próprio sob [MIT](LICENSE). O wrapper Gradle e as bibliotecas externas mantêm suas licenças; dados e modelos possuem condições independentes.
