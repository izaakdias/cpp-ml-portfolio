# Executar os projetos

Parta do quickstart do README e execute os comandos na raiz do repositório. Todos os binários aceitam `--help`; falhas produzem JSON em stderr e exit code 2.

## 1. ONNX inference engine

```bash
build/ml_baselines --task regression
uv run python tools/export_tabular.py
build/inference_engine --mode inspect
build/inference_engine --mode infer
build/inference_engine --mode benchmark --warmup 10 --runs 200
build/inference_engine --mode serve --workers 4
```

O modelo de demonstração é a regressão Auto MPG treinada em C++, com normalização incluída no grafo. Aceita lotes `[batch,7]`. Contrato HTTP: `{"inputs":{"features":{"dtype":"float32","shape":[1,7],"data":[...]}}}`. Consulte `models/linear-schema.json` para ordem das colunas. Também aceita tensores int64/uint8, usados por BERT/SSD.

## 2. Industrial vision inspector

```bash
uv run python tools/vision_model.py
build/vision_inspector --image data/mvtec/bottle/test/broken_large/000.png \
  --output artifacts/inspection.jpg
build/vision_inspector --camera 0 --frames 300
# Arquivo de vídeo em um formato que seu build OpenCV suporte:
build/vision_inspector --video sample.mp4 --headless --frames 100
```

O script baixa apenas a categoria bottle de um mirror fixado e verifica cada imagem. Ajusta mean/scale de embeddings ResNet18 em 167 imagens normais; 42 normais calibram o limiar. Defeitos de teste não entram no ajuste. O score marca a imagem inteira; não desenha uma localização fictícia do defeito. O build mínimo de OpenCV não inclui FFmpeg; codecs de vídeo dependem do backend do sistema.

## 3. Native sentiment service

```bash
uv run python tools/assets.py bert
build/sentiment_service --mode tokenize --text 'Café, excellent movie!'
build/sentiment_service --text 'This movie was excellent.'
build/sentiment_service --mode serve --port 8081
```

DistilBERT fine-tuned em SST-2, duas classes, **inglês**, sequence length 128. UTF-8, lowercase, NFD, retirada de acentos, pontuação/CJK, subpalavras, truncamento e padding são processados em C++. Probabilidade softmax não é garantia de calibração, explicação nem medida de qualidade em outro idioma.

## 4. Streaming anomaly monitor

```bash
build/anomaly_monitor
build/anomaly_monitor --mode replay
build/anomaly_monitor --mode stream --stream data/nab-stream.jsonl \
  --state artifacts/device-state > artifacts/events.jsonl
```

Cada linha: `{"device":"machine-1","timestamp_ms":123456789,"values":[83.2]}`. Após 24 leituras, gera score/alerta e distância à referência. Dispositivos têm janelas independentes. O checkpoint retém a versão do modelo e os últimos valores. Para retomar, forneça **somente timestamps posteriores** ao checkpoint; duplicatas/ordem regressiva são rejeitadas. Uma nova pasta `--state` inicia uma execução limpa. O replay em CSV pode incluir observações de treino; é demonstração de aplicação, não uma nova avaliação independente.

## 5. Native recommender

```bash
build/recommender --epochs 20 --user 1
build/recommender --mode recommend --user 1 --limit 10
```

MovieLens 100k, 16 fatores, SGD, vieses de usuário/item, regularização. A segunda execução recupera o snapshot e não retreina. Filmes já vistos são excluídos. Avaliação retém a interação mais recente de cada usuário; RMSE usa todas as 943 retenções e ranking usa as 486 com rating ≥4. Os IDs de filmes podem ser associados ao arquivo `u.item` do dataset. Não há API de catálogo fictícia nem treino online implícito.

## 6. Clustering explorer

```bash
build/clustering_explorer
build/clustering_explorer --epsilon 0.7 --minimum-points 5 --output artifacts/seeds-eps07.csv
uv run python tools/plot_clusters.py --output artifacts/clustering.png
```

CSV gerado tem pc1, pc2 e assignments dos três métodos; normalização ocorre antes das distâncias. `--clusters` controla KMeans/GMM e `--epsilon` afeta DBSCAN. O rótulo UCI entra somente no ARI. A configuração padrão DBSCAN não recupera bem as variedades — o resultado é publicado como evidência da sensibilidade do método. Os artefatos são assignments exploratórios, não um modelo universal de predição para pontos novos.

## 7. Tabular ML library

```bash
build/ml_baselines --task binary
build/ml_baselines --task regression
build/ml_baselines --data other.csv --task regression --seed 41
```

CSV numérico com cabeçalho único; a última coluna é o target. O treino/validação/teste padrão é 60/20/20, com arredondamento e estratificação por classe no caso binário. O target de Wine é classe 1 versus outras; a normalização usa somente treino. Ridge e L2/k são escolhidos em validação. Os parâmetros selecionados não são ajustados no teste nem retreinados nele.

## 8. Native CNN trainer

```bash
uv run python tools/assets.py mnist
build/native_cnn --epochs 5 --limit 60000
uv run python tools/export_cnn.py
build/inference_engine --model models/cnn/model.onnx --input models/cnn/input.json --mode infer
```

LeNet: conv 1→6→16 com kernels 5, average pooling, FC 256→120→84→10, ReLU, cross-entropy e Adam 0,001. 54.000 imagens treinam e 6.000 selecionam checkpoint; o teste oficial tem 10.000. IDX valida magic/dimensões/tamanho/classes. O exportador reconstrói os pesos nomeados do treino C++ e registra diferenças numéricas entre LibTorch, PyTorch e ONNX.

## 9. Native experiment tracker

```bash
uv sync --frozen --extra tracking
uv run mlflow server --backend-store-uri sqlite:///.cache/mlflow.db \
  --artifacts-destination .cache/mlflow-artifacts --host 127.0.0.1 --port 5000
# Outro terminal:
build/experiment_tracker --uri http://127.0.0.1:5000
```

O cliente C++ cria/busca experimento, inicia run, registra params/metrics e finaliza FINISHED/FAILED. O artefato local registra o run ID remoto. `MLFLOW_TRACKING_TOKEN` habilita Bearer header. Não há upload de pesos para MLflow neste cliente; eles permanecem no registro local. Não é uma substituição completa do SDK MLflow.

## 10. Android

[Guia próprio: preparar, compilar e instalar](../projects/android-detector/README.md).
