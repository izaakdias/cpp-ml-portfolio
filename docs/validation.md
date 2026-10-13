# Metodologia e evidências

## Execução de referência

macOS Apple Silicon, Apple Clang 21, CMake 4.4.1, Ninja 1.13.2, ONNX Runtime 1.31.0, LibTorch/PyTorch 2.14.1 e OpenCV 4.12.0. Seeds são fixadas, sem promessa de bitwise determinism entre compiladores/BLAS/plataformas. Os arquivos em [results](results/) guardam números da execução; versões baixadas e checksums estão nos manifests e `uv.lock`.

Nove aplicações desktop foram compiladas localmente. A suíte C++ inicial teve **26/26 casos aprovados** em Release. Inclui CSV malformed/quotes, finitude, métricas com ties, normalização constante, splits, modelos, persistência/corrupção/path traversal/escritas concorrentes, tensores ONNX reais, tokenização e protocolo REST MLflow. A CI compila novamente em Linux/Release e ASan/UBSan; leak detection está desabilitada pela interação com bibliotecas externas não instrumentadas.

## Protocolos de dados

| Dataset | Divisão / seleção | Métricas e cautelas |
|---|---|---|
| Wine | Classe 1 versus outras; 106 treino, 34 validação, 38 teste. Estratificação por classe; scaler no treino; L2, k e threshold no conjunto de validação. | F1/precision/recall/ROC-AUC/AP, maioria como baseline. kNN acertou o pequeno teste inteiro; isso não demonstra generalização universal. |
| Auto MPG | 392 linhas completas; 235/78/79, seed 41. Linhas sem horsepower são excluídas. Escalonamento somente treino; ridge escolhido por MAE de validação. | MAE/RMSE/R² versus média de treino. Não é uma previsão de preço nem uma estimativa causal do consumo. |
| Seeds | 210 linhas; exploração no dataset inteiro com scaler/PCA ajustados nesse conjunto. KMeans/GMM k=3 informado; DBSCAN eps=1,2/min_pts=5. | ARI compara assignments às variedades reais. DBSCAN padrão tem ARI próximo de zero: os grupos conectam por densidade. Não se escolheu epsilon pelo ARI do teste. |
| NAB | Série ordenada, deduplicada; partes cronológicas 60/20/20 **antes** de formar janelas de 24. Mean/std/last/range. Apenas janelas normais do treino ajustam o forest. | Limiar maximiza F1 em validação quando há as duas classes; fallback percentil 98 de normais. Teste independente tem 4.514 janelas. Rótulo positivo é qualquer ponto na janela dentro das janelas oficiais NAB. As métricas NÃO são o NAB score oficial nem medem lead time por incidente. Overlap de janelas gera amostras correlacionadas. |
| MovieLens 100k | Última interação por usuário retida; 99.057 ratings de treino e 943 de teste. 16 fatores, seed 61, 20 épocas. | RMSE/MAE/R² em ratings; HitRate e NDCG@10 em 486 retenções com rating ≥4, catálogo completo e exclusão de vistos. Há uma retenção por usuário: recall=hit por consulta. É holdout por usuário, não um corte temporal global; dados de outro usuário podem estar depois da retenção de um usuário. |
| MNIST | Permutação seed 41; 54.000 treino/6.000 validação. 5 épocas, batch 64, Adam 0,001. Melhor checkpoint por acurácia da validação. | Teste oficial de 10.000 imagens, uma avaliação após seleção. 98,52% de acurácia. LeNet é uma referência compacta, sem alegação de estado da arte. |
| MVTec AD bottle | 209 imagens normais de train divididas 167/42 com seed 41. Encoder ResNet18 ImageNet congelado. Mean/std por dimensão em treino; std mínimo 0,05. Limiar percentil 98 dos scores da validação normal. | 83 imagens oficiais test, 63 defeituosas e 20 normais. ROC-AUC/AP e F1 no limiar congelado. O método marca imagem inteira; não avalia segmentação/IoU de máscaras. |
| SST-2 pretrained | Pesos públicos DistilBERT congelados; tokenizer nativo e modelo exportado. Nenhum fine-tuning neste projeto. | Demonstração/paridade numérica. Não foi executada uma avaliação integral SST-2 nem de calibração. Inglês somente. |
| SSD COCO pretrained | Modelo público congelado; RGB uint8 NHWC e backend CPU Android. | Build/APK não mede qualidade de detecção/mAP. Não foi avaliado em aparelho/câmera reais nesta execução. |

## Paridade e desempenho

`tools/export_cnn.py` compara três imagens MNIST e registra o erro máximo: **0** entre os pesos reconstruídos em Python e C++, e aproximadamente **1,9×10⁻⁶** em ONNX. Esse resultado cobre essas imagens/arquitetura/versões; não é prova para todo modelo exportado.

`tools/evaluate_runtime.py` registra tokenização em exemplos de Unicode/truncamento, probabilidades DistilBERT, três imagens de inspeção e a regressão exportada com scaler. Os relatórios incluem tamanho da amostra e diferenças, sem usar paridade como substituta de avaliação da qualidade do modelo.

O benchmark da CLI inclui validação de JSON já carregado, alocação de tensors, inferência e cópia/serialização para objeto JSON. Exclui leitura de arquivo, inicialização de sessão, HTTP/rede e dump do JSON final. `p50/p95/p99` usa nearest-rank após warmup. A demonstração Auto MPG usa lote de três linhas, CPU/1 thread, 10 warmups e 200 repetições. **Não é uma comparação justa por si só de C++ versus Python**, nem uma promessa de SLA.

## Sistemas completos

- MLflow 3.17.0 local real: experimento criado, run aberto, métricas/params registrados e run finalizado. Os pesos ficaram no registro local conforme contrato do cliente.
- Streaming: série NAB inteira processada em JSONL, warmup e score por observação. O estado final preserva leituras, timestamp e run de modelo. Isso é replay, não dados novos recebidos de sensores.
- Visão: imagem defeituosa real processada pelo binário C++; anotação salva. A câmera macOS e a aplicação em dispositivo Android não foram executadas com hardware físico.
- APK e builds Linux: consulte o [workflow](https://github.com/izaakdias/cpp-ml-portfolio/actions/workflows/build.yml) para conclusão do commit exato. Um run antigo aprovado não garante um commit novo.

## Melhorias orientadas pelas medidas

O baseline de popularidade atingiu **HitRate@10 de 0,1111**, contra **0,0514** da fatoração. Para ratings, a fatoração teve RMSE **1,0092**, contra **1,2127** do baseline de média. Portanto, nesta configuração, melhorar a estimativa de rating não resultou em melhorar o ranking. O ranking usa score contínuo antes do clipping 1..5; a recomendação exibe score e rating limitado separadamente.

O recommender é uma base de ratings e seu ranking completo ainda é fraco. Compare com popularidade, adote protocolo temporal global e avalie BPR/ALS implícito antes de usar o sistema para recomendação comercial. O detector NAB favorece recall e gera falsos alertas; valide limiar, custo por incidente e política de cooldown em sensores reais. DBSCAN precisa de estudo de densidade/epsilon e pode exigir HDBSCAN. Estas limitações fazem parte do resultado publicado.
