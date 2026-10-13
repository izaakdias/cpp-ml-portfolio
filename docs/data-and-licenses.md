# Dados, modelos e licenças

A licença MIT da implementação não muda as licenças dos dados, pesos ou bibliotecas. Nenhum dataset, livro PDF ou peso de modelo é incluído no Git. Os scripts verificam hashes publicados antes de preparar os arquivos locais.

| Material | Fonte e condições |
|---|---|
| Livro | [Repositório oficial da editora](https://github.com/PacktPublishing/Hands-on-Machine-learning-with-C-Second-Edition); inspiração conceitual, sem reprodução do texto/código |
| Wine, Auto MPG e Seeds | [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), [Auto MPG](https://archive.ics.uci.edu/dataset/9/auto+mpg), [Seeds](https://archive.ics.uci.edu/dataset/236/seeds); CC BY 4.0 conforme páginas UCI |
| MovieLens 100k | [GroupLens](https://grouplens.org/datasets/movielens/100k/); condições de uso de pesquisa e atribuição no README do arquivo. Não presumir licença MIT para os ratings. |
| NAB | [Numenta Anomaly Benchmark](https://github.com/numenta/NAB); arquivo `realKnownCause/machine_temperature_system_failure.csv` e janelas de `combined_windows.json`; consultar licenças/atribuição do repositório original |
| MNIST | [Página original](https://yann.lecun.com/exdb/mnist/); arquivos servidos pelo mirror OSSCI PyTorch. Atribuição: LeCun, Cortes e Burges. |
| DistilBERT SST-2 | [Model card oficial](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english); Apache-2.0 conforme card; revisão e ONNX/vocab/tokenizer fixados em `assets.lock.json` |
| ResNet18 | [TorchVision weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html); pesos ImageNet1K V1 oficiais, hash completo em `vision-assets.lock.json`. As condições dos dados ImageNet e dos pesos devem ser consideradas separadamente da licença do código TorchVision. |
| MVTec AD bottle | [Fonte original](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), **CC BY-NC-SA 4.0**, sem uso comercial; [mirror utilizado](https://huggingface.co/datasets/foersben/mvtec-ad), revisão `c75b39616f84db43677bcc8228caaafaf5096d7f`, hashes por imagem. Referência: Bergmann et al., MVTec AD, CVPR 2019 / IJCV 2021. O mirror não concede condições adicionais. |
| SSD MobileNet v1 | [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1); modelo opset 12 derivado de TensorFlow Object Detection, classes COCO. Consultar card e condições originais dos pesos/dados. |

## Dependências

- Eigen 3.4.0: MPL-2.0 e avisos por arquivo.
- nlohmann/json: MIT.
- cpp-httplib: MIT.
- Catch2: BSL-1.0.
- utf8proc: MIT e licença dos dados Unicode incluída no pacote.
- OpenSSL: Apache-2.0 para a versão usada localmente; consulte a distribuição do seu sistema.
- ONNX Runtime: MIT; distribuição inclui third-party notices.
- OpenCV 4.12: Apache-2.0 e avisos de componentes opcionais.
- PyTorch/LibTorch e TorchVision: BSD e third-party notices nos pacotes.
- Gradle wrapper: Apache-2.0; scripts/JAR são os originais da release 8.9, com hashes em `cmake/gradle-wrapper.lock.json`.
- AndroidX/CameraX: Apache-2.0.

Os manifestos guardam URLs, tamanho e SHA. Alterar uma versão demanda revisar procedência/condições e atualizar os locks. Pesos preparados a partir de MVTec AD são para o contexto acadêmico/portfólio indicado, respeitando as condições do dataset.
