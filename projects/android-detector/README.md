# Android offline object detector

App Kotlin + CameraX + ONNX Runtime Android, com conversão YUV420, rotação, resize e filtragem/NMS em C++20 via JNI. SSD MobileNet v1 opset 12 recebe uint8 RGB NHWC `[1,300,300,3]` e produz caixas/classes/scores/num_detections. A inferência permanece no aparelho.

## Compilar

JDK 17, Android SDK 35, NDK 27.0.12077973 e CMake Android 3.22.1. AGP 8.7.3, Gradle 8.9 e Kotlin 2.1.10 são fixados. O wrapper verifica o SHA da distribuição Gradle.

```bash
# Android SDK instalado, JAVA_HOME apontando para JDK 17
sdkmanager 'platforms;android-35' 'build-tools;35.0.0' 'ndk;27.0.12077973' 'cmake;3.22.1'
cd projects/android-detector
./gradlew --no-daemon --max-workers=2 assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

O workflow do repositório fornece o APK debug como artefato `android-detector-debug`. Não é uma release assinada de produção nem uma publicação na Play Store.

## Usar

1. Abra o app e toque **Preparar modelo (28 MB)**. O download HTTPS verifica SHA-256 e só promove o arquivo completo.
2. Autorize a câmera. O modelo instalado é reaproveitado; não há download por frame.
3. Desative rede/Wi-Fi para avaliar o uso offline. O app não envia imagens nem usa analytics.

CameraX descarta frames antigos (`KEEP_ONLY_LATEST`); um executor executa pré-processamento, ORT e pós-processamento. `use` fecha tensores/resultados e `finally` fecha cada ImageProxy. A sessão é persistente e fechada ao terminar a Activity. A câmera é desvinculada quando a Activity para.

Pré-processamento usa YUV limitado BT.601 aproximado, orientação 0/90/180/270 e nearest-neighbor para 300×300. O overlay aplica FIT_CENTER aos frames rotacionados da câmera traseira. A configuração inicial usa CPU, dois threads e score mínimo 0,5; caixas são normalizadas, filtradas e limitadas a 30. Os números de classe são IDs COCO do modelo, não índices consecutivos de uma lista de 80 nomes.

## Evidência e limites

O build verifica Kotlin, recursos, integração JNI e linkage das ABIs arm64-v8a/x86_64. A qualidade visual, sincronização de preview, troca de orientação, latência, aquecimento e bateria precisam ser avaliados em aparelho físico. `ms` na UI mede conversão + inferência + filtragem por frame processado; não é FPS da câmera.

O APK baixa os pesos na primeira preparação. O URL é de um arquivo público do ONNX Model Zoo e o hash bloqueia alterações inesperadas. Uma instalação totalmente sem rede exige preparar o modelo antes. Permissão INTERNET é necessária somente para essa preparação.

Modelo/dados têm condições próprias: [licenças e procedência](../../docs/data-and-licenses.md).
