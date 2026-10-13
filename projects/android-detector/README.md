# Android offline object detector

Kotlin + CameraX + ONNX Runtime Android application, with YUV420 conversion, rotation, resizing, and filtering/NMS in C++20 through JNI. SSD MobileNet v1 opset 12 takes uint8 RGB NHWC `[1,300,300,3]` and produces boxes/classes/scores/num_detections. Inference stays on the device.

## Build

Requires JDK 17, Android SDK 35, NDK 27.0.12077973, and Android CMake 3.22.1. AGP 8.7.3, Gradle 8.9, and Kotlin 2.1.10 are pinned. The wrapper verifies the Gradle distribution's SHA hash.

```bash
# Android SDK installed; JAVA_HOME points to JDK 17
sdkmanager 'platforms;android-35' 'build-tools;35.0.0' 'ndk;27.0.12077973' 'cmake;3.22.1'
cd projects/android-detector
./gradlew --no-daemon --max-workers=2 assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The repository workflow provides the debug APK as the `android-detector-debug` artifact. It is a demonstration build, without production signing or Play Store publication.

## Usage

1. Open the application and tap **Prepare model (28 MB)**. The HTTPS download verifies SHA-256 and installs the file only after completion.
2. Grant camera access. The installed model is reused; there is no per-frame download.
3. Disable network/Wi-Fi to evaluate offline use. The application sends no images and uses no analytics.

CameraX drops old frames (`KEEP_ONLY_LATEST`); a single executor handles preprocessing, ORT, and postprocessing. `use` closes tensors/results, and `finally` closes each ImageProxy. The persistent session is closed when the Activity is destroyed. The camera is unbound when the Activity stops.

Preprocessing uses approximate limited-range BT.601 YUV conversion, 0/90/180/270-degree rotation, and nearest-neighbor resizing to 300×300. The overlay applies FIT_CENTER to rotated rear-camera frames. Initial configuration uses CPU, two threads, and a minimum score of 0.5; normalized boxes are filtered and limited to 30. Class numbers are the model's COCO IDs, rather than consecutive indices into an 80-name list.

## Evidence and limitations

The build checks Kotlin, resources, JNI integration, and linking for arm64-v8a/x86_64. Visual quality, preview synchronization, orientation changes, latency, heat, and battery use require evaluation on a physical device. The UI's `ms` value measures conversion + inference + filtering per processed frame; it is not camera FPS.

The APK downloads weights during initial preparation. The URL points to a public ONNX Model Zoo file, and the hash blocks unexpected changes. Fully offline use requires preparing the model beforehand. INTERNET permission is needed only for this preparation.

Models/data have their own terms: [licenses and provenance](../../docs/data-and-licenses.md).
