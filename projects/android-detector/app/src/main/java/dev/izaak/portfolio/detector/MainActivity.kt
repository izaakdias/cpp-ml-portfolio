package dev.izaak.portfolio.detector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.Gravity
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var preview: PreviewView
    private lateinit var overlay: Overlay
    private lateinit var status: TextView
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var detector: Detector? = null
    @Volatile private var destroyed = false
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else status.text = "Permita o acesso à câmera para detectar objetos."
    }
    private val checksum = "b8fba5e404077d4048d27fcd1667e85e27e192eb9bf51e696c46a3acd7d21058"
    private val modelUrl = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; fitsSystemWindows = true }
        status = TextView(this).apply { text = "Instale o modelo uma vez; depois a detecção funciona offline."; setPadding(20,20,20,20) }
        root.addView(status)
        val download = Button(this).apply { text = "Preparar modelo (28 MB)" }
        root.addView(download)
        val frame = FrameLayout(this)
        preview = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER; implementationMode = PreviewView.ImplementationMode.COMPATIBLE }
        overlay = Overlay(this)
        frame.addView(preview); frame.addView(overlay)
        root.addView(frame, LinearLayout.LayoutParams(-1,0,1f)); setContentView(root)
        download.setOnClickListener { download.isEnabled = false; prepareModel { download.isEnabled = true } }
        if (File(filesDir,"ssd.onnx").exists()) prepareModel {}
    }
    private fun hash(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input -> val buffer = ByteArray(65536); while (true) { val n = input.read(buffer); if(n<0) break; digest.update(buffer,0,n) } }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
    private fun prepareModel(finished: () -> Unit) {
        status.text = "Preparando modelo…"
        worker.execute {
            try {
                val model = File(filesDir,"ssd.onnx")
                if (!model.exists() || hash(model) != checksum) {
                    val temporary = File(filesDir,"ssd.download")
                    val connection = URL(modelUrl).openConnection() as HttpURLConnection
                    connection.connectTimeout = 20000; connection.readTimeout = 60000
                    try {
                        require(connection.responseCode == 200) { "Falha no download HTTP ${connection.responseCode}" }
                        connection.inputStream.use { input -> temporary.outputStream().use { output ->
                            val buffer = ByteArray(65536); var total = 0
                            while(true) { val n = input.read(buffer); if(n<0) break; total += n; require(total<=31000000) { "Modelo maior que o limite" }; output.write(buffer,0,n) }
                        } }
                        require(hash(temporary) == checksum) { "Checksum do modelo inválido" }
                        check(temporary.renameTo(model)) { "Não foi possível instalar o modelo" }
                    } finally { connection.disconnect(); temporary.delete() }
                }
                detector?.close(); detector = Detector(model)
                runOnUiThread { if(!destroyed) { status.text = "Modelo pronto. Inferência local; imagens não são enviadas."; finished(); requestCamera() } }
            } catch (error: Exception) {
                runOnUiThread { if(!destroyed) { status.text = error.message ?: "Falha ao preparar modelo"; finished() } }
            }
        }
    }
    private fun requestCamera() {
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
    }
    private fun startCamera() {
        if(destroyed || detector==null || !lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) return
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if(destroyed || !lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) return@addListener
            try {
                provider = future.get()
                val cameraPreview = Preview.Builder().build().also { it.surfaceProvider = preview.surfaceProvider }
                analysis?.clearAnalyzer()
                analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                analysis!!.setAnalyzer(worker) { image ->
                    try {
                        if(!destroyed) {
                            val start = System.nanoTime(); val values = detector?.detect(image) ?: floatArrayOf()
                            val milliseconds = (System.nanoTime()-start)/1e6
                            val rotated = image.imageInfo.rotationDegrees%180!=0
                            val w = if(rotated) image.height else image.width; val h = if(rotated) image.width else image.height
                            runOnUiThread { if(!destroyed) { overlay.update(values,w,h); status.text = "${values.size/6} objetos • ${"%.1f".format(milliseconds)} ms • offline" } }
                        }
                    } catch(error: Exception) { runOnUiThread { if(!destroyed) status.text = "Inferência falhou: ${error.message}" } }
                    finally { image.close() }
                }
                provider!!.unbindAll()
                provider!!.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,cameraPreview,analysis)
            } catch(error: Exception) { status.text = "Câmera indisponível: ${error.message}" }
        },ContextCompat.getMainExecutor(this))
    }
    override fun onStart() { super.onStart(); if(::preview.isInitialized && detector!=null) requestCamera() }
    override fun onStop() { analysis?.clearAnalyzer(); provider?.unbindAll(); super.onStop() }
    override fun onDestroy() {
        destroyed = true; analysis?.clearAnalyzer(); provider?.unbindAll()
        worker.execute { detector?.close(); detector = null }; worker.shutdown(); super.onDestroy()
    }
}
