package dev.izaak.portfolio.detector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var preview: PreviewView
    private lateinit var overlay: Overlay
    private lateinit var status: TextView
    private lateinit var modelStore: ModelStore
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    @Volatile private var detector: Detector? = null
    @Volatile private var destroyed = false
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else status.text = "Allow camera access to detect objects."
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modelStore = ModelStore(filesDir)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; fitsSystemWindows = true }
        status = TextView(this).apply { text = "Install the model once, then run detection offline."; setPadding(20,20,20,20) }
        root.addView(status)
        val download = Button(this).apply { text = "Prepare model (28 MB)" }
        root.addView(download)
        val frame = FrameLayout(this)
        preview = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FIT_CENTER; implementationMode = PreviewView.ImplementationMode.COMPATIBLE }
        overlay = Overlay(this)
        frame.addView(preview); frame.addView(overlay)
        root.addView(frame, LinearLayout.LayoutParams(-1,0,1f)); setContentView(root)
        download.setOnClickListener { download.isEnabled = false; prepareModel { download.isEnabled = true } }
        if (modelStore.exists()) prepareModel {}
    }
    private fun prepareModel(finished: () -> Unit) {
        status.text = "Preparing model…"
        worker.execute {
            try {
                val model = modelStore.prepare()
                detector?.close(); detector = null; detector = Detector(model)
                runOnUiThread { if(!destroyed) { status.text = "Model ready. Inference runs locally; images stay on this device."; finished(); requestCamera() } }
            } catch (error: Exception) {
                runOnUiThread { if(!destroyed) { status.text = error.message ?: "Could not prepare the model"; finished() } }
            }
        }
    }
    private fun requestCamera() {
        if(destroyed || !lifecycle.currentState.isAtLeast(androidx.lifecycle.Lifecycle.State.STARTED)) return
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
                val cameraPreview = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
                analysis?.clearAnalyzer()
                analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                analysis!!.setAnalyzer(worker) { image ->
                    try {
                        if(!destroyed) {
                            val start = System.nanoTime(); val values = detector?.detect(image) ?: floatArrayOf()
                            val milliseconds = (System.nanoTime()-start)/1e6
                            val rotated = image.imageInfo.rotationDegrees%180!=0
                            val w = if(rotated) image.cropRect.height() else image.cropRect.width()
                            val h = if(rotated) image.cropRect.width() else image.cropRect.height()
                            runOnUiThread { if(!destroyed) { overlay.update(values,w,h); status.text = "${values.size/6} objects • ${"%.1f".format(milliseconds)} ms • offline" } }
                        }
                    } catch(error: Exception) { runOnUiThread { if(!destroyed) status.text = "Inference failed: ${error.message}" } }
                    finally { image.close() }
                }
                provider!!.unbindAll()
                val viewport = preview.viewPort ?: throw IllegalStateException("Preview layout is not ready")
                val group = UseCaseGroup.Builder().addUseCase(cameraPreview).addUseCase(analysis!!).setViewPort(viewport).build()
                provider!!.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,group)
            } catch(error: Exception) { status.text = "Camera unavailable: ${error.message}" }
        },ContextCompat.getMainExecutor(this))
    }
    override fun onStart() { super.onStart(); if(::preview.isInitialized && detector!=null) requestCamera() }
    override fun onStop() { analysis?.clearAnalyzer(); provider?.unbindAll(); super.onStop() }
    override fun onDestroy() {
        destroyed = true; analysis?.clearAnalyzer(); provider?.unbindAll()
        worker.execute { detector?.close(); detector = null }; worker.shutdown(); super.onDestroy()
    }
}
