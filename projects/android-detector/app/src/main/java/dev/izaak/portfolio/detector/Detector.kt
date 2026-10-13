package dev.izaak.portfolio.detector

import ai.onnxruntime.*
import androidx.camera.core.ImageProxy
import java.io.File
import java.nio.ByteBuffer

object NativePixels {
    init { System.loadLibrary("detector_native") }
    external fun rgb(y: ByteBuffer, u: ByteBuffer, v: ByteBuffer, width: Int, height: Int,
                     left: Int, top: Int, right: Int, bottom: Int,
                     yr: Int, ur: Int, vr: Int, up: Int, vp: Int, rotation: Int): ByteArray
    external fun filter(boxes: FloatArray, classes: FloatArray, scores: FloatArray,
                        count: Int, threshold: Float): FloatArray
}

class Detector(model: File) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session = OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(2)
        environment.createSession(model.absolutePath, options)
    }
    fun detect(image: ImageProxy): FloatArray {
        require(image.planes.size == 3)
        val planes = image.planes
        val crop = image.cropRect
        val input = NativePixels.rgb(planes[0].buffer.slice(), planes[1].buffer.slice(), planes[2].buffer.slice(),
            image.width, image.height, crop.left, crop.top, crop.right, crop.bottom,
            planes[0].rowStride, planes[1].rowStride, planes[2].rowStride,
            planes[1].pixelStride, planes[2].pixelStride, image.imageInfo.rotationDegrees)
        OnnxTensor.createTensor(environment, ByteBuffer.wrap(input), longArrayOf(1,300,300,3), OnnxJavaType.UINT8).use { tensor ->
            session.run(mapOf("inputs" to tensor)).use { result ->
                fun array(name: String): FloatArray {
                    val value = result.get(name).orElseThrow() as OnnxTensor
                    val buffer = value.floatBuffer
                    return FloatArray(buffer.remaining()).also { buffer.get(it) }
                }
                val boxes = array("detection_boxes")
                val classes = array("detection_classes")
                val scores = array("detection_scores")
                val count = array("num_detections").first().toInt().coerceIn(0, scores.size)
                return NativePixels.filter(boxes, classes, scores, count, .5f)
            }
        }
    }
    override fun close() = session.close()
}
