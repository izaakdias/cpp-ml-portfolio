package dev.izaak.portfolio.detector

import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** Blocking model installation. Call on the single inference worker, never on the UI thread. */
class ModelStore(private val directory: File) {
    private val checksum = "b8fba5e404077d4048d27fcd1667e85e27e192eb9bf51e696c46a3acd7d21058"
    private val modelUrl = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx"
    private val model get() = File(directory, "ssd.onnx")

    fun exists(): Boolean = model.exists()

    fun prepare(): File {
        if (model.exists() && hash(model) == checksum) return model
        val temporary = File.createTempFile("ssd-", ".download", directory)
        val connection = URL(modelUrl).openConnection() as HttpURLConnection
        connection.connectTimeout = 20000
        connection.readTimeout = 60000
        try {
            require(connection.responseCode == 200) { "Falha no download HTTP ${connection.responseCode}" }
            connection.inputStream.use { input ->
                temporary.outputStream().use { output ->
                    val buffer = ByteArray(65536)
                    var total = 0
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        total += count
                        require(total <= 31000000) { "Modelo maior que o limite" }
                        output.write(buffer, 0, count)
                    }
                }
            }
            require(hash(temporary) == checksum) { "Checksum do modelo inválido" }
            check(temporary.renameTo(model)) { "Não foi possível instalar o modelo" }
            return model
        } finally {
            connection.disconnect()
            temporary.delete()
        }
    }

    private fun hash(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(65536)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }
}
