package dev.izaak.portfolio.detector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View
import kotlin.math.min

class Overlay(context: Context) : View(context) {
    private val stroke = Paint().apply { color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 4f }
    private val text = Paint().apply { color = Color.GREEN; textSize = 32f; isAntiAlias = true }
    private var detections = floatArrayOf()
    private var sourceWidth = 1
    private var sourceHeight = 1
    fun update(values: FloatArray, width: Int, height: Int) {
        detections = values; sourceWidth = width; sourceHeight = height; invalidate()
    }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scale = min(width.toFloat()/sourceWidth, height.toFloat()/sourceHeight)
        val w = sourceWidth*scale; val h = sourceHeight*scale
        val left = (width-w)/2; val top = (height-h)/2
        for (i in detections.indices step 6) {
            val x1 = left+detections[i+3]*w; val y1 = top+detections[i+2]*h
            canvas.drawRect(x1,y1,left+detections[i+5]*w,top+detections[i+4]*h,stroke)
            canvas.drawText("COCO ${detections[i].toInt()} • ${"%.0f".format(detections[i+1]*100)}%",x1,y1.coerceAtLeast(35f),text)
        }
    }
}
