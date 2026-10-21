package org.stypox.dicio.io.wake.oww

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.FloatBuffer

class OwwModel(
    melSpectrogramPath: File,
    embeddingPath: File,
    wakeWordPath: File
) : AutoCloseable {
    private val melInterpreter: Interpreter
    private val embInterpreter: Interpreter
    private var wakeInterpreter: Interpreter? = null
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var onnxInputName: String? = null

    private var accumulatedMelOutputs: Array<Array<FloatArray>> = Array(EMB_INPUT_COUNT) { arrayOf() }
    private var accumulatedEmbOutputs: Array<FloatArray> = Array(WAKE_INPUT_COUNT) { floatArrayOf() }
    private var isClosed: Boolean = false

    init {
        melInterpreter = loadModel(melSpectrogramPath, intArrayOf(1, MEL_INPUT_COUNT))
        try {
            embInterpreter = loadModel(embeddingPath)
        } catch (t: Throwable) {
            melInterpreter.close()
            throw t
        }

        try {
            if (wakeWordPath.extension.equals("onnx", ignoreCase = true)) {
                val env = OrtEnvironment.getEnvironment()
                val session = env.createSession(wakeWordPath.absolutePath, OrtSession.SessionOptions())
                val inputName = session.inputNames.firstOrNull()
                    ?: throw IllegalArgumentException("ONNX wake-word model has no input")
                if (session.outputNames.isEmpty()) {
                    throw IllegalArgumentException("ONNX wake-word model has no output")
                }
                ortEnvironment = env
                ortSession = session
                onnxInputName = inputName
            } else {
                wakeInterpreter = loadModel(wakeWordPath)
            }
        } catch (t: Throwable) {
            melInterpreter.close()
            embInterpreter.close()
            ortSession?.close()
            throw t
        }
    }

    fun processFrame(audio: FloatArray): Float {
        synchronized(this) {
            if (isClosed) return 0.0f
            if (audio.size != MEL_INPUT_COUNT) {
                throw IllegalArgumentException("OwwModel can only process audio frames of $MEL_INPUT_COUNT samples")
            }

            val melOutput = Array(MEL_OUTPUT_COUNT) { FloatArray(MEL_FEATURE_SIZE) }
            melInterpreter.run(arrayOf(audio), arrayOf(arrayOf(melOutput)))
            for (i in 0..<EMB_INPUT_COUNT) {
                accumulatedMelOutputs[i] = if (i < EMB_INPUT_COUNT - MEL_OUTPUT_COUNT) {
                    accumulatedMelOutputs[i + MEL_OUTPUT_COUNT]
                } else {
                    melOutput[i - EMB_INPUT_COUNT + MEL_OUTPUT_COUNT]
                        .map { floatArrayOf((it / 10.0f) + 2.0f) }
                        .toTypedArray()
                }
            }
            if (accumulatedMelOutputs[0].isEmpty()) return 0.0f

            val embOutput = Array(EMB_OUTPUT_COUNT) { FloatArray(EMB_FEATURE_SIZE) }
            embInterpreter.run(arrayOf(accumulatedMelOutputs), arrayOf(arrayOf(embOutput)))
            for (i in 0..<WAKE_INPUT_COUNT) {
                accumulatedEmbOutputs[i] = if (i < WAKE_INPUT_COUNT - EMB_OUTPUT_COUNT) {
                    accumulatedEmbOutputs[i + EMB_OUTPUT_COUNT]
                } else {
                    embOutput[i - WAKE_INPUT_COUNT + EMB_OUTPUT_COUNT]
                }
            }
            if (accumulatedEmbOutputs[0].isEmpty()) return 0.0f

            val session = ortSession
            if (session != null) {
                val env = ortEnvironment ?: return 0.0f
                val flattened = FloatArray(WAKE_INPUT_COUNT * EMB_FEATURE_SIZE)
                for (i in 0..<WAKE_INPUT_COUNT) {
                    accumulatedEmbOutputs[i].copyInto(flattened, i * EMB_FEATURE_SIZE)
                }
                OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(flattened),
                    longArrayOf(1, WAKE_INPUT_COUNT.toLong(), EMB_FEATURE_SIZE.toLong())
                ).use { tensor ->
                    session.run(mapOf(onnxInputName!! to tensor)).use { result ->
                        val value = result[0].value
                        return when (value) {
                            is Array<*> -> {
                                val first = value.firstOrNull()
                                when (first) {
                                    is FloatArray -> first.firstOrNull() ?: 0.0f
                                    is Array<*> -> (first.firstOrNull() as? FloatArray)?.firstOrNull() ?: 0.0f
                                    else -> 0.0f
                                }
                            }
                            is FloatArray -> value.firstOrNull() ?: 0.0f
                            else -> 0.0f
                        }
                    }
                }
            }

            val wakeOutput = FloatArray(1)
            wakeInterpreter!!.run(arrayOf(accumulatedEmbOutputs), arrayOf(wakeOutput))
            return wakeOutput[0]
        }
    }

    override fun close() {
        synchronized(this) {
            isClosed = true
            melInterpreter.close()
            embInterpreter.close()
            wakeInterpreter?.close()
            ortSession?.close()
            // OrtEnvironment is a process-wide singleton; do not close it here.
        }
    }

    companion object {
        const val MEL_INPUT_COUNT = 512 + 160 * 4
        const val MEL_OUTPUT_COUNT = (MEL_INPUT_COUNT - 512) / 160 + 1
        const val MEL_FEATURE_SIZE = 32
        const val EMB_INPUT_COUNT = 76
        const val EMB_OUTPUT_COUNT = 1
        const val EMB_FEATURE_SIZE = 96
        const val WAKE_INPUT_COUNT = 16

        private fun loadModel(modelPath: File, inputDims: IntArray? = null): Interpreter {
            val interpreter = Interpreter(modelPath)
            if (inputDims != null) interpreter.resizeInput(0, inputDims)
            interpreter.allocateTensors()
            return interpreter
        }
    }
}
