package org.stypox.dicio.io.wake.oww

import android.content.Context
import android.provider.OpenableColumns
import android.net.Uri
import android.util.Log
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import org.stypox.dicio.io.wake.WakeDevice
import org.stypox.dicio.io.wake.WakeState
import org.stypox.dicio.ui.util.Progress
import org.stypox.dicio.util.FileToDownload
import org.stypox.dicio.util.downloadBinaryFilesWithPartial
import java.io.File
import java.io.IOException

class OpenWakeWordDevice(
    @param:ApplicationContext private val appContext: Context,
    private val okHttpClient: OkHttpClient,
) : WakeDevice {
    private val _state: MutableStateFlow<WakeState>
    override val state: StateFlow<WakeState>

    private val cacheDir: File = appContext.cacheDir
    private val owwFolder = File(appContext.filesDir, "openWakeWord")
    private val melFile = FileToDownload(MEL_URL, File(owwFolder, "melspectrogram.tflite"))
    private val embFile = FileToDownload(EMB_URL, File(owwFolder, "embedding.tflite"))
    private val wakeFile = FileToDownload(WAKE_URL, File(owwFolder, "wake.tflite"))
    // Resolve the custom model dynamically so importing it while the app is open takes effect.
    private val allModelFiles = listOf(melFile, embFile, wakeFile)

    private val audio = FloatArray(OwwModel.MEL_INPUT_COUNT)
    private var model: OwwModel? = null

    private val scope = CoroutineScope(Dispatchers.IO)

    init {
        _state = if (allModelFiles.any(FileToDownload::needsToBeDownloaded)) {
            MutableStateFlow(WakeState.NotDownloaded)
        } else {
            MutableStateFlow(WakeState.NotLoaded)
        }
        state = _state
    }

    override fun download() {
        _state.value = WakeState.Downloading(Progress.UNKNOWN)

        scope.launch {
            try {
                owwFolder.mkdirs()
                downloadBinaryFilesWithPartial(
                    urlsFiles = allModelFiles,
                    httpClient = okHttpClient,
                    cacheDir = cacheDir,
                ) { progress ->
                    _state.value = WakeState.Downloading(progress)
                }
            } catch (e: Throwable) {
                Log.e(TAG, "Can't download OpenWakeWord model", e)
                _state.value = WakeState.ErrorDownloading(e)
                return@launch
            }

            _state.value = WakeState.NotLoaded
        }
    }

    override fun processFrame(audio16bitPcm: ShortArray): Boolean {
        if (audio16bitPcm.size != OwwModel.MEL_INPUT_COUNT) {
            throw IllegalArgumentException(
                "OwwModel can only process audio frames of ${OwwModel.MEL_INPUT_COUNT} samples"
            )
        }

        if (model == null) {
            if (_state.value.let { it != WakeState.NotLoaded && it !is WakeState.ErrorLoading }) {
                throw IOException("Model has not been downloaded yet")
            }

            try {
                _state.value = WakeState.Loading
                model = OwwModel(
                    melFile.file,
                    embFile.file,
                    userWakeModelFile(appContext) ?: wakeFile.file,
                )
                _state.value = WakeState.Loaded
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to load model", t)
                _state.value = WakeState.ErrorLoading(t)
                throw t
            }
        }

        for (i in 0..<OwwModel.MEL_INPUT_COUNT) {
            audio[i] = audio16bitPcm[i].toFloat() / 32768.0f
        }

        // The FALKOR Colab trainer exports sigmoid probabilities and validates at 0.5.
        // Keep the original Dicio threshold for legacy TFLite models.
        val activeWakeFile = userWakeModelFile(appContext) ?: wakeFile.file
        val threshold = if (activeWakeFile.extension.equals("onnx", ignoreCase = true)) 0.5f else 0.8f
        return model!!.processFrame(audio) > threshold
    }

    override fun frameSize(): Int {
        return OwwModel.MEL_INPUT_COUNT
    }

    override fun isOccupyingResources(): Boolean = model != null

    override fun destroy() {
        model?.close()
        model = null
        scope.cancel()
    }

    override fun isHeyDicio(): Boolean = userWakeModelFile(appContext) == null

    companion object {
        val TAG = OpenWakeWordDevice::class.simpleName
        const val MEL_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite"
        const val EMB_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite"
        const val WAKE_URL = "https://github.com/Stypox/dicio-android/releases/download/v2.0/hey_dicio_v6.0.tflite"

        private fun userWakeTfliteFile(context: Context) =
            File(context.filesDir, "openWakeWord/userwake.tflite")

        private fun userWakeOnnxFile(context: Context) =
            File(context.filesDir, "openWakeWord/userwake.onnx")

        private fun userWakeModelFile(context: Context): File? {
            val onnx = userWakeOnnxFile(context)
            if (onnx.exists()) return onnx
            val tflite = userWakeTfliteFile(context)
            return if (tflite.exists()) tflite else null
        }

        suspend fun addUserWakeFile(context: Context, source: Uri) {
            withContext(Dispatchers.IO) {
                val displayName = context.contentResolver.query(
                    source, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null
                )?.use { cursor ->
                    if (cursor.moveToFirst()) cursor.getString(0) else null
                } ?: source.lastPathSegment.orEmpty()

                val extension = displayName.substringAfterLast('.', "").lowercase()
                if (extension != "onnx" && extension != "tflite") {
                    throw IOException("Unsupported wake-word model. Choose an .onnx or .tflite file.")
                }

                val target = if (extension == "onnx") userWakeOnnxFile(context) else userWakeTfliteFile(context)
                val other = if (extension == "onnx") userWakeTfliteFile(context) else userWakeOnnxFile(context)
                val folder = target.parentFile ?: throw IOException("Invalid model folder")
                if (!folder.exists() && !folder.mkdirs()) throw IOException("Cannot create model folder")
                val partialFile = File.createTempFile(target.name, ".part", context.cacheDir)
                try {
                    val inputStream = context.contentResolver.openInputStream(source)
                        ?: throw IOException("Cannot open selected model file")
                    inputStream.use { input ->
                        partialFile.outputStream().use { output -> input.copyTo(output) }
                    }
                    if (target.exists() && !target.delete()) throw IOException("Cannot replace previous model")
                    if (!partialFile.renameTo(target)) {
                        throw IOException("Cannot move imported model into place")
                    }
                    other.delete()
                } finally {
                    partialFile.delete()
                }
            }
        }

        suspend fun removeUserWakeFile(context: Context) {
            withContext(Dispatchers.IO) {
                userWakeOnnxFile(context).delete()
                userWakeTfliteFile(context).delete()
            }
        }
    }
}
