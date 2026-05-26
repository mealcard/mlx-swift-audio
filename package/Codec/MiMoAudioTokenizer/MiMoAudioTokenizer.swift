// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Top-level MiMo audio tokenizer encoder.
///
/// Mel (128, T) → Conv1d(128→1280, k=3, pad=1) → GELU
///              → Conv1d(1280→1280, k=3, stride=2, pad=1) → GELU
///              → 32 × MiMoEncoderLayer (RoPE, full attention)
///              → skip-connect layer 3 output
///              → LayerNorm
///              → optional Conv1d(k=2, stride=2, bias=false) + GELU + LayerNorm   [avg_pooler=2]
///              → RVQ.encode → (n_q, T_tokens) int32
///
/// Port of `mlx_audio/stt/models/mimo_v2_asr/audio_encoder.py:284-427`.
class MiMoAudioTokenizer: Module {
    let config: MiMoTokenizerConfig

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo var layers: [MiMoEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "down_sample") var downSample: Conv1d?
    @ModuleInfo(key: "down_sample_norm") var downSampleNorm: LayerNorm?
    @ModuleInfo var quantizer: MiMoResidualVectorQuantizer

    /// 1-indexed skip-connection layer (set in init from config).
    let skipLayerIdx: Int?

    /// Embed scale (= sqrt(d_model) when scale_embedding, else 1.0).
    let embedScale: Float

    /// Cached RoPE tables. Lazy-built on first call sized to the longest
    /// sequence we've seen; subsequent calls slice to actual length.
    private var cachedRopeLen: Int = 0
    private var cachedRopeCos: MLXArray = MLXArray.zeros([0])
    private var cachedRopeSin: MLXArray = MLXArray.zeros([0])

    init(config: MiMoTokenizerConfig) {
        self.config = config
        self.skipLayerIdx = config.encoderSkipLayerId
        self.embedScale = config.scaleEmbedding ? sqrt(Float(config.dModel)) : 1.0

        _conv1.wrappedValue = Conv1d(
            inputChannels: config.nMels,
            outputChannels: config.dModel,
            kernelSize: config.kernelSize,
            stride: 1,
            padding: 1,
            bias: true
        )
        _conv2.wrappedValue = Conv1d(
            inputChannels: config.dModel,
            outputChannels: config.dModel,
            kernelSize: config.kernelSize,
            stride: config.strideSize,
            padding: 1,
            bias: true
        )

        _layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in
            MiMoEncoderLayer(
                dModel: config.dModel,
                nHeads: config.encoderAttentionHeads,
                ffnDim: config.encoderFfnDim
            )
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)

        if config.avgPooler > 1 {
            _downSample.wrappedValue = Conv1d(
                inputChannels: config.dModel,
                outputChannels: config.dModel,
                kernelSize: config.avgPooler,
                stride: config.avgPooler,
                padding: 0,
                bias: false
            )
            _downSampleNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        } else {
            _downSample.wrappedValue = nil
            _downSampleNorm.wrappedValue = nil
        }

        _quantizer.wrappedValue = MiMoResidualVectorQuantizer(
            dimension: config.dModel,
            nQ: config.numQuantizers,
            codebookSize: config.codebookSize
        )
    }

    // MARK: - Forward

    /// - Parameter mel: log-mel spectrogram, shape `(nMels, melLen)`.
    /// - Returns: hidden features `(T_out, d_model)`.
    func features(_ mel: MLXArray) -> MLXArray {
        // NLC convention: (1, melLen, nMels).
        var x = MLX.expandedDimensions(mel.T, axis: 0)

        // conv1: (1, melLen, d_model)
        x = MLXNN.gelu(conv1(x))
        // conv2 stride 2: (1, ⌈melLen/2⌉, d_model)
        x = MLXNN.gelu(conv2(x))

        // Drop batch axis: (T', d_model).
        x = (x * embedScale)[0]
        let seqLen = x.shape[0]

        // RoPE cos/sin tables sized to seqLen.
        let headDim = config.dModel / config.encoderAttentionHeads
        let (cos, sin) = ensureRopeTables(
            length: seqLen,
            headDim: headDim,
            dtype: x.dtype
        )

        // Transformer stack with skip connection from layer `skipLayerIdx - 1`.
        var skipHidden: MLXArray? = nil
        for (i, layer) in layers.enumerated() {
            x = layer(x, ropeCos: cos, ropeSin: sin)
            if let s = skipLayerIdx, i == s - 1 {
                skipHidden = x
            }
        }
        if let s = skipHidden {
            x = x + s
        }

        x = layerNorm(x)

        // Optional down-sampling: pad → Conv1d → GELU → LayerNorm.
        if let conv = downSample, let norm = downSampleNorm {
            let pool = config.avgPooler
            let T = x.shape[0]
            if T % pool != 0 {
                let padLen = pool - (T % pool)
                let padding = MLXArray.zeros([padLen, x.shape[1]]).asType(x.dtype)
                x = MLX.concatenated([x, padding], axis: 0)
            }
            // NLC for Conv1d: (1, T', d_model).
            var y = MLX.expandedDimensions(x, axis: 0)
            y = MLXNN.gelu(conv(y))
            x = y[0]
            x = norm(x)
        }

        return x
    }

    /// Encode a mel spectrogram to RVQ codes.
    ///
    /// - Parameters:
    ///   - mel: log-mel spectrogram `(nMels, melLen)`.
    ///   - nQ: number of quantizer channels to use (default: `audio_channels = 8`).
    /// - Returns: int32 codes `(nQ, T_out)`.
    func encode(_ mel: MLXArray, nQ: Int? = nil) -> MLXArray {
        let h = features(mel)
        return quantizer.encode(h, nQ: nQ)
    }

    // MARK: - Sanitization (called at weight-load time)

    /// Sanitize HF safetensors keys for the encoder. Implements the rules from
    /// `_sanitize_audio_encoder_weights` (Python `asr.py:125-162`) plus the
    /// rev-3 additions: drop `position_embedding.inv_freq` (we recompute it),
    /// no `k_proj.bias` zero-injection.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]

        for (key, tensor) in weights {
            // Decoder weights are not used at inference.
            if key.hasPrefix("decoder.") { continue }

            var newKey = key
            // Strip "encoder." prefix.
            if newKey.hasPrefix("encoder.") {
                newKey = String(newKey.dropFirst("encoder.".count))
            }

            // Drop RoPE inv_freq (recomputed on demand).
            if newKey.hasSuffix("position_embedding.inv_freq") { continue }

            // EMA / training state (not present at inference).
            if newKey.contains(".cluster_size") ||
               newKey.contains(".embed_avg") ||
               newKey.contains(".inited") {
                continue
            }

            // Rename: down_sample_layer.0 → down_sample (HF stores as Sequential[Conv1d, GELU]).
            if newKey.contains("down_sample_layer.0") {
                newKey = newKey.replacingOccurrences(
                    of: "down_sample_layer.0",
                    with: "down_sample"
                )
            }

            // Rename: _codebook. → codebook.
            if newKey.contains("_codebook.") {
                newKey = newKey.replacingOccurrences(
                    of: "_codebook.",
                    with: "codebook."
                )
            }

            // Conv1d NCL→NLC: 3-D weight tensors with last dim small need
            // transpose (0, 2, 1) so MLX consumes them as (out, kernel, in).
            var t = tensor
            if newKey.hasSuffix(".weight"),
               t.ndim == 3,
               t.shape[2] <= 8,
               t.shape[1] > t.shape[2] {
                t = t.transposed(0, 2, 1)
            }

            out[newKey] = t
        }

        return out
    }

    // MARK: - Private

    private func ensureRopeTables(length: Int, headDim: Int, dtype: DType) -> (MLXArray, MLXArray) {
        if length <= cachedRopeLen {
            return (
                cachedRopeCos[0 ..< length],
                cachedRopeSin[0 ..< length]
            )
        }
        let (cos, sin) = mimoRotaryTables(
            base: config.ropeTheta,
            dim: headDim,
            seqLen: length,
            dtype: dtype
        )
        cachedRopeLen = length
        cachedRopeCos = cos
        cachedRopeSin = sin
        return (cos, sin)
    }
}
