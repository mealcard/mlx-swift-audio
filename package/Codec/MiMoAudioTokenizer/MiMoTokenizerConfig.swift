// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation

/// Configuration for the MiMo-Audio-Tokenizer encoder.
///
/// Loaded from `MiMo-Audio-Tokenizer/config.json` (next to model.safetensors).
/// Mirrors the relevant subset of `MiMoAudioTokenizerConfig` from the
/// Python port (`mlx_audio/stt/models/mimo_v2_asr/audio_encoder.py:49-82`).
struct MiMoTokenizerConfig: Codable, Sendable {

    // MARK: - Conv front-end + transformer

    var nMels: Int = 128
    var dModel: Int = 1280
    var kernelSize: Int = 3
    var strideSize: Int = 2          // conv2 stride (2× downsample)
    var scaleEmbedding: Bool = false
    var encoderLayers: Int = 32
    /// 1-indexed layer whose hidden state is added back to the final hidden state.
    /// `nil` disables the skip connection.
    var encoderSkipLayerId: Int? = 3
    var encoderAttentionHeads: Int = 20
    var encoderFfnDim: Int = 5120
    var encoderCausal: Bool = false

    // MARK: - Down-sampling pool + LayerNorm after transformer

    var avgPooler: Int = 2

    // MARK: - RVQ

    var numQuantizers: Int = 20
    var codebookSize: Int = 1024

    // MARK: - RoPE

    var ropeTheta: Float = 10_000.0

    // MARK: - Audio

    var samplingRate: Int = 24_000
    var hopLength: Int = 240
    var maxAudioSeconds: Int = 1800

    var maxSourcePositions: Int {
        maxAudioSeconds * samplingRate / hopLength / strideSize
    }

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case dModel = "d_model"
        case kernelSize = "kernel_size"
        case strideSize = "stride_size"
        case scaleEmbedding = "scale_embedding"
        case encoderLayers = "encoder_layers"
        case encoderSkipLayerId = "encoder_skip_layer_id"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case encoderCausal = "encoder_causal"
        case avgPooler = "avg_pooler"
        case numQuantizers = "num_quantizers"
        case codebookSize = "codebook_size"
        case ropeTheta = "rope_theta"
        case samplingRate = "sampling_rate"
        case hopLength = "hop_length"
        case maxAudioSeconds = "max_audio_seconds"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        nMels = (try? c.decode(Int.self, forKey: .nMels)) ?? nMels
        dModel = (try? c.decode(Int.self, forKey: .dModel)) ?? dModel
        kernelSize = (try? c.decode(Int.self, forKey: .kernelSize)) ?? kernelSize
        strideSize = (try? c.decode(Int.self, forKey: .strideSize)) ?? strideSize
        scaleEmbedding = (try? c.decode(Bool.self, forKey: .scaleEmbedding)) ?? scaleEmbedding
        encoderLayers = (try? c.decode(Int.self, forKey: .encoderLayers)) ?? encoderLayers
        encoderSkipLayerId = try? c.decode(Int?.self, forKey: .encoderSkipLayerId)
        encoderAttentionHeads = (try? c.decode(Int.self, forKey: .encoderAttentionHeads)) ?? encoderAttentionHeads
        encoderFfnDim = (try? c.decode(Int.self, forKey: .encoderFfnDim)) ?? encoderFfnDim
        encoderCausal = (try? c.decode(Bool.self, forKey: .encoderCausal)) ?? encoderCausal
        avgPooler = (try? c.decode(Int.self, forKey: .avgPooler)) ?? avgPooler
        numQuantizers = (try? c.decode(Int.self, forKey: .numQuantizers)) ?? numQuantizers
        codebookSize = (try? c.decode(Int.self, forKey: .codebookSize)) ?? codebookSize
        ropeTheta = (try? c.decode(Float.self, forKey: .ropeTheta)) ?? ropeTheta
        samplingRate = (try? c.decode(Int.self, forKey: .samplingRate)) ?? samplingRate
        hopLength = (try? c.decode(Int.self, forKey: .hopLength)) ?? hopLength
        maxAudioSeconds = (try? c.decode(Int.self, forKey: .maxAudioSeconds)) ?? maxAudioSeconds
    }

    func encode(to encoder: Encoder) throws {
        // No-op; defined for Codable conformance only.
    }
}
