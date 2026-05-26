// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation

/// MiMo-V2.5-ASR LLM-side configuration.
///
/// Mirrors the HuggingFace `MiMoAudioConfig` shipped in
/// `mlx-community/MiMo-V2.5-ASR-MLX/config.json`. Only fields the Swift port
/// consumes are decoded; everything else is ignored so future config additions
/// don't break loading.
struct MiMoAudioConfig: Codable, Sendable {

    // MARK: - Qwen2 LLM backbone

    var vocabSize: Int = 151_680
    var hiddenSize: Int = 4096
    var intermediateSize: Int = 11008
    var numHiddenLayers: Int = 36
    var numAttentionHeads: Int = 32
    var numKeyValueHeads: Int = 8
    var headDim: Int = 128
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 640_000.0
    var maxPositionEmbeddings: Int = 8192

    // MARK: - Speech

    var audioChannels: Int = 8
    var groupSize: Int = 4
    /// Per-channel speech vocab sizes. Encoded as "1025-1025-129-129-129-129-129-129"
    /// in HF config; decoded into `[Int]` of length `audioChannels`.
    var speechVocabSize: SpeechVocabField = .uniform(1025)
    /// Per-channel zero-embedding indices. Same encoding.
    var speechZeroembIdx: SpeechVocabField = .uniform(1024)
    /// Delay pattern for the local transformer's autoregressive decode loop.
    /// Length must equal `audioChannels`.
    var delayPattern: [Int] = [0, 1, 2, 3, 4, 5, 6, 7]
    var nRvq: Int = 20

    // MARK: - Input Local Transformer (full attention, 6 layers, d=1024)

    var inputLocalLayers: Int = 6
    var inputLocalDim: Int = 1024
    var inputLocalAttnHeads: Int = 64
    var inputLocalIntermediateSize: Int = 4096
    var inputLocalHeadDim: Int = 16
    var inputFullAttention: Bool = true

    // MARK: - Local Transformer (causal, 16 layers, d=1024)

    var localLayers: Int = 16
    var localDim: Int = 1024
    var localAttnHeads: Int = 64
    var localFfnDim: Int = 4096
    var localRotaryBase: Float = 640_000.0

    // MARK: - Special tokens (defaults from tokenizer_config.json)

    var eotIdx: Int = 151_672
    var sospIdx: Int = 151_665
    var eospIdx: Int = 151_666
    var eostmIdx: Int = 151_671
    var sostmIdx: Int = 151_670
    var speechlmIdx: Int = 151_669
    var emptyIdx: Int = 151_667

    // MARK: - Quantization (read from artifact, applied at load time)

    var quantization: QuantizationConfig? = nil

    // MARK: - Audio tokenizer config (subset, sometimes nested as `audio_config`)

    var audioConfig: NestedAudioConfig? = nil

    // MARK: - Decoded helpers

    var parsedSpeechVocabSizes: [Int] {
        speechVocabSize.values(channels: audioChannels)
    }

    var parsedSpeechEmptyIds: [Int] {
        speechZeroembIdx.values(channels: audioChannels)
    }

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"

        case audioChannels = "audio_channels"
        case groupSize = "group_size"
        case speechVocabSize = "speech_vocab_size"
        case speechZeroembIdx = "speech_zeroemb_idx"
        case delayPattern = "delay_pattern"
        case nRvq = "n_rvq"

        case inputLocalLayers = "input_local_layers"
        case inputLocalDim = "input_local_dim"
        case inputLocalAttnHeads = "input_local_attn_heads"
        case inputLocalIntermediateSize = "input_local_intermediate_size"
        case inputLocalHeadDim = "input_local_head_dim"
        case inputFullAttention = "input_full_attention"

        case localLayers = "local_layers"
        case localDim = "local_dim"
        case localAttnHeads = "local_attn_heads"
        case localFfnDim = "local_ffn_dim"
        case localRotaryBase = "local_rotary_base"

        case eotIdx = "eot_idx"
        case sospIdx = "sosp_idx"
        case eospIdx = "eosp_idx"
        case eostmIdx = "eostm_idx"
        case sostmIdx = "sostm_idx"
        case speechlmIdx = "speechlm_idx"
        case emptyIdx = "empty_idx"

        case quantization
        case quantizationConfig = "quantization_config"
        case audioConfig = "audio_config"
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        vocabSize = (try? c.decode(Int.self, forKey: .vocabSize)) ?? vocabSize
        hiddenSize = (try? c.decode(Int.self, forKey: .hiddenSize)) ?? hiddenSize
        intermediateSize = (try? c.decode(Int.self, forKey: .intermediateSize)) ?? intermediateSize
        numHiddenLayers = (try? c.decode(Int.self, forKey: .numHiddenLayers)) ?? numHiddenLayers
        numAttentionHeads = (try? c.decode(Int.self, forKey: .numAttentionHeads)) ?? numAttentionHeads
        numKeyValueHeads = (try? c.decode(Int.self, forKey: .numKeyValueHeads)) ?? numKeyValueHeads
        headDim = (try? c.decode(Int.self, forKey: .headDim)) ?? headDim
        rmsNormEps = (try? c.decode(Float.self, forKey: .rmsNormEps)) ?? rmsNormEps
        ropeTheta = (try? c.decode(Float.self, forKey: .ropeTheta)) ?? ropeTheta
        maxPositionEmbeddings = (try? c.decode(Int.self, forKey: .maxPositionEmbeddings)) ?? maxPositionEmbeddings

        audioChannels = (try? c.decode(Int.self, forKey: .audioChannels)) ?? audioChannels
        groupSize = (try? c.decode(Int.self, forKey: .groupSize)) ?? groupSize
        speechVocabSize = (try? c.decode(SpeechVocabField.self, forKey: .speechVocabSize)) ?? speechVocabSize
        speechZeroembIdx = (try? c.decode(SpeechVocabField.self, forKey: .speechZeroembIdx)) ?? speechZeroembIdx
        if let dp = try? c.decode([Int].self, forKey: .delayPattern) {
            delayPattern = dp
        } else if let dpStr = try? c.decode(String.self, forKey: .delayPattern) {
            delayPattern = dpStr.split(separator: "-").compactMap { Int($0) }
        }
        nRvq = (try? c.decode(Int.self, forKey: .nRvq)) ?? nRvq

        inputLocalLayers = (try? c.decode(Int.self, forKey: .inputLocalLayers)) ?? inputLocalLayers
        inputLocalDim = (try? c.decode(Int.self, forKey: .inputLocalDim)) ?? inputLocalDim
        inputLocalAttnHeads = (try? c.decode(Int.self, forKey: .inputLocalAttnHeads)) ?? inputLocalAttnHeads
        inputLocalIntermediateSize = (try? c.decode(Int.self, forKey: .inputLocalIntermediateSize)) ?? inputLocalIntermediateSize
        inputLocalHeadDim = (try? c.decode(Int.self, forKey: .inputLocalHeadDim)) ?? inputLocalHeadDim
        inputFullAttention = (try? c.decode(Bool.self, forKey: .inputFullAttention)) ?? inputFullAttention

        localLayers = (try? c.decode(Int.self, forKey: .localLayers)) ?? localLayers
        localDim = (try? c.decode(Int.self, forKey: .localDim)) ?? localDim
        localAttnHeads = (try? c.decode(Int.self, forKey: .localAttnHeads)) ?? localAttnHeads
        localFfnDim = (try? c.decode(Int.self, forKey: .localFfnDim)) ?? localFfnDim
        localRotaryBase = (try? c.decode(Float.self, forKey: .localRotaryBase)) ?? localRotaryBase

        eotIdx = (try? c.decode(Int.self, forKey: .eotIdx)) ?? eotIdx
        sospIdx = (try? c.decode(Int.self, forKey: .sospIdx)) ?? sospIdx
        eospIdx = (try? c.decode(Int.self, forKey: .eospIdx)) ?? eospIdx
        eostmIdx = (try? c.decode(Int.self, forKey: .eostmIdx)) ?? eostmIdx
        sostmIdx = (try? c.decode(Int.self, forKey: .sostmIdx)) ?? sostmIdx
        speechlmIdx = (try? c.decode(Int.self, forKey: .speechlmIdx)) ?? speechlmIdx
        emptyIdx = (try? c.decode(Int.self, forKey: .emptyIdx)) ?? emptyIdx

        // Quantization key is "quantization" in some artifacts, "quantization_config" in others.
        quantization = (try? c.decode(QuantizationConfig.self, forKey: .quantization))
            ?? (try? c.decode(QuantizationConfig.self, forKey: .quantizationConfig))

        audioConfig = try? c.decode(NestedAudioConfig.self, forKey: .audioConfig)
    }

    func encode(to encoder: Encoder) throws {
        // Encoding is not used; defined for Codable conformance only.
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
    }
}

/// Per-channel speech vocab / empty index field — either a string like
/// "1025-1025-129-129-129-129-129-129" or a single int that broadcasts.
enum SpeechVocabField: Codable, Sendable, Equatable {
    case uniform(Int)
    case perChannel([Int])

    init(from decoder: Decoder) throws {
        let v = try decoder.singleValueContainer()
        if let i = try? v.decode(Int.self) {
            self = .uniform(i)
        } else {
            let s = try v.decode(String.self)
            self = .perChannel(s.split(separator: "-").compactMap { Int($0) })
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .uniform(let i): try c.encode(i)
        case .perChannel(let arr): try c.encode(arr.map(String.init).joined(separator: "-"))
        }
    }

    func values(channels: Int) -> [Int] {
        switch self {
        case .uniform(let i): return Array(repeating: i, count: channels)
        case .perChannel(let arr): return arr
        }
    }
}

/// Mirrors `quantization_config` in mlx-community HF configs.
struct QuantizationConfig: Codable, Sendable {
    var groupSize: Int
    var bits: Int
    var mode: String

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }

    init(groupSize: Int = 64, bits: Int = 4, mode: String = "affine") {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }
}

/// Nested audio_config field present in some artifacts.
struct NestedAudioConfig: Codable, Sendable {
    var samplingRate: Int?
    var hopLength: Int?
    var nMels: Int?

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case hopLength = "hop_length"
        case nMels = "n_mels"
    }
}
