// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © Anthony DePasquale (mlx-swift-audio Qwen2 reference)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen2 backbone for MiMo-V2.5-ASR with caller-selected attention mask mode.
///
/// Mirrors `TTS/Shared/CosyVoiceQwen2Backbone.swift` but exposes
/// `ScaledDotProductAttentionMaskMode` so the MiMo input-local transformer
/// can run with `.none` (full attention) and the backbone can run with
/// `.causal` for multi-token prefill while still flipping to `.none` for
/// cached single-token decode (codex pass 2 finding).
///
/// Default mask selection preserves CosyVoice's behavior: `L > 1 ? .causal :
/// .none`. Callers that want full attention must pass `.none` explicitly.
struct MiMoQwen2Config: Sendable {
    var hiddenSize: Int
    var numHiddenLayers: Int
    var intermediateSize: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var ropeTheta: Float

    /// Mimo backbone (`model.*` keys in safetensors).
    static let backbone = MiMoQwen2Config(
        hiddenSize: 4096,
        numHiddenLayers: 36,
        intermediateSize: 11008,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        headDim: 128,
        rmsNormEps: 1e-6,
        ropeTheta: 640_000.0
    )

    /// Mimo input-local transformer (`input_local_transformer.*`).
    static let inputLocal = MiMoQwen2Config(
        hiddenSize: 1024,
        numHiddenLayers: 6,
        intermediateSize: 4096,
        numAttentionHeads: 64,
        numKeyValueHeads: 64,
        headDim: 16,
        rmsNormEps: 1e-6,
        ropeTheta: 640_000.0
    )

    /// Mimo local transformer (`local_transformer.*`).
    static let local = MiMoQwen2Config(
        hiddenSize: 1024,
        numHiddenLayers: 16,
        intermediateSize: 4096,
        numAttentionHeads: 64,
        numKeyValueHeads: 64,
        headDim: 16,
        rmsNormEps: 1e-6,
        ropeTheta: 640_000.0
    )
}

// MARK: - Attention

class MiMoQwen2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let config: MiMoQwen2Config
    let scale: Float
    let rope: RoPE

    init(_ config: MiMoQwen2Config) {
        self.config = config
        self.scale = pow(Float(config.headDim), -0.5)

        let nHeads = config.numAttentionHeads
        let nKVHeads = config.numKeyValueHeads
        let headDim = config.headDim
        let hidden = config.hiddenSize

        // Qwen2 has biased Q/K/V; o_proj is biasless.
        _qProj.wrappedValue = Linear(hidden, nHeads * headDim, bias: true)
        _kProj.wrappedValue = Linear(hidden, nKVHeads * headDim, bias: true)
        _vProj.wrappedValue = Linear(hidden, nKVHeads * headDim, bias: true)
        _oProj.wrappedValue = Linear(nHeads * headDim, hidden, bias: false)

        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    /// - Parameters:
    ///   - x: input embeddings `(B, L, hidden)`.
    ///   - maskMode: SDPA mask mode. `nil` selects `L > 1 ? .causal : .none`.
    ///   - cache: optional KV cache (advanced in-place).
    func callAsFunction(
        _ x: MLXArray,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCacheSimple? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])
        let nHeads = config.numAttentionHeads
        let nKVHeads = config.numKeyValueHeads
        let headDim = config.headDim

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)
        keys = keys.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        values = values.reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Default mask selection preserves CosyVoice's behavior — needed
        // because the backbone instances of this class share the same call
        // pattern as CosyVoice (autoregressive single-token decode after
        // multi-token prefill).
        let effectiveMask: MLXFast.ScaledDotProductAttentionMaskMode = maskMode
            ?? (L > 1 ? .causal : .none)

        let out = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: effectiveMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped([B, L, -1])

        return oProj(out)
    }
}

// MARK: - MLP

class MiMoQwen2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

// MARK: - Transformer Block

class MiMoQwen2TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: MiMoQwen2Attention
    @ModuleInfo var mlp: MiMoQwen2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MiMoQwen2Config) {
        _attention.wrappedValue = MiMoQwen2Attention(config)
        _mlp.wrappedValue = MiMoQwen2MLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.intermediateSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: KVCacheSimple? = nil
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), maskMode: maskMode, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Stack with embed_tokens (full backbone)

/// Qwen2 model with token embedding. Used for the mimo backbone (key prefix
/// `model.*` in the safetensors).
class MiMoQwen2: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [MiMoQwen2TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    let config: MiMoQwen2Config

    init(_ config: MiMoQwen2Config, vocabSize: Int) {
        self.config = config

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: vocabSize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            MiMoQwen2TransformerBlock(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Forward with pre-computed embeddings (mimo-style; backbone gets fused
    /// text+speech embeddings from MiMoAudioMLX, never raw token IDs).
    ///
    /// - Parameters:
    ///   - inputEmbeddings: `(B, T, hidden)`.
    ///   - maskMode: SDPA mask mode for every layer (`nil` → `L > 1 ? .causal : .none`).
    ///   - cache: per-layer KV caches.
    /// - Returns: hidden states after the final RMSNorm, `(B, T, hidden)`.
    func forward(
        inputEmbeddings: MLXArray,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCacheSimple]?
    ) -> MLXArray {
        var h = inputEmbeddings
        for (i, layer) in layers.enumerated() {
            h = layer(h, maskMode: maskMode, cache: cache?[i])
        }
        return norm(h)
    }
}
