// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Sibling Qwen2 stack used for the mimo "input local" (6 L, full attention)
/// and "local" (16 L, causal) transformers.
///
/// Differs from `MiMoQwen2`:
/// - **No `embed_tokens` property** — the safetensors index does NOT contain
///   `input_local_transformer.embed_tokens.*` or `local_transformer.embed_tokens.*`.
///   This sibling class accepts only `inputEmbeddings:`. Codex pass 2 confirmed
///   `verify: [.all]` does not require parameters that aren't declared on the
///   module, so the absent embed weights cause no load failure.
/// - Otherwise architecturally identical to `MiMoQwen2` (same attention,
///   same MLP, same RMSNorm, same caller-selected mask mode).
class MiMoLocalQwen2: Module {
    @ModuleInfo var layers: [MiMoQwen2TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    let config: MiMoQwen2Config

    init(_ config: MiMoQwen2Config) {
        self.config = config

        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            MiMoQwen2TransformerBlock(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// - Parameters:
    ///   - inputEmbeddings: `(B, T, hidden)`.
    ///   - maskMode: SDPA mask mode.
    ///       - `.none` for input-local (full attention across 4-token groups).
    ///       - `nil` for local-decode (`L > 1 ? .causal : .none`).
    ///   - cache: per-layer KV caches.
    /// - Returns: `(B, T, hidden)`.
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
