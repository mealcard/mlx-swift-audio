// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Pre-norm transformer block used inside the MiMo audio tokenizer encoder.
///
/// Architecture (port of `mlx_audio/stt/models/mimo_v2_asr/audio_encoder.py
/// :247-281`):
/// - `self_attn_layer_norm` (LayerNorm)
/// - `self_attn` (multi-head, full attention, RoPE on Q/K)
/// - residual
/// - `final_layer_norm` (LayerNorm)
/// - `fc1` (Linear d → ffn) → GELU → `fc2` (Linear ffn → d)
/// - residual
///
/// Weight key layout (HF-compatible, no remapping required):
///     self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}
///     self_attn_layer_norm.{weight,bias}
///     fc1.{weight,bias}
///     fc2.{weight,bias}
///     final_layer_norm.{weight,bias}
///
/// Note: `k_proj` has `bias=False`; all other projections have biases. The
/// safetensors index omits `k_proj.bias`, so the Linear is constructed with
/// `bias: false` (no zero-injection needed). See spec §5.2.
class MiMoEncoderAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(embedDim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue  = Linear(embedDim, embedDim, bias: true)
        _kProj.wrappedValue  = Linear(embedDim, embedDim, bias: false)
        _vProj.wrappedValue  = Linear(embedDim, embedDim, bias: true)
        _outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
    }

    /// - Parameters:
    ///   - x: `(seqLen, embedDim)` (no batch dim — encoder runs one utterance).
    ///   - ropeCos: `(seqLen, headDim)` cosine table.
    ///   - ropeSin: `(seqLen, headDim)` sine table.
    /// - Returns: `(seqLen, embedDim)`.
    func callAsFunction(
        _ x: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray
    ) -> MLXArray {
        let seqLen = x.shape[0]

        // Project. Shape: (seqLen, embedDim).
        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        // Reshape → (1, seqLen, heads, headDim).
        q = q.reshaped([1, seqLen, numHeads, headDim])
        k = k.reshaped([1, seqLen, numHeads, headDim])
        v = v.reshaped([1, seqLen, numHeads, headDim])

        // RoPE on Q and K. Add a head broadcast axis so cos/sin is
        // (seqLen, 1, headDim) and broadcasts across the heads.
        let cos = MLX.expandedDimensions(ropeCos, axis: 1)
        let sin = MLX.expandedDimensions(ropeSin, axis: 1)
        q = mimoApplyRotaryPosEmb(q, cos: cos, sin: sin)
        k = mimoApplyRotaryPosEmb(k, cos: cos, sin: sin)

        // (1, seqLen, heads, headDim) → (1, heads, seqLen, headDim).
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // Full attention (no mask, non-causal).
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale,
            mask: .none
        )

        // (1, heads, seqLen, headDim) → (1, seqLen, embedDim) → (seqLen, embedDim).
        let merged = attn.transposed(0, 2, 1, 3).reshaped([1, seqLen, numHeads * headDim])
        return outProj(merged)[0]
    }
}

/// Apply HF-compatible rotary position embeddings.
///
/// Given `x` of shape `(..., d)` and cos/sin tables of shape `(..., d)`,
/// returns `(x * cos) + (mimoRotateHalf(x) * sin)`.
func mimoApplyRotaryPosEmb(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    (x * cos) + (mimoRotateHalf(x) * sin)
}

private func mimoRotateHalf(_ x: MLXArray) -> MLXArray {
    let halfDim = x.shape.last! / 2
    let x1 = x[.ellipsis, 0 ..< halfDim]
    let x2 = x[.ellipsis, halfDim ..< 2 * halfDim]
    return MLX.concatenated([-x2, x1], axis: -1)
}

/// Pre-norm transformer block. Wraps Attention + GELU MLP.
class MiMoEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MiMoEncoderAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(dModel: Int, nHeads: Int, ffnDim: Int) {
        _selfAttn.wrappedValue = MiMoEncoderAttention(embedDim: dModel, numHeads: nHeads)
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        _fc1.wrappedValue = Linear(dModel, ffnDim, bias: true)
        _fc2.wrappedValue = Linear(ffnDim, dModel, bias: true)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
    }

    func callAsFunction(
        _ x: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray
    ) -> MLXArray {
        // Self-attention block.
        var h = selfAttn(selfAttnLayerNorm(x), ropeCos: ropeCos, ropeSin: ropeSin)
        h = x + h

        // FFN block.
        let normed = finalLayerNorm(h)
        let ff = fc2(MLXNN.gelu(fc1(normed)))
        return h + ff
    }
}

/// Precompute RoPE cos/sin tables.
///
/// - Returns: `(cos, sin)`, each shape `(seqLen, dim)`.
func mimoRotaryTables(base: Float, dim: Int, seqLen: Int, dtype: DType) -> (MLXArray, MLXArray) {
    let invFreq = MLXArray(
        (0 ..< dim / 2).map { i in
            1.0 / pow(base, 2.0 * Float(i) / Float(dim))
        }
    )                                                       // (dim/2,)
    let positions = MLXArray(0 ..< Int32(seqLen)).asType(.float32)   // (seqLen,)
    // (dim/2, 1) @ (1, seqLen) → (dim/2, seqLen) → transposed → (seqLen, dim/2)
    let freqs = MLX.matmul(
        MLX.expandedDimensions(invFreq, axis: 1),
        MLX.expandedDimensions(positions, axis: 0)
    ).T                                                     // (seqLen, dim/2)
    let emb = MLX.concatenated([freqs, freqs], axis: -1)    // (seqLen, dim)
    return (MLX.cos(emb).asType(dtype), MLX.sin(emb).asType(dtype))
}
