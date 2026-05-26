// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXLMCommon

/// Token sampler with optional temperature / top-k / top-p filtering.
///
/// Mirrors `MiMoSampler` in the Python port (`model.py:114-182`):
/// - Text decoding uses the user-supplied temperature/top-k/top-p.
/// - Speech (local transformer) decoding uses greedy (argmax) with a
///   "forbid empty token" mask, instantiated with `doSample = false`.
///
/// Operates on logits of shape `(batch, vocabSize)`; returns int32 ids
/// of shape `(batch,)`.
struct MiMoSampler {
    var doSample: Bool
    var temperature: Float
    var topK: Int
    var topP: Float

    init(
        doSample: Bool = true,
        temperature: Float = 1.0,
        topK: Int = 0,
        topP: Float = 1.0
    ) {
        self.doSample = doSample
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }

    /// Sample one token per row from logits.
    ///
    /// - Parameters:
    ///   - scores: logits of shape `(batch, vocabSize)`.
    ///   - removedTokens: token IDs to mask to -inf before sampling.
    /// - Returns: int32 token IDs of shape `(batch,)`.
    func sample(_ scores: MLXArray, removedTokens: [Int] = []) -> MLXArray {
        var s = scores

        if !removedTokens.isEmpty {
            s = maskTokens(s, ids: removedTokens)
        }

        if !doSample || temperature == 0 {
            // Greedy with masking only.
            return MLX.argMax(s, axis: -1)
        }

        // Temperature.
        if temperature > 0 {
            s = s / temperature
        }

        // Top-k filter (set bottom (vocab - topK) tokens to -inf).
        if topK > 0 && topK < s.shape[1] {
            s = topKMask(s, k: topK)
        }

        // Top-p (nucleus) filter.
        if topP > 0.0 && topP < 1.0 {
            s = topPMask(s, p: topP)
        }

        // Categorical sample from filtered logits.
        // MLXRandom.categorical operates over the last axis by default.
        return MLXRandom.categorical(s)
    }

    // MARK: - Private helpers

    private func maskTokens(_ scores: MLXArray, ids: [Int]) -> MLXArray {
        let vocab = scores.shape[1]
        let allIds = MLXArray(0 ..< Int32(vocab))
        var maskExpr = MLXArray.zeros([vocab], dtype: .bool)
        for t in ids where t >= 0 && t < vocab {
            maskExpr = maskExpr .|| (allIds .== MLXArray(Int32(t)))
        }
        let neginf = MLXArray(-Float.infinity).asType(scores.dtype)
        return MLX.where(maskExpr, neginf, scores)
    }

    private func topKMask(_ scores: MLXArray, k: Int) -> MLXArray {
        // Threshold = kth-largest value per row.
        // argSort returns ascending order, so take index (-k).
        let sorted = MLX.takeAlong(scores, MLX.argSort(scores, axis: -1), axis: -1)
        let threshold = sorted[0..., sorted.shape[1] - k ..< sorted.shape[1] - k + 1]
        let keep = scores .>= threshold
        let neginf = MLXArray(-Float.infinity).asType(scores.dtype)
        return MLX.where(keep, scores, neginf)
    }

    private func topPMask(_ scores: MLXArray, p: Float) -> MLXArray {
        // Sort ascending (matches Python reference), compute softmax cumulative,
        // mark tokens whose cumulative probability is <= (1 - p) for removal.
        // The last (highest-prob) sorted token is always kept.
        let sortedIdx = MLX.argSort(scores, axis: -1)
        let sortedScores = MLX.takeAlong(scores, sortedIdx, axis: -1)
        let probs = MLX.softmax(sortedScores, axis: -1)
        let cumulative = MLX.cumsum(probs, axis: -1)

        var sortedRemove = cumulative .<= (1.0 - p)
        // Always keep the highest-prob token (last position in ascending order).
        // Force that column to false.
        let n = sortedRemove.shape[1]
        let positions = MLXArray(0 ..< Int32(n))
        let isLast = positions .== MLXArray(Int32(n - 1))
        sortedRemove = sortedRemove .&& .!(isLast)

        // Scatter mask back to original-order positions via inverse permutation.
        let inverseIdx = MLX.argSort(sortedIdx, axis: -1)
        let removeMask = MLX.takeAlong(sortedRemove, inverseIdx, axis: -1)

        let neginf = MLXArray(-Float.infinity).asType(scores.dtype)
        return MLX.where(removeMask, neginf, scores)
    }
}
