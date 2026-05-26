// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Top-level MiMo-V2.5-ASR LLM. Combines:
/// - 8 per-channel speech embeddings (sum across channels per group)
/// - 6-layer input local transformer (full attention)
/// - speech_group_downcast: 1024 * 4 → 4096
/// - 36-layer Qwen2 backbone (text + fused speech embeddings)
/// - hidden_states_downcast: 4096 → 1024
/// - 16-layer local transformer (causal) with delay-pattern decode
/// - 8 per-channel speech LM heads
/// - text lm_head (4096 → vocab)
///
/// Port of `mlx_audio/stt/models/mimo_v2_asr/model.py:185-617` (MiMoAudioMLX).
class MiMoAudioMLX: Module {
    // Backbone (registered as "model" to match safetensors `model.*` prefix).
    @ModuleInfo(key: "model") var qwen2Backbone: MiMoQwen2
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    // Speech embeddings: 8 per-channel codebook embeddings, summed in
    // _prepareInputEmbeds.
    @ModuleInfo(key: "speech_embeddings") var speechEmbeddings: [Embedding]

    // Input-local transformer (6 layers, FULL attention via MaskMode.none).
    @ModuleInfo(key: "input_local_transformer") var inputLocalTransformer: MiMoLocalQwen2

    // Down/up casts bridging 4096 ↔ 1024.
    @ModuleInfo(key: "speech_group_downcast") var speechGroupDowncast: Linear
    @ModuleInfo(key: "hidden_states_downcast") var hiddenStatesDowncast: Linear

    // Local transformer for delay-pattern speech-token generation
    // (16 layers, causal).
    @ModuleInfo(key: "local_transformer") var localTransformer: MiMoLocalQwen2
    @ModuleInfo(key: "local_transformer_lm_heads") var localTransformerLmHeads: [Linear]

    let config: MiMoAudioConfig
    let speechEmptyIds: [Int]
    let speechVocabSizes: [Int]
    let delayPattern: [Int]
    let groupSize: Int
    let audioChannels: Int

    init(_ config: MiMoAudioConfig) {
        self.config = config
        self.speechEmptyIds = config.parsedSpeechEmptyIds
        self.speechVocabSizes = config.parsedSpeechVocabSizes
        self.delayPattern = config.delayPattern
        self.groupSize = config.groupSize
        self.audioChannels = config.audioChannels

        // Backbone — note key "model" matches safetensors prefix.
        _qwen2Backbone.wrappedValue = MiMoQwen2(.backbone, vocabSize: config.vocabSize)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)

        // 8 per-channel speech embeddings.
        var speechEmbs: [Embedding] = []
        for i in 0 ..< config.audioChannels {
            speechEmbs.append(
                Embedding(
                    embeddingCount: speechVocabSizes[i],
                    dimensions: config.inputLocalDim
                )
            )
        }
        _speechEmbeddings.wrappedValue = speechEmbs

        // Input local transformer (full attention).
        _inputLocalTransformer.wrappedValue = MiMoLocalQwen2(.inputLocal)

        // Bridge projections.
        _speechGroupDowncast.wrappedValue = Linear(
            config.inputLocalDim * config.groupSize,
            config.hiddenSize,
            bias: false
        )
        _hiddenStatesDowncast.wrappedValue = Linear(
            config.hiddenSize,
            config.localDim,
            bias: false
        )

        // Local transformer (causal).
        _localTransformer.wrappedValue = MiMoLocalQwen2(.local)

        // 8 per-channel speech LM heads.
        var lmHeads: [Linear] = []
        for i in 0 ..< config.audioChannels {
            lmHeads.append(Linear(config.localDim, speechVocabSizes[i], bias: false))
        }
        _localTransformerLmHeads.wrappedValue = lmHeads
    }

    // MARK: - Input embedding preparation

    /// Build the combined (text + speech) input embeddings for the backbone.
    ///
    /// Mirrors `_prepare_input_embeds` in the Python reference
    /// (`model.py:305-380`).
    ///
    /// - Parameter inputIds: `(B, audio_channels + 1, T)` int32. Row 0 is
    ///   text tokens (every `group_size`-th cell is a real token; others are
    ///   filler `empty_idx`); rows 1..audio_channels are per-channel speech
    ///   token IDs.
    /// - Returns: `(B, T_groups, hiddenSize)` ready for the backbone.
    func prepareInputEmbeds(_ inputIds: MLXArray) -> MLXArray {
        let B = inputIds.shape[0]
        let ac = audioChannels
        let gs = groupSize
        let totalT = inputIds.shape[2]
        let tGroups = totalT / gs

        // Text IDs from row 0, every gs-th column.
        let textInputIds = inputIds[0..., 0, .stride(from: 0, to: totalT, by: gs)]
        // Speech IDs from rows 1..1+ac, reshaped to (B, T_g, ac, gs).
        let speechAll = inputIds[0..., 1 ..< (1 + ac), 0...]    // (B, ac, T)
            .reshaped([B, ac, tGroups, gs])
            .transposed(0, 2, 1, 3)                              // (B, T_g, ac, gs)

        // is_speech mask per group (true where text token is empty).
        let isSpeech = (textInputIds .== MLXArray(Int32(config.emptyIdx)))   // (B, T_g) bool

        // Sum embeddings across 8 channels, masking each channel's empty positions.
        var acc = MLXArray.zeros(
            [B, tGroups, gs, config.inputLocalDim]
        ).asType(.float32)

        for i in 0 ..< ac {
            let empty = Int32(speechEmptyIds[i])
            let ids = speechAll[0..., 0..., i, 0...]              // (B, T_g, gs)
            let emb = speechEmbeddings[i](ids)                    // (B, T_g, gs, inputLocalDim)
            let emptyMask = (ids .== MLXArray(empty))             // (B, T_g, gs)
            let masked = MLX.where(
                MLX.expandedDimensions(emptyMask, axis: -1),
                MLXArray.zeros(emb.shape).asType(emb.dtype),
                emb
            )
            acc = acc + masked.asType(acc.dtype)
        }

        // Zero out non-speech text positions.
        let isSpeechFloat = isSpeech.asType(acc.dtype)
        acc = acc * MLX.expandedDimensions(
            MLX.expandedDimensions(isSpeechFloat, axis: -1),
            axis: -1
        )

        // Input-local transformer: reshape (B, T_g, gs, hid) → (B*T_g, gs, hid),
        // full attention, reshape back.
        let inputLocalHidden = inputLocalTransformer.forward(
            inputEmbeddings: acc.reshaped([B * tGroups, gs, config.inputLocalDim]),
            maskMode: .none,
            cache: nil
        )
        acc = inputLocalHidden.reshaped([B, tGroups, gs, config.inputLocalDim])
        acc = acc * MLX.expandedDimensions(
            MLX.expandedDimensions(isSpeechFloat, axis: -1),
            axis: -1
        )

        // speech_group_downcast: flatten group_size axis (B, T_g, gs * hid) → (B, T_g, 4096).
        let flat = acc.reshaped([B, tGroups, gs * config.inputLocalDim])
        let speechEmbeds = speechGroupDowncast(flat)

        // Text embeddings — zero out empty positions.
        var textEmbeds = qwen2Backbone.embedTokens(textInputIds)
        let textEmptyMask = MLX.expandedDimensions(
            (textInputIds .== MLXArray(Int32(config.emptyIdx))),
            axis: -1
        )
        textEmbeds = MLX.where(
            textEmptyMask,
            MLXArray.zeros(textEmbeds.shape).asType(textEmbeds.dtype),
            textEmbeds
        )

        return textEmbeds + speechEmbeds
    }

    // MARK: - Backbone forward (one outer step)

    /// One backbone forward pass: input_ids → (text_logits, local_hidden_for_speech, cache).
    func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCacheSimple]
    ) -> (MLXArray, MLXArray) {
        let embeds = prepareInputEmbeds(inputIds)

        // Backbone with .causal for multi-token prefill, .none for cached 1-token decode.
        let hidden = qwen2Backbone.forward(
            inputEmbeddings: embeds,
            maskMode: nil,        // → L > 1 ? .causal : .none
            cache: cache
        )

        // Take last position only — that's where the model emits the next token.
        let last = hidden[0..., (hidden.shape[1] - 1)..., 0...]   // (B, 1, hidden)

        let textLogits = lmHead(last)
        let localHidden = hiddenStatesDowncast(last)
        return (textLogits, localHidden)
    }

    // MARK: - Local transformer (delay-pattern speech-token decoder)

    /// Run the delay-pattern local-transformer to produce a group's worth of
    /// speech tokens after the backbone has emitted an `empty_idx` text token.
    ///
    /// - Parameters:
    ///   - localEmbeds: `(B, 1, localDim)` — from `hiddenStatesDowncast`.
    ///   - sampler: speech-token sampler (greedy with empty-token mask).
    /// - Returns: `(B, groupSize, audioChannels)` int32 token IDs.
    func localForward(
        localEmbeds: MLXArray,
        sampler: MiMoSampler
    ) -> MLXArray {
        let B = localEmbeds.shape[0]
        let gs = groupSize
        let ac = audioChannels
        let delayIters = gs + (delayPattern.max() ?? 0)        // 4 + 7 = 11

        // Initialize KV cache for the 16 local-transformer layers.
        let cache = (0 ..< localTransformer.layers.count).map { _ in KVCacheSimple() }

        // Output buffer: per (group_position, channel) token.
        var tokens = [[Int32]](
            repeating: [Int32](repeating: 0, count: ac),
            count: gs
        )

        var currentInput = localEmbeds       // (B, 1, localDim)

        for t in 0 ..< delayIters {
            let h = localTransformer.forward(
                inputEmbeddings: currentInput,
                maskMode: nil,                              // L = 1 → .none
                cache: cache
            )
            let last = h[0..., (h.shape[1] - 1)..., 0...]   // (B, 1, localDim)

            // Build the next step's input by summing active channels' embeddings.
            var nextInput = MLXArray.zeros(localEmbeds.shape).asType(localEmbeds.dtype)

            for ch in 0 ..< ac {
                let curStart = delayPattern[ch]
                let curEnd = curStart + gs

                guard curStart <= t && t < curEnd else { continue }

                let scores = localTransformerLmHeads[ch](last)[0..., 0, 0...]   // (B, vocab_ch)
                let curEmpty = speechEmptyIds[ch]
                let sampled = sampler.sample(scores, removedTokens: [curEmpty]) // (B,)

                let rowIdx = t - curStart
                if rowIdx < gs {
                    // Record for batch element 0 (B == 1 in v1).
                    tokens[rowIdx][ch] = sampled[0].item(Int32.self)
                }

                // Embed for next-step input. Embedding lookup expects shape (B, 1).
                let lookupIds = MLX.expandedDimensions(sampled, axis: -1)
                let emb = speechEmbeddings[ch](lookupIds)                       // (B, 1, inputLocalDim)
                nextInput = nextInput + emb.asType(nextInput.dtype)
            }

            currentInput = nextInput
        }

        // Pack into MLXArray of shape (B=1, gs, ac).
        var flat = [Int32](repeating: 0, count: gs * ac)
        for k in 0 ..< gs {
            for c in 0 ..< ac {
                flat[k * ac + c] = tokens[k][c]
            }
        }
        return MLXArray(flat).reshaped([B, gs, ac])
    }

    // MARK: - Generation loop

    enum MiMoGenerationError: Error {
        case batchNotSupported(Int)
    }

    /// Outer autoregressive loop.
    ///
    /// - Parameters:
    ///   - inputIds: prompt `(B, audio_channels + 1, prompt_len)` from `MiMoPrompt.build`.
    ///   - maxNewTokens: stop after this many backbone forwards.
    ///   - globalSampler: text-token sampler.
    ///   - localSampler: speech-token sampler.
    ///   - stopTokens: terminate when the sampled text token is in this set.
    /// - Returns: full `(B, audio_channels + 1, total_len)` (prompt + generated).
    func generate(
        inputIds: MLXArray,
        maxNewTokens: Int = 256,
        globalSampler: MiMoSampler,
        localSampler: MiMoSampler,
        stopTokens: Set<Int>
    ) throws -> MLXArray {
        let B = inputIds.shape[0]
        guard B == 1 else { throw MiMoGenerationError.batchNotSupported(B) }

        let cache = (0 ..< qwen2Backbone.layers.count).map { _ in KVCacheSimple() }
        var currentIds = inputIds

        for step in 0 ..< maxNewTokens {
            let toFeed: MLXArray
            if step == 0 {
                toFeed = currentIds                              // full prompt prefill
            } else {
                // Only the new group_size positions per channel.
                let endT = currentIds.shape[2]
                toFeed = currentIds[0..., 0..., (endT - groupSize) ..< endT]
            }

            let (textLogits, localHidden) = self(toFeed, cache: cache)
            let textScores = textLogits[0..., (textLogits.shape[1] - 1), 0...]
            let nextTextToken = globalSampler.sample(textScores)     // (B,)
            let nextTextInt = Int(nextTextToken[0].item(Int32.self))

            // Speech token branch: only when next text token is the empty marker.
            let nextSpeech: MLXArray
            if nextTextInt == config.emptyIdx {
                nextSpeech = localForward(
                    localEmbeds: localHidden,
                    sampler: localSampler
                )                                                // (B, gs, ac)
            } else {
                // Fill with per-channel empty IDs.
                var buf = [Int32](repeating: 0, count: groupSize * audioChannels)
                for k in 0 ..< groupSize {
                    for c in 0 ..< audioChannels {
                        buf[k * audioChannels + c] = Int32(speechEmptyIds[c])
                    }
                }
                nextSpeech = MLXArray(buf).reshaped([B, groupSize, audioChannels])
            }

            // Build the (B, ac + 1, gs) step block.
            // - text row: broadcast nextTextToken across `gs` columns.
            let textRow = MLX.broadcast(
                MLX.expandedDimensions(
                    MLX.expandedDimensions(nextTextToken, axis: -1),
                    axis: -1
                ),
                to: [B, 1, groupSize]
            )
            // - speech rows: (B, gs, ac) → transposed (B, ac, gs).
            let speechRows = nextSpeech.transposed(0, 2, 1)

            let stepBlock = MLX.concatenated([textRow, speechRows], axis: 1)  // (B, ac+1, gs)
            currentIds = MLX.concatenated([currentIds, stepBlock], axis: -1)

            if stopTokens.contains(nextTextInt) { break }
        }

        return currentIds
    }
}
