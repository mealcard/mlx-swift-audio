// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX

/// Build the (audio_channels + 1, totalLen) prompt tensor for MiMo-V2.5-ASR.
///
/// Row 0: text token IDs with `group_size - 1` filler tokens between real
/// tokens. Rows 1..audio_channels: speech token IDs (per-channel empty IDs
/// for text regions; codebook indices for audio regions).
///
/// Port of `mlx_audio/stt/models/mimo_v2_asr/prompt.py`.
public enum MiMoPrompt {
    public enum Language: Sendable { case auto, chinese, english }

    // MARK: - Hardcoded reference templates (deterministic; see spec §5.6)

    /// Chinese ASR instruction tokens — matches Python reference's training-
    /// time tokenization of "将音频内容转换成文字格式" (prompt.py:217).
    static let chineseInstructionTokens: [Int] =
        [44063, 111268, 43815, 105359, 12857, 87335, 68805]

    /// English ASR instruction template pinned for v1 (spec §5.6).
    /// Encoded via the tokenizer at runtime to allow vocab evolution.
    static let englishInstructionTemplate: String =
        "Please transcribe this audio file"

    /// Trailing "thinking/response" tokens for ZH (prompt.py:295).
    static let zhSeg6Tokens: [Int] =
        [13708, 766, 1339, 522, 26865, 397, 27, 331, 7346, 29]

    /// Trailing "thinking/response" tokens for EN (prompt.py:297).
    static let enSeg6Tokens: [Int] =
        [13708, 766, 1339, 522, 26865, 397, 27, 974, 975, 678, 29]

    // MARK: - Public entry point

    /// Build the full ASR prompt.
    ///
    /// - Parameters:
    ///   - audioCodes: flattened speech codes of shape `(T*audio_channels,)`.
    ///                 Must be divisible by `groupSize * audioChannels` (caller
    ///                 is responsible for padding).
    ///   - config: model config (provides empty IDs, sosp/eosp, group size).
    ///   - language: selects instruction template + segment-6 tokens.
    ///   - encode: text → token IDs encoder (typically `tok.encode(text:)`).
    /// - Returns: int32 tensor of shape `(audioChannels + 1, totalLen)`.
    static func build(
        audioCodes: MLXArray,
        config: MiMoAudioConfig,
        language: Language,
        encode: (String) -> [Int]
    ) -> MLXArray {
        let segments: [MLXArray] = [
            textSegment(
                text: "<|im_start|>user\n",
                config: config,
                encode: encode,
                addSospEosp: true
            ),
            audioSegment(
                codes: audioCodes,
                config: config
            ),
            instructionSegment(
                language: language,
                config: config,
                encode: encode
            ),
            textSegment(
                text: "<|im_end|>\n",
                config: config,
                encode: encode,
                addSospEosp: false
            ),
            textSegment(
                text: "<|im_start|>assistant\n",
                config: config,
                encode: encode,
                addSospEosp: false
            ),
            seg6Segment(language: language, config: config, encode: encode),
        ]
        return MLX.concatenated(segments, axis: 1)
    }

    // MARK: - Segment builders

    /// Build a text-only segment. Speech rows are filled with per-channel
    /// empty IDs. Text is expanded with `groupSize - 1` filler tokens
    /// between each real token (filler is the model's `emptyIdx`).
    ///
    /// `addSospEosp` is accepted for API symmetry with the Python reference's
    /// `InputSegment` constructor but has no effect on text-only segments —
    /// sosp/eosp markers wrap the audio segment only.
    private static func textSegment(
        text: String,
        config: MiMoAudioConfig,
        encode: (String) -> [Int],
        addSospEosp: Bool
    ) -> MLXArray {
        _ = addSospEosp
        let ids = encode(text)
        let expanded = expandWithFillers(
            ids: ids,
            filler: config.emptyIdx,
            groupSize: config.groupSize
        )
        return assembleSegment(textRow: expanded, config: config)
    }

    /// Build the audio segment. Reshapes the flat codes into per-channel
    /// rows, wraps with sosp/eosp markers (one group each, all-empty speech),
    /// and fills the text row with `emptyIdx` for every speech-token group.
    private static func audioSegment(
        codes: MLXArray,
        config: MiMoAudioConfig
    ) -> MLXArray {
        let ac = config.audioChannels
        let gs = config.groupSize
        let total = codes.shape[0]
        precondition(total % (gs * ac) == 0,
                     "audio_codes length \(total) not divisible by group_size * audio_channels")
        let nGroups = total / (gs * ac)

        // (nGroups, gs * ac) → (nGroups, gs, ac) → (ac, nGroups, gs) → (ac, nGroups * gs)
        let speech = codes
            .reshaped([nGroups, gs * ac])
            .reshaped([nGroups, gs, ac])
            .transposed(2, 0, 1)
            .reshaped([ac, nGroups * gs])

        // Text row: emptyIdx per group, plus sosp + eosp markers (one group each).
        var textIds = [Int](repeating: config.emptyIdx, count: nGroups)
        textIds.insert(config.sospIdx, at: 0)
        textIds.append(config.eospIdx)

        let textRow = expandWithFillers(
            ids: textIds,
            filler: config.emptyIdx,
            groupSize: gs
        )

        // Build speech rows with sosp/eosp framing groups (gs columns of empty per channel).
        let emptyIds = config.parsedSpeechEmptyIds
        var sospSpeech = [Int32](repeating: 0, count: gs * ac)
        var eospSpeech = [Int32](repeating: 0, count: gs * ac)
        for ch in 0 ..< ac {
            for k in 0 ..< gs {
                sospSpeech[ch * gs + k] = Int32(emptyIds[ch])
                eospSpeech[ch * gs + k] = Int32(emptyIds[ch])
            }
        }
        let sospMx = MLXArray(sospSpeech).reshaped([ac, gs])
        let eospMx = MLXArray(eospSpeech).reshaped([ac, gs])

        let speechWithFrames = MLX.concatenated([sospMx, speech, eospMx], axis: 1)

        // Text row already expanded with fillers; stack on top of speech.
        return MLX.concatenated([
            MLX.expandedDimensions(textRow, axis: 0),
            speechWithFrames
        ], axis: 0)
    }

    private static func instructionSegment(
        language: Language,
        config: MiMoAudioConfig,
        encode: (String) -> [Int]
    ) -> MLXArray {
        let ids: [Int] = switch language {
        case .chinese, .auto: chineseInstructionTokens
        case .english: encode(englishInstructionTemplate)
        }
        let expanded = expandWithFillers(
            ids: ids,
            filler: config.emptyIdx,
            groupSize: config.groupSize
        )
        return assembleSegment(textRow: expanded, config: config)
    }

    private static func seg6Segment(
        language: Language,
        config: MiMoAudioConfig,
        encode: (String) -> [Int]
    ) -> MLXArray {
        let ids: [Int] = switch language {
        case .chinese, .auto: zhSeg6Tokens
        case .english: enSeg6Tokens
        }
        let expanded = expandWithFillers(
            ids: ids,
            filler: config.emptyIdx,
            groupSize: config.groupSize
        )
        return assembleSegment(textRow: expanded, config: config)
    }

    // MARK: - Helpers

    /// Insert (groupSize - 1) filler tokens between each real token.
    /// Output length = `ids.count * groupSize`. Real tokens land at indices
    /// 0, gs, 2·gs, ...
    private static func expandWithFillers(
        ids: [Int],
        filler: Int,
        groupSize gs: Int
    ) -> MLXArray {
        let outLen = ids.count * gs
        var buf = [Int32](repeating: Int32(filler), count: outLen)
        for (i, id) in ids.enumerated() {
            buf[i * gs] = Int32(id)
        }
        return MLXArray(buf)
    }

    /// Stack a text row above per-channel empty speech rows.
    private static func assembleSegment(
        textRow: MLXArray,
        config: MiMoAudioConfig
    ) -> MLXArray {
        let ac = config.audioChannels
        let segLen = textRow.shape[0]
        let emptyIds = config.parsedSpeechEmptyIds

        var speechBuf = [Int32](repeating: 0, count: ac * segLen)
        for ch in 0 ..< ac {
            let v = Int32(emptyIds[ch])
            for k in 0 ..< segLen {
                speechBuf[ch * segLen + k] = v
            }
        }
        let speech = MLXArray(speechBuf).reshaped([ac, segLen])

        return MLX.concatenated([
            MLX.expandedDimensions(textRow, axis: 0),
            speech
        ], axis: 0)
    }
}
