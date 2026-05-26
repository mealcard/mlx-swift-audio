// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXLMCommon
import MLXLMTokenizers

/// Thread-safe wrapper that holds the loaded model + tokenizer and runs one
/// transcription at a time.
///
/// Mirrors `FunASRSTT`'s actor pattern. Non-Sendable members are marked
/// `nonisolated(unsafe)` because they are only accessed inside this actor's
/// methods.
actor MiMoSTT {
    nonisolated(unsafe) let model: MiMoModel
    nonisolated(unsafe) let tokenizer: MiMoTokenizer

    init(model: MiMoModel, tokenizer: MiMoTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    static func load(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader = TokenizersLoader()
    ) async throws -> MiMoSTT {
        let model = try MiMoModel.load(from: directory)
        let tokenizer = try await MiMoTokenizer.load(
            from: model.modelDirectory,
            using: tokenizerLoader
        )
        return MiMoSTT(model: model, tokenizer: tokenizer)
    }

    /// Transcribe a 24 kHz mono waveform.
    func transcribe(
        waveform: sending MLXArray,
        language: MiMoPrompt.Language,
        options: MiMoOptions
    ) throws -> MiMoTranscription {
        let start = Date()

        // 1. Mel + audio codes.
        let mel = MiMoAudio.logMelSpectrogram(waveform)
        let codesAll = model.audioEncoder.encode(mel, nQ: model.config.audioChannels)
        // codesAll: (n_q, T_tokens). Reshape to flat (T_tokens * n_q,) row-major
        // by channel-first → frame-first interleaving, matching Python:
        //   audio_codes = codes.transpose(1, 0).reshape(-1)
        let codesFlat = codesAll.transposed(1, 0).reshaped([-1])

        // 2. Pad to a multiple of (group_size * audio_channels) by tiling the
        //    final frame (Python `_encode_audio_codes`).
        let gs = model.config.groupSize
        let ac = model.config.audioChannels
        let totalNeeded = gs * ac
        let codes: MLXArray
        let total = codesFlat.shape[0]
        let remainder = total % totalNeeded
        if remainder == 0 {
            codes = codesFlat
        } else {
            let padLen = totalNeeded - remainder
            let lastFrame = codesFlat[(total - ac) ..< total]
            // Tile lastFrame enough times to cover padLen elements.
            let copies = (padLen + ac - 1) / ac
            var padPieces: [MLXArray] = []
            for _ in 0 ..< copies { padPieces.append(lastFrame) }
            let pad = MLX.concatenated(padPieces, axis: 0)[0 ..< padLen]
            codes = MLX.concatenated([codesFlat, pad], axis: 0)
        }

        // 3. Build prompt.
        let prompt = MiMoPrompt.build(
            audioCodes: codes,
            config: model.config,
            language: language,
            encode: { tokenizer.encode($0) }
        )
        // (channels+1, T) → (1, channels+1, T)
        let promptBatched = MLX.expandedDimensions(prompt, axis: 0)

        // 4. Generate.
        let globalSampler = MiMoSampler(
            doSample: options.temperature > 0,
            temperature: options.temperature > 0 ? options.temperature : 1.0,
            topK: options.topK,
            topP: options.topP
        )
        let localSampler = MiMoSampler(doSample: false)

        let stopTokens: Set<Int> = [
            tokenizer.eosTokenId,
            model.config.eotIdx
        ]

        let generated = try model.llm.generate(
            inputIds: promptBatched,
            maxNewTokens: options.maxTokens,
            globalSampler: globalSampler,
            localSampler: localSampler,
            stopTokens: stopTokens
        )

        // 5. Extract text tokens: text row, every group_size-th column,
        //    starting from prompt end.
        let promptLen = prompt.shape[1]
        let totalLen = generated.shape[2]
        let textRow = generated[0, 0, 0...]
        // Indices: promptLen, promptLen + gs, promptLen + 2*gs, ...
        var ids: [Int] = []
        var idx = promptLen
        while idx < totalLen {
            let tok = Int(textRow[idx].item(Int32.self))
            ids.append(tok)
            idx += gs
        }

        // Drop trailing eos/eot/eostm tokens.
        let strip: Set<Int> = [
            tokenizer.eosTokenId,
            model.config.eotIdx,
            model.config.eostmIdx
        ]
        while let last = ids.last, strip.contains(last) {
            ids.removeLast()
        }

        let raw = tokenizer.decode(ids)
        let text = cleanOutput(raw)

        let total_time = Date().timeIntervalSince(start)
        let duration = Double(waveform.shape[0]) / Double(MiMoAudio.sampleRate)
        return MiMoTranscription(
            text: text,
            language: language == .auto ? nil : language,
            audioDuration: duration,
            totalTime: total_time,
            promptTokens: promptLen,
            generationTokens: ids.count
        )
    }

    private nonisolated func cleanOutput(_ text: String) -> String {
        text
            .replacingOccurrences(of: "<|empty|>", with: "")
            .replacingOccurrences(of: "<|eot|>", with: "")
            .replacingOccurrences(of: "<|eostm|>", with: "")
            .replacingOccurrences(of: "<chinese>", with: "")
            .replacingOccurrences(of: "<english>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Public transcription-input options.
public struct MiMoOptions: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int

    public init(maxTokens: Int = 256, temperature: Float = 0.0, topP: Float = 0.95, topK: Int = 0) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }
}

/// Public transcription result.
public struct MiMoTranscription: Sendable {
    public let text: String
    public let language: MiMoPrompt.Language?
    public let audioDuration: TimeInterval
    public let totalTime: TimeInterval
    public let promptTokens: Int
    public let generationTokens: Int
}
