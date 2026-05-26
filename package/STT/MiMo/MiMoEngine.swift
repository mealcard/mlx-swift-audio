// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXLMHFAPI
import MLXLMTokenizers

/// MiMo-V2.5-ASR engine — Xiaomi's SOTA multilingual ASR (zh/en/dialects/code-switch).
///
/// v1 loads the int4 affine quantization shipped in
/// `mlx-community/MiMo-V2.5-ASR-MLX`. Selectable precision variants
/// (q8/fp16) are deferred until artifact-backed repository IDs exist.
@Observable
@MainActor
public final class MiMoEngine: STTEngine {
    // MARK: - STTEngine

    public let provider: STTProvider = .mimo
    public private(set) var isLoaded: Bool = false
    public private(set) var isTranscribing: Bool = false
    public private(set) var transcriptionTime: TimeInterval = 0

    // MARK: - Private

    @ObservationIgnored private var stt: MiMoSTT?
    @ObservationIgnored private let downloader: any Downloader
    @ObservationIgnored private let tokenizerLoader: any TokenizerLoader
    @ObservationIgnored private let explicitLocalDirectory: URL?

    /// HuggingFace repo for the LLM weights.
    public static let llmRepoId = "mlx-community/MiMo-V2.5-ASR-MLX"
    /// HuggingFace repo for the audio tokenizer weights.
    public static let tokenizerRepoId = "mlx-community/MiMo-Audio-Tokenizer"

    public init(
        from downloader: any Downloader = HubClient.default,
        using tokenizerLoader: any TokenizerLoader = TokenizersLoader()
    ) {
        self.downloader = downloader
        self.tokenizerLoader = tokenizerLoader
        self.explicitLocalDirectory = nil
    }

    /// Load directly from a local directory, skipping HF download. Useful when
    /// weights are already in `~/.lmstudio/models/...` or similar.
    public init(
        localDirectory: URL,
        using tokenizerLoader: any TokenizerLoader = TokenizersLoader()
    ) {
        self.downloader = HubClient.default
        self.tokenizerLoader = tokenizerLoader
        self.explicitLocalDirectory = localDirectory
    }

    // MARK: - Lifecycle

    public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
        guard !isLoaded else { return }

        let modelDir: URL
        if let local = explicitLocalDirectory {
            modelDir = local
            // Make sure the audio tokenizer is also reachable.
            _ = try MiMoModel.resolveAudioTokenizerDir(modelDir: modelDir)
        } else {
            // Download both repos. Audio tokenizer first so the LLM's
            // mlx_manifest.json can resolve `audio_tokenizer_dir` as a sibling.
            _ = try await downloader.download(
                id: Self.tokenizerRepoId,
                revision: nil,
                matching: ["*.json", "*.safetensors", "*.txt"],
                useLatest: false,
                progressHandler: progressHandler ?? { _ in }
            )
            modelDir = try await downloader.download(
                id: Self.llmRepoId,
                revision: nil,
                matching: ["*.json", "*.safetensors", "*.txt", "*.jinja"],
                useLatest: false,
                progressHandler: progressHandler ?? { _ in }
            )
        }

        stt = try await MiMoSTT.load(from: modelDir, using: tokenizerLoader)
        isLoaded = true
    }

    public func stop() async {
        // No mid-generation cancellation in v1 (generation is short — typically
        // < 1 s per clip — and runs on the actor).
    }

    public func unload() async {
        stt = nil
        isLoaded = false
    }

    public func cleanup() async throws {
        await unload()
    }

    // MARK: - Transcription

    /// Transcribe audio from a file URL (any AVFoundation-readable format).
    /// Audio is resampled to 24 kHz mono internally.
    public func transcribe(
        _ audioURL: URL,
        language: MiMoPrompt.Language = .auto,
        options: MiMoOptions = .init()
    ) async throws -> MiMoTranscription {
        let waveform = try MiMoAudio.loadAudio(audioURL)
        return try await transcribe(
            waveform: waveform,
            sampleRate: MiMoAudio.sampleRate,
            language: language,
            options: options
        )
    }

    /// Transcribe a pre-loaded waveform.
    public func transcribe(
        waveform: sending MLXArray,
        sampleRate: Int,
        language: MiMoPrompt.Language = .auto,
        options: MiMoOptions = .init()
    ) async throws -> MiMoTranscription {
        guard let stt else {
            throw MiMoEngineError.notLoaded
        }
        guard sampleRate == MiMoAudio.sampleRate else {
            // For v1 we only accept already-24 kHz waveforms here. Use the
            // file-URL overload above for arbitrary-rate input.
            throw MiMoEngineError.invalidSampleRate(expected: MiMoAudio.sampleRate, got: sampleRate)
        }
        guard waveform.ndim == 1 else {
            throw MiMoEngineError.invalidWaveformShape(waveform.shape)
        }

        isTranscribing = true
        defer { isTranscribing = false }

        let result = try await stt.transcribe(
            waveform: waveform,
            language: language,
            options: options
        )
        transcriptionTime = result.totalTime
        return result
    }
}

public enum MiMoEngineError: LocalizedError {
    case notLoaded
    case invalidSampleRate(expected: Int, got: Int)
    case invalidWaveformShape([Int])
    case batchNotSupported

    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "MiMoEngine.load() must be called before transcribing."
        case .invalidSampleRate(let expected, let got):
            return "MiMoEngine expects \(expected) Hz waveform; got \(got) Hz. Use the file-URL overload for arbitrary rates."
        case .invalidWaveformShape(let shape):
            return "MiMoEngine expects 1-D waveform (mono); got shape \(shape). v1 does not support batching."
        case .batchNotSupported:
            return "MiMoEngine v1 supports B == 1 only."
        }
    }
}

// MARK: - STT factory extension

public extension STT {
    /// MiMo-V2.5-ASR: Xiaomi multilingual ASR (zh/en/dialects/code-switch).
    ///
    /// v1 uses the int4 affine quantization shipped in
    /// `mlx-community/MiMo-V2.5-ASR-MLX`.
    static func mimo() -> MiMoEngine {
        MiMoEngine()
    }

    /// MiMo from an explicit local directory (skips HF download).
    static func mimo(localDirectory: URL) -> MiMoEngine {
        MiMoEngine(localDirectory: localDirectory)
    }
}
