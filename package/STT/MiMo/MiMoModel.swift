// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Pairs the MiMo LLM (`MiMoAudioMLX`) with its audio tokenizer
/// (`MiMoAudioTokenizer`) and handles end-to-end weight loading from
/// the `mlx-community/MiMo-V2.5-ASR-MLX` artifact layout.
///
/// See spec §5.7 + codex rev-2 fixes (full-shard merge, tuple-filter
/// quantize, strict `verify: [.noUnusedKeys]`, drop `position_embedding.inv_freq`).
final class MiMoModel {
    let config: MiMoAudioConfig
    let llm: MiMoAudioMLX
    let audioEncoder: MiMoAudioTokenizer
    let modelDirectory: URL

    private init(
        config: MiMoAudioConfig,
        llm: MiMoAudioMLX,
        audioEncoder: MiMoAudioTokenizer,
        modelDirectory: URL
    ) {
        self.config = config
        self.llm = llm
        self.audioEncoder = audioEncoder
        self.modelDirectory = modelDirectory
    }

    // MARK: - Loading

    static func load(from directory: URL) throws -> MiMoModel {
        // Config.
        let configURL = directory.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(MiMoAudioConfig.self, from: configData)

        // Resolve the audio tokenizer directory: prefer mlx_manifest.json,
        // fall back to sibling "MiMo-Audio-Tokenizer".
        let tokDir = try resolveAudioTokenizerDir(modelDir: directory)
        let tokConfigURL = tokDir.appendingPathComponent("config.json")
        let tokConfigData = try Data(contentsOf: tokConfigURL)
        let tokConfig = try JSONDecoder().decode(MiMoTokenizerConfig.self, from: tokConfigData)

        // Build modules.
        let llm = MiMoAudioMLX(config)
        let audioEncoder = MiMoAudioTokenizer(config: tokConfig)

        // ── LLM weights ─────────────────────────────────────────────────────
        let llmShards = try safetensorShards(in: directory)
        var llmMerged: [String: MLXArray] = [:]
        for url in llmShards {
            for (k, v) in try MLX.loadArrays(url: url) {
                llmMerged[k] = v
            }
        }
        let llmWeights = sanitizeLLMWeights(llmMerged)

        // Quantize layers whose weights actually arrived quantized.
        if let q = config.quantization {
            let mode = parseQuantMode(q.mode)
            quantize(
                model: llm,
                groupSize: q.groupSize,
                bits: q.bits,
                mode: mode,
                filter: { path, _ in
                    llmWeights["\(path).scales"] != nil
                }
            )
        }

        try llm.update(
            parameters: ModuleParameters.unflattened(llmWeights),
            verify: [.noUnusedKeys]
        )
        eval(llm)

        // ── Audio tokenizer weights ─────────────────────────────────────────
        let tokShards = try safetensorShards(in: tokDir)
        var tokMerged: [String: MLXArray] = [:]
        for url in tokShards {
            for (k, v) in try MLX.loadArrays(url: url) {
                tokMerged[k] = v
            }
        }
        let tokWeights = MiMoAudioTokenizer.sanitize(tokMerged)

        try audioEncoder.update(
            parameters: ModuleParameters.unflattened(tokWeights),
            verify: [.noUnusedKeys]
        )
        eval(audioEncoder)

        return MiMoModel(
            config: config,
            llm: llm,
            audioEncoder: audioEncoder,
            modelDirectory: directory
        )
    }

    // MARK: - Helpers

    static func resolveAudioTokenizerDir(modelDir: URL) throws -> URL {
        let manifestURL = modelDir.appendingPathComponent("mlx_manifest.json")
        if let data = try? Data(contentsOf: manifestURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let rel = json["audio_tokenizer_dir"] as? String {
                let candidate = modelDir.appendingPathComponent(rel)
                    .standardizedFileURL
                if FileManager.default.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        // Fall back to sibling directory.
        let sibling = modelDir.deletingLastPathComponent()
            .appendingPathComponent("MiMo-Audio-Tokenizer")
        if FileManager.default.fileExists(atPath: sibling.path) {
            return sibling
        }
        throw MiMoLoadError.audioTokenizerDirNotFound(searchedNear: modelDir)
    }

    static func safetensorShards(in directory: URL) throws -> [URL] {
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        return contents
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Sanitize the LLM safetensors. The artifact is already in mlx-compatible
    /// format (no Conv1d transposition needed for the LLM side), so the only
    /// rule is "pass through". Listed as a function for parallel structure
    /// with the audio-tokenizer sanitizer + future rule additions.
    static func sanitizeLLMWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        return weights
    }

    static func parseQuantMode(_ s: String) -> QuantizationMode {
        switch s.lowercased() {
        case "affine": return .affine
        default:       return .affine
        }
    }
}

enum MiMoLoadError: Error, LocalizedError {
    case audioTokenizerDirNotFound(searchedNear: URL)

    var errorDescription: String? {
        switch self {
        case .audioTokenizerDirNotFound(let url):
            return """
            Unable to resolve the MiMo audio tokenizer directory near \(url.path).
            Looked at mlx_manifest.json's audio_tokenizer_dir and the sibling \
            "MiMo-Audio-Tokenizer" directory.
            """
        }
    }
}
