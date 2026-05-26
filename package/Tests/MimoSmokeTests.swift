// Copyright © OpenTTS contributors
// License: MIT
//
// Real-world podcast smoke test. Not part of the parity oracle; prints
// Swift transcripts for visual inspection against the Python reference.

import Foundation
import MLX
import MLXLMCommon
import MLXLMTokenizers
import Testing

@testable import MLXAudio

@Suite("MimoSmoke")
struct MimoSmokeTests {

    @Test @MainActor
    func runOnPodcastClips() async throws {
        let clips: [(path: String, lang: MiMoPrompt.Language, label: String)] = [
            ("/tmp/mimo-podcast-test/npr_en.wav",      .english, "NPR podcast (real)"),
            ("/tmp/mimo-podcast-test/en_uk_long.wav",  .english, "UK English (synth)"),
            ("/tmp/mimo-podcast-test/zh_realistic.wav", .chinese, "Mandarin (synth)"),
        ]
        for (path, _, _) in clips where !FileManager.default.fileExists(atPath: path) {
            Issue.record("Skipping: \(path) missing")
            return
        }

        let weightsDir = ProcessInfo.processInfo.environment["MIMO_MODEL_DIR"]
            ?? NSString(string: "~/.lmstudio/models/mlx-community/MiMo-V2.5-ASR-MLX")
                .expandingTildeInPath
        let engine = STT.mimo(localDirectory: URL(fileURLWithPath: weightsDir))
        try await engine.load()

        print()
        print("=== Swift mimo transcripts ===")
        for clip in clips {
            let result = try await engine.transcribe(
                URL(fileURLWithPath: clip.path),
                language: clip.lang
            )
            print("[\(clip.label)] \(String(format: "%.2f", result.totalTime))s -> \(result.text)")
        }
        print("===")
    }
}
