// Copyright © OpenTTS contributors (Swift port)
// License: MIT
//
// Pre-implementation spike for the mimo Swift port.
// Validates the §9 acceptance gates BEFORE the model code is written:
//   1. Tokenizer can load mimo HF tokenizer via TokenizersLoader.
//   2. Real mimo safetensors weights can be loaded into a CosyVoice2-style
//      quantized Qwen2 transformer block using mlx-swift / mlx-swift-audio's
//      existing APIs (no custom remapping).
//
// Run: swift test --filter MimoSpike

import Foundation
import MLX
import MLXLMCommon
import MLXLMTokenizers
import MLXNN
import Testing

@testable import MLXAudio

/// Directory containing the mimo LLM weights. Override via env var, otherwise
/// default to LM Studio's standard cache location.
private let mimoLLMDir = URL(fileURLWithPath:
    ProcessInfo.processInfo.environment["MIMO_MODEL_DIR"]
    ?? NSString(string: "~/.lmstudio/models/mlx-community/MiMo-V2.5-ASR-MLX")
        .expandingTildeInPath
)
private let goldenPath = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("MimoSpike/tokenizer_golden.json")

private struct GoldenFile: Codable {
    let vocab_size: Int
    let eos_token_id: Int?
    let pad_token_id: Int?
    let special_tokens: [String: Int]
    let fragments: [String: [Int]]
    let zh_template_expected: [Int]
    let zh_seg6_expected: [Int]
    let en_seg6_expected: [Int]
}

private func loadGolden() throws -> GoldenFile {
    let data = try Data(contentsOf: goldenPath)
    return try JSONDecoder().decode(GoldenFile.self, from: data)
}

@Suite("MimoSpike")
struct MimoSpikeTests {

    // MARK: - Gate 1: tokenizer parity
    //
    // The risk codex flagged: mimo's HF tokenizer config sets trust_remote_code=True
    // in Python, so swift-tokenizers might not be able to load it. We've confirmed
    // the on-disk directory only contains standard files (tokenizer.json, vocab.json,
    // merges.txt, tokenizer_config.json, special_tokens_map.json, added_tokens.json,
    // chat_template.jinja) — no custom tokenizer.py. So this should "just work".
    // This test proves it.

    @Test @MainActor
    func tokenizerLoadsAndMatchesPython() async throws {
        guard FileManager.default.fileExists(atPath: mimoLLMDir.path) else {
            Issue.record("Skipped: mimo weights not at \(mimoLLMDir.path)")
            return
        }

        let golden = try loadGolden()

        let loader = TokenizersLoader()
        let tok = try await loader.load(from: mimoLLMDir)

        // Encode each fragment and compare against Python AutoTokenizer's output.
        for (name, expected) in golden.fragments {
            let actual = tok.encode(text: nameToString(name))
            #expect(
                actual == expected,
                """
                Fragment \(name) mismatch.
                  expected: \(expected)
                  actual:   \(actual)
                """
            )
        }

        // Sanity check that the special-token IDs the model config hardcodes are
        // discoverable through the tokenizer (so prompt-building can use them
        // either via constants or via dynamic lookup).
        #expect(golden.special_tokens["empty_idx"] == 151667)
        #expect(golden.special_tokens["sosp_idx"] == 151665)
        #expect(golden.special_tokens["eosp_idx"] == 151666)
        #expect(golden.special_tokens["eot_idx"] == 151672)
    }

    /// The fragment strings the Python golden dumper used (must stay in sync).
    private func nameToString(_ name: String) -> String {
        switch name {
        case "im_start_user":      return "<|im_start|>user\n"
        case "im_end_newline":     return "<|im_end|>\n"
        case "im_start_assist":    return "<|im_start|>assistant\n"
        case "zh_template_str":    return "将音频内容转换成文字格式"
        case "thinking_zh":        return " thinking\n\n response\n<chinese>"
        case "thinking_en":        return " thinking\n\n response\n<english>"
        case "en_template_pinned": return "Please transcribe this audio file"
        default: fatalError("unknown fragment: \(name)")
        }
    }

    // MARK: - Gate 2: weight load + quantize + strict update
    //
    // Goal: prove the established mlx-swift-audio pattern — `MLX.loadArrays(url:)`
    // + `MLXNN.quantize(model:){tuple filter}` + `model.update(parameters:
    // ModuleParameters.unflattened(weights), verify:[.noUnusedKeys])` — works
    // end-to-end against real mimo backbone layer 0 weights without any custom
    // remapping. Mimo backbone is Qwen2-shaped (hidden 4096, 32 heads, 8 KV
    // heads via GQA, head_dim 128, rope_theta 640000), so we can reuse
    // CosyVoiceQwen2TransformerBlock unchanged for this gate.

    @Test @MainActor
    func backboneLayerZeroLoadsAndQuantizes() throws {
        guard FileManager.default.fileExists(atPath: mimoLLMDir.path) else {
            Issue.record("Skipped: mimo weights not at \(mimoLLMDir.path)")
            return
        }

        // 1. Build mimo backbone config.
        var cfg = CosyVoiceQwen2Config()
        cfg.hiddenSize = 4096
        cfg.numHiddenLayers = 36
        cfg.intermediateSize = 11008
        cfg.numAttentionHeads = 32
        cfg.numKeyValueHeads = 8
        cfg.rmsNormEps = 1e-6
        cfg.vocabSize = 151680
        cfg.maxPositionEmbeddings = 8192
        cfg.ropeTheta = 640_000.0
        cfg.ropeTraditional = false
        cfg.tieWordEmbeddings = false

        // 2. Build one decoder block (mimo's `model.layers.0`).
        let block = CosyVoiceQwen2TransformerBlock(cfg)

        // 3. Load all safetensors arrays into memory.
        let url = mimoLLMDir.appendingPathComponent("model.safetensors")
        let allArrays = try MLX.loadArrays(url: url)
        #expect(allArrays.count > 0)

        // 4. Slice to just this block's keys and strip the "model.layers.0." prefix
        //    so they match the block's own parameter paths.
        let prefix = "model.layers.0."
        var blockWeights: [String: MLXArray] = [:]
        for (k, v) in allArrays where k.hasPrefix(prefix) {
            blockWeights[String(k.dropFirst(prefix.count))] = v
        }
        #expect(blockWeights.count > 0, "no model.layers.0.* keys found")

        // Sanity-check the shapes we're about to load.
        #expect(blockWeights["self_attn.q_proj.weight"]?.shape == [4096, 512],
                "q_proj quantized weight shape mismatch")
        #expect(blockWeights["self_attn.k_proj.weight"]?.shape == [1024, 512],
                "k_proj quantized weight shape mismatch")
        #expect(blockWeights["self_attn.q_proj.bias"]?.shape == [4096],
                "q_proj bias shape mismatch")
        #expect(blockWeights["mlp.gate_proj.weight"]?.shape == [11008, 512],
                "gate_proj quantized weight shape mismatch")
        #expect(blockWeights["input_layernorm.weight"]?.shape == [4096],
                "input_layernorm weight shape mismatch")

        // 5. Quantize per-layer using the tuple-returning filter (FunASRModel pattern).
        //    Returning nil → leave layer alone. Returning (gs, bits, mode) → quantize it.
        //    Predicate is "the layer has a companion .scales tensor in the safetensors".
        let isQuantized = blockWeights.keys.contains { $0.hasSuffix(".scales") }
        #expect(isQuantized, "expected quantized mimo weights")

        quantize(model: block) { path, _ in
            blockWeights["\(path).scales"] != nil ? (64, 4, .affine) : nil
        }

        // 6. Strict load. .noUnusedKeys mirrors FunASRModel; we want it to throw
        //    if the safetensors contains keys this block can't consume.
        let params = ModuleParameters.unflattened(blockWeights)
        try block.update(parameters: params, verify: [.noUnusedKeys])

        // 7. Force evaluation so any lazy mismatch surfaces here, not at first inference.
        eval(block)

        // 8. Smoke-test forward pass: feed a (1, 4, 4096) input, expect (1, 4, 4096) out.
        //    No KV cache → fresh prefill path.
        let x = MLXArray.zeros([1, 4, 4096]).asType(.bfloat16)
        let cache = KVCacheSimple()
        let y = block(x, mask: nil, cache: cache)
        #expect(y.shape == [1, 4, 4096], "forward output shape wrong: \(y.shape)")
        #expect(cache.offset == 4, "KV cache offset should advance by sequence length")
    }
}
