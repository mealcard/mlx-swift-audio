// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import Foundation
import MLXLMCommon
import MLXLMTokenizers

/// Thin wrapper around the swift-tokenizers loader for the mimo HF tokenizer.
///
/// Codex pass 2 confirmed the mimo artifact ships only standard tokenizer
/// files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
/// `special_tokens_map.json`, `added_tokens.json`). There is NO custom
/// `tokenizer.py` — `trust_remote_code=True` in Python is defensive. So the
/// standard `TokenizersLoader().load(from:)` path works without remapping.
///
/// `tokenizer_class` is `Qwen2Tokenizer` (a standard byte-level BPE with
/// added special tokens like `<|im_start|>`, `<|im_end|>`, `<|empty|>`,
/// `<|sosp|>`, `<|eosp|>`, `<|eot|>`, etc.).
final class MiMoTokenizer {
    let tokenizer: any Tokenizer
    let eosTokenId: Int
    let padTokenId: Int

    private init(tokenizer: any Tokenizer, eosId: Int, padId: Int) {
        self.tokenizer = tokenizer
        self.eosTokenId = eosId
        self.padTokenId = padId
    }

    /// Load the tokenizer from the model directory.
    ///
    /// - Parameters:
    ///   - directory: model directory containing `tokenizer.json` etc.
    ///   - loader: tokenizer loader (defaults to `TokenizersLoader()`).
    static func load(
        from directory: URL,
        using loader: any TokenizerLoader = TokenizersLoader()
    ) async throws -> MiMoTokenizer {
        let tok = try await loader.load(from: directory)

        // Pull eos/pad ids from the underlying tokenizer. Fall back to the
        // mimo-default eos token id (151645) if unspecified.
        let eos = tok.eosTokenId ?? 151_645
        let pad = tok.encode(text: "<|endoftext|>").first ?? eos

        return MiMoTokenizer(tokenizer: tok, eosId: eos, padId: pad)
    }

    func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokenIds: tokens)
    }
}
