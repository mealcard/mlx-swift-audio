// Copyright © Anthony DePasquale

import Foundation

/// Available STT (Speech-to-Text) providers
public enum STTProvider: String, CaseIterable, Identifiable, Sendable {
  case whisper
  case funASR
  case mimo

  public var id: String { rawValue }

  /// Canonical display name with proper casing/branding
  public var displayName: String {
    switch self {
      case .whisper: "Whisper"
      case .funASR: "Fun-ASR"
      case .mimo: "MiMo-V2.5-ASR"
    }
  }

  // MARK: - Audio Properties

  /// Sample rate for this provider's audio input (Hz)
  public var sampleRate: Int {
    switch self {
      case .whisper: 16000
      case .funASR: 16000
      case .mimo: 24000
    }
  }

  // MARK: - Feature Flags

  /// Whether this provider supports language detection
  public var supportsLanguageDetection: Bool {
    switch self {
      case .whisper: true
      case .funASR: true
      case .mimo: true // .auto language path
    }
  }

  /// Whether this provider supports translation to English
  public var supportsTranslation: Bool {
    switch self {
      case .whisper: true
      case .funASR: true
      case .mimo: false // transcription-only in v1
    }
  }

  /// Whether this provider supports word-level timestamps
  public var supportsWordTimestamps: Bool {
    switch self {
      case .whisper: false // Will be added in future phase
      case .funASR: false // LLM-based model doesn't produce word timestamps
      case .mimo: false // LLM-based model doesn't produce word timestamps
    }
  }

  /// Whether this provider supports streaming transcription
  public var supportsStreaming: Bool {
    switch self {
      case .whisper: false // Will be added in future phase
      case .funASR: true // LLM-based model supports token streaming
      case .mimo: false // not yet — Python reference is whole-clip only
    }
  }
}
