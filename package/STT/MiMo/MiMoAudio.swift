// Copyright © 2025 Xiaomi LLM-Core-Team (original model architecture)
// Copyright © 2025 ailuntx (Python MLX port)
// Copyright © OpenTTS contributors (Swift port)
// License: licenses/mimo.txt (Apache-2.0) + licenses/mlx-audio.txt (MIT)

import AVFoundation
import Foundation
import MLX

/// Mel spectrogram helpers for MiMo-V2.5-ASR (24 kHz, n_fft=960, hop=240,
/// win=960, n_mels=128, **HTK mel scale with no normalization**, natural log).
///
/// Ports `mlx_audio/stt/models/mimo_v2_asr/mel.py` (Python reference) and
/// reuses the existing `stft` + `hanningWindow` helpers from
/// `Codec/S3Tokenizer/S3TokenizerUtils.swift`. The mel filterbank is new
/// because S3's `melFilters` is Slaney-normalized whereas mimo wants HTK
/// with no normalization (matches Python `mel_filters(..., norm=None,
/// mel_scale="htk")`, see `mlx_audio/dsp.py:500-572`).
enum MiMoAudio {
    static let sampleRate = 24_000
    static let nFft = 960
    static let hopLength = 240
    static let winLength = 960
    static let nMels = 128

    /// Compute the log-mel spectrogram with mimo parameters.
    ///
    /// - Parameter audio: 1-D waveform at 24 kHz.
    /// - Returns: mel spectrogram of shape `(nMels, nFrames)`.
    static func logMelSpectrogram(_ audio: MLXArray) -> MLXArray {
        precondition(audio.ndim == 1, "audio must be 1-D")

        // STFT (complex spectrogram, shape: (nFrames, nFft/2 + 1)).
        let window = hanningWindow(length: winLength)
        let spec = stft(
            audio,
            window: window,
            nFft: nFft,
            hopLength: hopLength,
            winLength: winLength,
            center: true,
            padMode: "reflect"
        )

        // Magnitude spectrum (power = 1.0). Keep all bins including Nyquist.
        let magnitude = MLX.abs(spec)

        // Apply HTK mel filterbank (no normalization).
        // Shape: (nMels, nFft/2 + 1) → transpose for matmul.
        let filters = melFiltersHTKNoNorm(
            sampleRate: sampleRate,
            nFft: nFft,
            nMels: nMels
        )
        let mel = MLX.matmul(magnitude, filters.T)   // (nFrames, nMels)

        // Natural log (floor at 1e-7), then transpose to (nMels, nFrames).
        let logMel = MLX.log(MLX.maximum(mel, MLXArray(Float(1e-7))))
        return logMel.T
    }

    /// HTK mel filterbank with no normalization.
    ///
    /// - mel(f) = 2595 · log10(1 + f / 700)         (HTK formula)
    /// - inverse: f(m) = 700 · (10^(m/2595) - 1)
    /// - triangle filters span [fLeft, fRight] with peak 1.0 at fCenter
    /// - no Slaney area normalization (matches Python `norm=None`)
    ///
    /// - Returns: filterbank of shape `(nMels, nFft/2 + 1)`.
    static func melFiltersHTKNoNorm(
        sampleRate: Int,
        nFft: Int,
        nMels: Int,
        fMin: Float = 0.0,
        fMax: Float? = nil
    ) -> MLXArray {
        let actualFMax = fMax ?? Float(sampleRate) / 2.0

        func hzToMel(_ hz: Float) -> Float {
            2595.0 * log10(1.0 + hz / 700.0)
        }
        func melToHz(_ mel: Float) -> Float {
            700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(actualFMax)

        // (nMels + 2) mel points → bin centers + edges for triangle filters.
        let melPoints: [Float] = (0 ... nMels + 1).map { i in
            melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
        }
        let hzPoints = melPoints.map(melToHz)

        // FFT bin → frequency lookup.
        let fftFreqs: [Float] = (0 ..< nFft / 2 + 1).map { i in
            Float(i) * Float(sampleRate) / Float(nFft)
        }

        // Build triangle filters; NO normalization.
        var bank = [Float](repeating: 0, count: nMels * (nFft / 2 + 1))
        for m in 0 ..< nMels {
            let fLeft = hzPoints[m]
            let fCenter = hzPoints[m + 1]
            let fRight = hzPoints[m + 2]
            for k in 0 ..< nFft / 2 + 1 {
                let freq = fftFreqs[k]
                let val: Float
                if freq >= fLeft && freq <= fCenter {
                    val = (freq - fLeft) / max(fCenter - fLeft, .leastNormalMagnitude)
                } else if freq > fCenter && freq <= fRight {
                    val = (fRight - freq) / max(fRight - fCenter, .leastNormalMagnitude)
                } else {
                    val = 0
                }
                bank[m * (nFft / 2 + 1) + k] = val
            }
        }

        return MLXArray(bank).reshaped([nMels, nFft / 2 + 1])
    }

    // MARK: - File loading

    /// Load an audio file at any sample rate, return a 24 kHz mono Float32 waveform.
    ///
    /// Uses `AVAudioConverter` for sample-rate conversion (same pattern as
    /// `package/Audio/AudioResampler.swift`).
    static func loadAudio(_ url: URL) throws -> MLXArray {
        let file = try AVAudioFile(forReading: url)
        let inputFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: frameCount
        ) else {
            throw MiMoAudioError.bufferAllocationFailed
        }
        try file.read(into: inputBuffer)

        // If already at 24 kHz mono, fast-path skip the converter.
        if Int(inputFormat.sampleRate) == sampleRate
            && inputFormat.channelCount == 1
            && inputFormat.commonFormat == .pcmFormatFloat32 {
            return pcmBufferToMLXArray(inputBuffer)
        }

        // Otherwise build a converter to 24 kHz mono Float32.
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw MiMoAudioError.invalidFormat
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw MiMoAudioError.invalidFormat
        }

        let ratio = Double(sampleRate) / inputFormat.sampleRate
        let outFrameCapacity = AVAudioFrameCount(Double(frameCount) * ratio + 1024)
        guard let outBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: outFrameCapacity
        ) else {
            throw MiMoAudioError.bufferAllocationFailed
        }

        var supplied = false
        var error: NSError?
        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if supplied {
                outStatus.pointee = .endOfStream
                return nil
            }
            supplied = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        if status == .error, let error {
            throw error
        }

        return pcmBufferToMLXArray(outBuffer)
    }

    private static func pcmBufferToMLXArray(_ buf: AVAudioPCMBuffer) -> MLXArray {
        // Always-mono path: take channel 0.
        let frames = Int(buf.frameLength)
        guard let channelData = buf.floatChannelData else {
            return MLXArray.zeros([0])
        }
        let pointer = channelData[0]
        let samples = Array(UnsafeBufferPointer(start: pointer, count: frames))
        return MLXArray(samples)
    }
}

enum MiMoAudioError: Error {
    case bufferAllocationFailed
    case invalidFormat
}
