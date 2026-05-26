#!/usr/bin/env python3
# Copyright © OpenTTS contributors
# License: MIT
#
# Dump per-stage golden tensors from the Python mimo reference for the
# Swift port's parity tests.
#
# Usage:
#   python tools/dump_mimo_golden.py \
#       --model ~/.lmstudio/models/mlx-community/MiMo-V2.5-ASR-MLX \
#       --out  package/Tests/MiMoTests/Fixtures/golden \
#       --clips package/Tests/MiMoTests/Fixtures/audio
#
# Produces, per clip:
#   <clip>.mel.npy            (n_mels, n_frames)
#   <clip>.codes.npy          (8, T_tokens) int32
#   <clip>.text.json          { "text": "...", "tokens": [...] }
#
# Designed to be run by hand when the model or audio fixtures change. Outputs
# are checked in as the parity oracle.

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clips", required=True, type=Path)
    ap.add_argument("--language", default="auto",
                    help="Pass 'auto'/'zh'/'en'; matches MiMo's Python API.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading mimo from {args.model}...")
    from mlx_audio.stt import load
    model = load(str(args.model))

    # Reach into the internal components to dump per-stage tensors.
    from mlx_audio.stt.models.mimo_v2_asr.mel import log_mel_spectrogram
    from mlx_audio.stt.utils import load_audio
    import mlx.core as mx

    for wav in sorted(args.clips.glob("*.wav")):
        name = wav.stem
        print(f"  [{name}]", end=" ")

        # 1. Mel
        audio = load_audio(str(wav), sr=24000)
        mel = log_mel_spectrogram(audio)
        np.save(args.out / f"{name}.mel.npy",
                np.ascontiguousarray(np.asarray(mel)))
        print(f"mel={tuple(mel.shape)}", end=" ")

        # 2. Codes (first 8 of 20 channels).
        codes = model.audio_encoder.encode(mel, n_q=model.config.audio_channels)
        mx.eval(codes)
        np.save(args.out / f"{name}.codes.npy",
                np.ascontiguousarray(np.asarray(codes).astype(np.int32)))
        print(f"codes={tuple(codes.shape)}", end=" ")

        # 3. End-to-end transcript.
        result = model.generate(str(wav), language=args.language)
        text = getattr(result, "text", str(result))
        with open(args.out / f"{name}.text.json", "w", encoding="utf-8") as f:
            json.dump({"text": text}, f, ensure_ascii=False, indent=2)
        print(f"text={text!r}")


if __name__ == "__main__":
    main()
