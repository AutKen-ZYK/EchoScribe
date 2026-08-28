# EchoScribe

Real-time speech recognition from scratch in pure C++17: microphone in, timestamped subtitles out. No PyTorch, no ONNX, no inference libraries — every operator (FFT, mel, Conv1D, Attention, LayerNorm, GELU) is hand-written.

## Status

- [x] M0: build skeleton + CLI
- [x] M1: GGUF parsing + mmap weight loading
- [x] M2: log-mel frontend (FFT from scratch)
- [x] M3: operator library with unit tests
- [x] M4: encoder forward pass
- [x] M5: decoder + tokenizer (greedy)
- [ ] M6: file transcription end-to-end
- [ ] M7: real-time microphone mode (VAD + streaming)
- [ ] M8 (optional): SIMD / OpenMP optimization

## Architecture

```
mic / wav ─> audio ─> frontend ─> encoder ─> decoder ─> text + timestamps ─> terminal
                         ▲                      │
                         │                      └── tokenizer <── vocab
                  GGUF weights (mmap)

src/
  tensor/     f32 tensor: shape/stride/reshape
  gguf/       GGUF parser + mmap loader
  ops/        hand-written operators
  frontend/   FFT + 80-mel filterbank + log-mel
  audio/      capture (miniaudio), resample, wav reader
  encoder/    Whisper encoder
  decoder/    autoregressive decode + KV cache
  tokenizer/  BPE decode + special tokens
  streaming/  sliding window + energy VAD + incremental output
  cli/        main entry
```

## Dependencies

- CMake >= 3.16, GCC or Clang, C++17
- [miniaudio](https://github.com/mackron/miniaudio) (vendored single header, audio capture only)
- Catch2 v3 (tests only, via FetchContent)

Everything else — FFT, operators, GGUF parsing — is implemented in this repo.

## Build

```bash
cmake -B build
cmake --build build
```

## Download a model

Place a Whisper **GGUF** model under `models/`:

```bash
# whisper tiny, F16 (~78 MB), GGUF v3
curl -L -o models/whisper-tiny-F16.gguf \
  https://huggingface.co/handy-computer/whisper-tiny-gguf/resolve/main/whisper-tiny-F16.gguf
```

Notes:
- Verify with `./build/echoscribe --list-tensors` (169 tensors, arch=whisper).
- The classic `ggml-tiny.bin` from ggerganov/whisper.cpp is the **old GGML format** (magic `ggml`), *not* GGUF — do not use it.
- Quantized variants (q4/q5/q8) exist but EchoScribe currently loads f32/f16 only.

Model files are git-ignored; never commit weights.

## Reference data

`tests/data/*.f32` holds reference log-mel features and encoder outputs from the
official OpenAI whisper tiny model (f32), generated on synthetic 30 s audio:

```python
import numpy as np, torch, whisper
sr = 16000
t = np.arange(30 * sr) / sr
audio = (0.3*np.sin(2*np.pi*220*t) + 0.2*np.sin(2*np.pi*445*t)
         + 0.15*np.sin(2*np.pi*883*t)*np.exp(-t/2)
         + 0.1*np.sin(2*np.pi*(300+50*t)*t)*np.clip(np.sin(np.pi*t/2.0), 0, 1)).astype(np.float32)
model = whisper.load_model("tiny").float().eval()
mel = whisper.log_mel_spectrogram(audio)          # [80, 3000]
enc = model.encoder(mel.unsqueeze(0))             # [1, 1500, 384]
mel.numpy().tofile("tests/data/mel_ref.f32")
enc.squeeze(0).numpy().tofile("tests/data/enc_ref.f32")
```

Current agreement: log-mel max diff 6.9e-5; encoder mean abs diff 2.6e-3
(f32 accumulation-order noise through 4 transformer layers).

## Usage

```bash
./build/echoscribe --help
# M6+: ./build/echoscribe sample.wav
# M7+: ./build/echoscribe --mic
```
