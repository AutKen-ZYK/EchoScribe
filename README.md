# EchoScribe

Real-time speech recognition from scratch in pure C++17: microphone in, timestamped subtitles out. No PyTorch, no ONNX, no inference libraries — every operator (FFT, mel, Conv1D, Attention, LayerNorm, GELU) is hand-written.

## Status

- [x] M0: build skeleton + CLI
- [ ] M1: GGUF parsing + mmap weight loading
- [ ] M2: log-mel frontend (FFT from scratch)
- [ ] M3: operator library with unit tests
- [ ] M4: encoder forward pass
- [ ] M5: decoder + tokenizer (greedy)
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

Place the Whisper tiny model (GGUF) under `models/`:

```bash
# from https://huggingface.co/ggerganov/whisper.cpp
curl -L -o models/whisper-tiny.gguf \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
```

Model files are git-ignored; never commit weights.

## Usage

```bash
./build/echoscribe --help
# M6+: ./build/echoscribe sample.wav
# M7+: ./build/echoscribe --mic
```
