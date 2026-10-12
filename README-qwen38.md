# Qwen3.8-27B (IQ3_S) specialized llama.cpp build

Private fork specialized to serve **one** model on **one** machine:

- Model: Unsloth `Qwen3.8-27B-UD-IQ3_S.gguf` (architecture `qwen35`, hybrid DeltaNet + full attention).
- Hardware: Windows 10 x64, Ryzen 9 5900X, RTX 3090 (24 GiB), CUDA 13.4, MSVC 19.44.
- Serving: text-only, OpenAI-compatible chat/completions, streaming, tool calling, structured output.

This is NOT upstream-compatible. It deletes unused model and backend code and refuses non-qwen35
models and multimodal input.

## Runtime profiles

One flag, `-Speculative`, selects the whole profile:

```
# default profile: full context, full quality
--ctx-size 262144 --temp 1.0 --parallel 1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
--n-gpu-layers 99 --fit off --batch-size 2048 --ubatch-size 1024 --threads 8
--no-context-shift --jinja --no-webui

# -Speculative profile: faster generation (adds MTP self-speculative decoding)
--ctx-size 196608 --temp 0.7 --spec-type draft-mtp --spec-draft-n-max 2
--spec-draft-type-k q8_0 --spec-draft-type-v q8_0   (other flags unchanged)
```

ubatch 1024 is ~1.5% faster at prompt processing than 512 and keeps the same ~2.2 GiB VRAM free at
full context; ubatch 2048 was rejected because it shrinks headroom to ~1.5 GiB. Full GPU offload fits
with ~2.2 GiB VRAM free. Do not lower context, cache precision, or
Flash Attention; they are required and were validated together.

## Build

Requires VS 2022 Build Tools (MSVC 14.44), CUDA 13.4, and the bundled CMake/Ninja.

Configure + build the production server:

```
cmake --preset qwen38-prod
cmake --build build-qwen38-prod --target llama-server
```

Build flags of note (see `CMakeUserPresets.json`):
- `CMAKE_CUDA_ARCHITECTURES=86-real` (RTX 3090 machine code only).
- `GGML_AVX2`/`GGML_BMI2` on, AVX-512 off (Zen 3), `GGML_NATIVE=OFF` for reproducibility.
- `GGML_CUDA_FA_QUANTS=q8_0-q8_0` (only the Q8_0 Flash Attention kernels, plus f16-f16).
- `GGML_CUDA_GRAPHS=ON`, `BUILD_SHARED_LIBS=OFF` (static libs), UI/OpenSSL/subprocess/LLGuidance off.
- `LLAMA_SERVER_TEXT_ONLY=ON` (refuses `--mmproj`).

The `qwen38-baseline` preset additionally builds tests and `llama-bench` for validation.

## Run

```
# default: 262144 context, temp 1.0, full quality
pwsh scripts/serve-qwen38.ps1

# fast: 196608 context, temp 0.7, MTP self-speculative decoding
pwsh scripts/serve-qwen38.ps1 -Speculative
```

The launcher checks the model path, puts the CUDA 13.4 `bin\x64` on PATH, binds to 127.0.0.1:8080,
and applies the selected profile. Override any default with `-ModelPath`, `-Port`, `-CtxSize`,
`-Temperature`, `-SpecDraftNMax`, etc.

## Speculative decoding

The model ships an MTP/NextN head, so self-speculative decoding needs no separate draft model.
`-Speculative` turns it on and switches to the fast profile (196608 context, temp 0.7).

Measured generation speedup at 196608 (vs ~39.8 tok/s without spec), same prod binary:

| temperature | spec tok/s | speedup | draft accept |
| --- | --- | --- | --- |
| 0 (greedy) | ~64.8 | 1.63x | 77% |
| 0.7 (fast profile) | ~58.6 | 1.47x | 65% |
| 1.0 | ~57.7 | 1.45x | 62% |

Speculative decoding is distribution-preserving here, so it stays a win even at temp 1.0 (~1.45x);
it does not require a low temperature. The MTP draft context adds ~2 GiB of VRAM, so at the full
262144 context only ~191 MiB stays free (OOM risk on a GPU that also drives the display) - the
launcher warns if you force `-CtxSize` above 196608 with `-Speculative`. At 196608 it fits with
~2.5 GiB free.

## Runtime dependencies

`llama-server.exe` links the project libraries statically but needs these DLLs at run time
(all already present on this machine):

- CUDA 13.4: `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll` (in the toolkit `bin\x64`, on PATH).
- Driver: `nvcuda.dll` (system).
- MSVC runtime: `MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, `VCOMP140.dll` (OpenMP).

## What was pruned

- `src/models/`: kept only `qwen35.cpp` and `delta-net-base.cpp`; deleted 155 other model sources.
  The model factory serves only `qwen35` and throws for any other architecture.
- `ggml/src/`: kept only `ggml-cpu` and `ggml-cuda`; deleted 16 other backend directories.
- `tools/mtmd/models/`: kept only `models.h`; deleted 45 vision/audio graph sources and narrowed the
  clip graph factory. mtmd still links (its types back the server), but no projector can load.

## Behavior

- Non-qwen35 GGUF: rejected ("unsupported model architecture").
- `--mmproj`: rejected ("text-only server build: --mmproj is not supported").
- Image/audio/video in a request: rejected ("image input is not supported ...").
- Prompt over 262144 tokens: rejected with HTTP 400, not truncated.

## Measured results (RTX 3090)

- Load (full ctx, full offload): ~22.1 GiB VRAM used, ~2.2 GiB free.
- Prompt processing (depth 0): ~1144 tok/s (ubatch 512).
- Generation: ~38 tok/s at low depth, ~13 tok/s near 262k depth.
- Near-limit: 261900-token prompt prefilled and generated with no OOM, no truncation.

See `.copilot` session artifacts (`baseline-results.md`, `validation-results.md`, `tuning-results.md`)
for the full run details.
