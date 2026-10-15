# Target Environment

Captured on 2026-10-15 from the development machine. Re-check if the hardware or tooling changes.

## Detected

| Item | Value | Notes |
|------|-------|-------|
| OS | Windows 11 Home (10.0.26200) | Shells: PowerShell, Git Bash |
| GPU | **NVIDIA GeForce RTX 5060**, 8 GB VRAM | Blackwell architecture (sm_120) |
| NVIDIA driver | 610.62 | New enough for CUDA 12.8+ |
| System Python | 3.14.6 (only version installed) | **Too new** for the TTS stack. uv will provision 3.12 for the project |
| uv | 0.12.19 | Used for env and dependency management (D10). Provisioned CPython 3.12.14 |
| PyTorch | 2.14.1+cu130 | From the `cu130` index; `torch.cuda.is_available()` is `True`, capability (12, 0) |
| ffmpeg | Not installed | Not needed: MP3 uses `lameenc` (D6) |
| espeak-ng | Not installed, **not needed** | The `espeakng-loader` wheel bundles `espeak-ng.dll` and its data; misaki's fallback uses it (Phase 2) |
| Project folder | `D:\Workflow\repos\voicify` | Git repo since Phase 2 |

## Implications

1. **PyTorch must be a CUDA 12.8+ build (`cu128` or newer).** The project uses `cu130`, because the
   `cu128` index stops at torch 2.11. Blackwell GPUs such as the RTX 50
   series are not supported by older wheels such as `cu121`/`cu124`. On those, torch either reports
   no CUDA or fails with "no kernel image is available for execution on the device". The uv config
   will pin the PyTorch index explicitly.
2. **CPU fallback:** `TTS_DEVICE=auto` uses CUDA when `torch.cuda.is_available()` is true, otherwise CPU.
   `TTS_DEVICE=cpu` forces CPU, which is useful for checking the README instructions for users without a GPU.
3. **Disk:** the venv measured 3.3 GB with CUDA torch, or 0.9 GB with the CPU-only wheels
   (Phase 9), plus about 340 MB for the model and all 28 English voices in the Hugging Face cache.

## Expected resource use (to be measured)

| Mode | RAM | VRAM | Speed (rough expectation) |
|------|-----|------|---------------------------|
| GPU (RTX 5060) | ~2–3 GB | ~1–2 GB | Many times faster than real time |
| CPU only | ~2–3 GB | none | Around or faster than real time on a modern 6–8 core CPU |
| 30-min script output | +~175 MB (float32 buffer) | — | Final WAV about 86 MB (16-bit, 24 kHz mono) |

### Measured results

_Phase 2 numbers come from the smoke test: one 15-word sentence, voice `af_heart`. Phase 5 numbers come
from `scripts/long_form_sample.txt` (2,873 words, 58 chunks), with the model already loaded and warm._

| Test | Device | Audio length | Wall time | Real-time factor |
|------|--------|--------------|-----------|------------------|
| Model load, first run (includes ~313 MB download) | CUDA | — | 35.0 s | — |
| Model load, cached | CUDA | — | 1.7 s | — |
| Model load, cached | CPU | — | 2.0 s | — |
| First synthesis (CUDA warm-up) | CUDA | 5.55 s | 0.80 s | 0.14 |
| Warm synthesis | CUDA | 5.55 s | 0.10 s | 0.018 |
| Warm synthesis | CPU | 5.55 s | 0.85 s | 0.15 |
| Long-form, 58 chunks (Phase 5) | CUDA | 16.2 min (973.8 s) | 13.7 s | 0.014 |
| Long-form, 58 chunks (Phase 5) | CPU | 16.2 min (973.8 s) | 127.7 s | 0.131 |

Peak VRAM allocated by torch: about 690 MB for one sentence, and about 1.08 GB for the 58-chunk
long-form run. The long-form WAV is 46.7 MB, as expected for 16 minutes of 16-bit 24 kHz mono.
`af_heart` at speed 1.0 reads at about 177 words per minute. The Hugging Face cache holds about 313 MB after the
first run (model plus `af_heart`). Each other voice adds about 0.5 MB on first use; all 28 English
voices together are 14.7 MB, for about 342 MB in total (Phase 4).
