# Decisions Log

These decisions came out of the clarifying questions asked on 2026-10-15, before any code
was written. Each entry gives the answer and what it means for the build. To change a
decision, edit its entry and update [phases.md](phases.md).

| # | Topic | Decision | Consequence |
|---|-------|----------|-------------|
| D1 | TTS engine | **Kokoro-82M** | Natural English, fast on GPU and fine on CPU, small model. No voice cloning. See [engine-evaluation.md](engine-evaluation.md). |
| D2 | Licensing / use | **Commercial use** (monetized YouTube) | Only permissively licensed models and voices. This rules out XTTS-v2 (CPML, non-commercial) and other non-commercial weights. Kokoro is Apache-2.0. |
| D3 | Hardware | **NVIDIA GPU (CUDA)**, RTX 5060 8 GB | Install CUDA PyTorch with **cu128 or newer** (Blackwell needs it). Fall back to CPU automatically. See [environment.md](environment.md). |
| D4 | Python | **3.12** | Pinned with `.python-version`. uv provisions it because the system Python is 3.14, which is too new for parts of the TTS stack. |
| D5 | Voice cloning | **Not needed**; preset voices only. *Revised by D13.* | Kokoro alone. The `TTSEngine` interface still allows adding another engine later. |
| D6 | MP3 export | **pip-only encoder (`lameenc`)** | No system ffmpeg needed (it isn't installed on this machine). |
| D7 | Existing output files | **Auto-suffix** (`narration_1.wav`, ...) | `--overwrite` / `overwrite=True` replaces instead. The final path is always reported. |
| D8 | Project location | **Repo root `D:\Workflow\repos\voicify`** | Uses the spec's `src/tts_agent/` layout directly at the root, not nested in `local-tts-agent/`. |
| D9 | Languages | **English only for now** (American + British) | The API keeps a `language` field. Other languages are a future improvement. |
| D10 | Packaging | **uv** | `pyproject.toml` + `uv.lock`, with a PyTorch CUDA index configured in uv (`cu130`; see Phase 2 notes). |
| D11 | Narration controls | **All four**: pause markers, pronunciation dictionary, number/abbreviation normalization, paragraph pauses | Built in Phase 8, after the MVP and long-form work. Normalization only covers gaps that misaki doesn't already handle. |
| D12 | Subtitles | **SRT produced alongside audio** (on by default) | Sentence-level timings from the chunk timeline (Phase 7). `--no-srt` turns it off. |
| D13 | Voice cloning (Phase 10) | **Chatterbox, as an optional `clone` extra** (zero-shot cloning from one clip) | MIT code and weights, so D2 still holds. XTTS-v2 and F5-TTS are non-commercial, and RVC voice conversion needs a trained model and old dependencies. `uv sync` without the extra stays Kokoro-only. Chatterbox's pins (torch 2.6, which has no Blackwell support, plus numpy<2 and exact versions of several libraries) are relaxed with `[tool.uv] override-dependencies`, and torchaudio 2.11 runs on torch 2.14 because Chatterbox only uses its pure-Python parts. Subtitles fall back to character-count timing because Chatterbox has no word timings. uv resolves extras together, and diffusers caps `huggingface-hub<2`, so the whole lock moved from huggingface-hub 2.2.0 to 1.33.0. Kokoro and the no-network test pass on it. |

## Defaults chosen without asking

These are reasonable defaults. Any of them can be changed.

| Topic | Default | Why |
|-------|---------|-----|
| Default voice | `af_heart` | Kokoro's only grade-A English voice. Confirmed by ear in Phase 4 (2026-10-15). |
| Sample rate | 24 kHz mono, 16-bit PCM WAV | Kokoro's native output, so no resampling. |
| MP3 bitrate | 160 kbps CBR at 24 kHz (192 kbps from 32 kHz up) | 192 kbps is not possible at 24 kHz (MPEG-2 maximum is 160). 160 is transparent for speech and avoids resampling. Changed in Phase 7. |
| Sentence pause | ~250 ms | Kokoro already pauses naturally inside a chunk. This only smooths chunk seams. |
| Paragraph pause | ~700 ms | Gives an audible section break in narration. |
| `[pause]` default | 1.0 s | Easy to reason about when writing scripts. |
| Max chunk size | ~400 chars | Stays under Kokoro's ~510-phoneme-token context limit with headroom. |
| Chunk failure policy | Retry once, then skip and report (`on_chunk_error="skip"`) | One bad sentence shouldn't throw away 30 minutes of narration. Exit code 3 flags it. |
| CLI framework | `argparse` | Standard library. The spec asks for no unnecessary frameworks. |
| Config | Env vars + `.env` via `python-dotenv` into a `Settings` dataclass | "Environment-based configuration when appropriate." |
| Version control | Initialize git in Phase 2 | The folder isn't a repo yet. |

## Open items to resolve during implementation

- ~~**espeak-ng on Windows:**~~ **Resolved in Phase 2:** the bundled `espeakng-loader` is enough, so
  the MSI is not needed. `KPipeline(lang_code="a")` attaches an `EspeakFallback` that loads
  `espeak-ng.dll` from the wheel. Without the fallback, misaki turns unknown words such as
  "Floogleborp" into `❓` and they are not spoken. With it they are phonemized.
- ~~**Token timestamps:**~~ **Resolved in Phase 7:** yes, and they are used for SRT timing. The
  character-proportional split was a median 0.34 s off (30% over 0.5 s). The SRT built from token
  times sits a median 0.07 s from the actual speech onsets.
- ~~**Inline phoneme syntax**~~ **Resolved in Phase 8:** `[word](/phonemes/)` works through `KPipeline`. Dictionary values written as `/phonemes/` use it.
- ~~**Default voice:**~~ **Resolved in Phase 4:** `af_heart` stays the default after listening to all 28 English voices.
