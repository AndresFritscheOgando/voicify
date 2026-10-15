# Voicify

A local text-to-speech agent for long-form narration, built on
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). It turns a sentence or a 30-minute video
script into one natural-sounding audio file (WAV or MP3), plus an `.srt` subtitle file. You can
use it from the command line or as a Python library.

- **Local and private.** Synthesis runs in-process on your CPU or NVIDIA GPU. Your text is never sent
  anywhere. The only network access is the one-time download of the model (~313 MB) and voices from
  Hugging Face. After that it makes no network connections at all.
- **Built for long scripts.** It splits text at sentence boundaries, keeps the same voice and speed
  throughout, puts natural pauses between sentences and paragraphs, retries a failed chunk, and
  reports anything it had to skip.
- **Ready for video.** It writes WAV or MP3 audio, and sentence-level SRT subtitles timed from the
  model's own word timings.
- **Script controls.** You can add exact `[pause 1.5s]` markers and keep a pronunciation dictionary,
  and common numbers and abbreviations Kokoro misreads (`Main St.`, `3:30 pm`, `5 km`, `1-10`) are
  fixed automatically.
- **Commercial-friendly model.** Kokoro-82M and its voices are Apache-2.0 (see
  [License and credits](#license-and-credits) for the libraries).

```bash
uv sync
uv run voicify --text "Hello, welcome to my channel." --output outputs/hello.wav
```

---

## 1. Project overview

```
text or .txt script
  → normalize text, read [pause] markers, split into paragraphs → sentences → chunks (≤ 400 chars)
  → for each chunk: apply pronunciations + number fixes (spoken text only) → Kokoro → audio + word timings
  → join chunks with pauses → WAV or MP3
  → word timings → sentence cues → SRT
```

| Path | What it is |
|------|------------|
| `main.py` | `python main.py ...` entry point (the same CLI as `voicify`) |
| `src/tts_agent/agent.py` | `TTSAgent`: validation, chunk loop, pauses, retries, output |
| `src/tts_agent/cli.py` | The argparse command-line interface |
| `src/tts_agent/engines/` | The `TTSEngine` interface, the Kokoro engine, and a mock engine for tests |
| `src/tts_agent/processing/` | Text chunking, audio/MP3 writing, subtitles, pronunciations, normalization |
| `docs/` | Requirements, design decisions, engine evaluation, environment, and the build log ([phases.md](docs/phases.md)) |

## 2. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **OS** | Windows 10/11, Linux, or macOS. It was developed and measured on Windows 11. |
| **[uv](https://docs.astral.sh/uv/)** | uv manages the environment, and it downloads **Python 3.12** for you. Python 3.13+ is too new for parts of the TTS stack. |
| **NVIDIA GPU** *(optional)* | It runs about 70× faster than real time on an RTX 5060, versus about 7.5× on CPU. Blackwell cards (RTX 50 series) need a CUDA 12.8+ PyTorch, which this project pins (see [Troubleshooting](#9-troubleshooting)). |
| **Disk** | 3.3 GB for the environment with CUDA PyTorch (0.9 GB with the CPU-only variant), plus about 340 MB for the model and voices. |
| **espeak-ng** | **Not needed.** The `espeakng-loader` wheel bundles it. Kokoro uses it only to pronounce out-of-dictionary words. |
| **ffmpeg** | **Not needed.** MP3 is encoded by the `lameenc` wheel. |

## 3. Local installation

```bash
git clone <this repo> voicify
cd voicify
uv sync                      # creates .venv with Python 3.12 and all dependencies
uv run voicify --help
```

`uv sync` needs no manual steps. It installs the package in editable mode, the `voicify` command,
and the spaCy English model that Kokoro's text front end needs.

**CUDA (default on Windows and Linux).** `pyproject.toml` takes PyTorch from the official `cu130`
index (CUDA 13.0 wheels), which supports every recent NVIDIA GPU, including Blackwell (sm_120).
Check it with:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**CPU only.** The same `uv sync` works on machines without an NVIDIA GPU. The CUDA build of PyTorch
falls back to the CPU automatically (`TTS_DEVICE=auto`). To skip the large CUDA download (3.3 GB
environment, versus 0.9 GB), point PyTorch at the CPU wheels before syncing. In `pyproject.toml`,
replace the `pytorch-cu130` index URL (the name can stay):

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cpu"   # was .../whl/cu130
explicit = true
```

then run the commands below. They give `torch 2.14.1+cpu`, and Voicify logs
`cuda_unavailable fallback=cpu` and runs on the CPU.

```bash
uv lock
uv sync
```

macOS always gets the regular PyPI build of PyTorch (CPU / Apple Silicon), with nothing to change.

**Configuration file (optional).**

```bash
cp .env.example .env         # then edit; every setting is described in section 8
```

## 4. Model installation and setup

There is nothing to install by hand. The first time you synthesize, Kokoro downloads its files from
Hugging Face into the standard cache:

| What | Size | When |
|------|------|------|
| `kokoro-v1_0.pth` + `config.json` | ~313 MB | First run |
| Each voice (`voices/<id>.pt`) | ~0.5 MB | The first time that voice is used (all 28 English voices total 14.7 MB) |

- **Cache location:** `~/.cache/huggingface/hub` (Windows: `C:\Users\<you>\.cache\huggingface\hub`).
  Set `HF_HOME` to move it, for example `HF_HOME=D:\hf-cache` in `.env`.
- **The first run** takes about 35 s with the download. Later, each new process spends about 10 s
  importing libraries and 1–2 s loading the model; the very first run in a fresh environment
  takes about 45 s while Python compiles its bytecode.
- **Offline:** once a file is cached, Voicify reads it straight from the cache and makes **no network
  connections**, with no setting needed. This was verified by blocking every socket. A voice you
  have never used is downloaded the first time you use it. To forbid that too, set
  `HF_HUB_OFFLINE=1`, in which case a missing file is an error that says so. To cache every English
  voice in advance:

```bash
uv run python -c "from huggingface_hub import hf_hub_download as d; from tts_agent.engines.kokoro_engine import VOICES, REPO_ID; [d(REPO_ID, 'kokoro-v1_0.pth'), d(REPO_ID, 'config.json')] + [d(REPO_ID, f'voices/{v.id}.pt') for v in VOICES]"
```

## 5. Available voices

28 English voices: 20 American and 8 British. List them any time with `voicify --list-voices`.
"Grade" is Kokoro's own estimate of each voice's training-data quality; A is best. The default is
`af_heart`. Change it with `TTS_DEFAULT_VOICE`.

| ID | Name | Accent | Gender | Grade |
|----|------|--------|--------|-------|
| `af_heart` ★ | Heart | American | female | A |
| `af_alloy` | Alloy | American | female | C |
| `af_aoede` | Aoede | American | female | C+ |
| `af_bella` | Bella | American | female | A- |
| `af_jessica` | Jessica | American | female | D |
| `af_kore` | Kore | American | female | C+ |
| `af_nicole` | Nicole | American | female | B- |
| `af_nova` | Nova | American | female | C |
| `af_river` | River | American | female | D |
| `af_sarah` | Sarah | American | female | C+ |
| `af_sky` | Sky | American | female | C- |
| `am_adam` | Adam | American | male | F+ |
| `am_echo` | Echo | American | male | D |
| `am_eric` | Eric | American | male | D |
| `am_fenrir` | Fenrir | American | male | C+ |
| `am_liam` | Liam | American | male | D |
| `am_michael` | Michael | American | male | C+ |
| `am_onyx` | Onyx | American | male | D |
| `am_puck` | Puck | American | male | C+ |
| `am_santa` | Santa | American | male | D- |
| `bf_alice` | Alice | British | female | D |
| `bf_emma` | Emma | British | female | B- |
| `bf_isabella` | Isabella | British | female | C |
| `bf_lily` | Lily | British | female | D |
| `bm_daniel` | Daniel | British | male | D |
| `bm_fable` | Fable | British | male | C |
| `bm_george` | George | British | male | C |
| `bm_lewis` | Lewis | British | male | D+ |

Voice IDs are case-insensitive. A typo fails with suggestions, for example
`Unsupported voice 'af_hart'. Did you mean: af_heart, af_sarah?`.

## 6. CLI examples

Run `uv run voicify ...` or `uv run python main.py ...`. Inside the activated `.venv`, plain
`voicify ...` works too.

```bash
# Short text to WAV (default output: outputs/narration.wav, plus outputs/narration.srt)
voicify --text "Hello, welcome to my channel."

# A script file, another voice, a little slower
voicify --file scripts/example.txt --voice am_michael --speed 0.9 --output outputs/episode1.wav

# MP3 for upload; keep a WAV copy for editing
voicify --file scripts/long_form_sample.txt --format mp3 --keep-wav

# The format can also come from the file name
voicify --file scripts/example.txt --output outputs/ep2.mp3

# Audio only, no subtitles; replace an existing file instead of writing ep2_1.mp3
voicify --file scripts/example.txt --output outputs/ep2.mp3 --no-srt --overwrite

# Pronunciation dictionary, forced CPU, quiet output
voicify --file scripts/example.txt --pronunciations scripts/pronunciations.example.txt --device cpu --quiet

voicify --list-voices
voicify --help
```

On success it prints:

```
Saved:     outputs\episode1.mp3
Subtitles: outputs\episode1.srt
Duration:  16:13.8 (973.8 s)
Voice:     af_heart
Generated: 15.0 s (58 chunks, plus 11.9 s loading)
```

| Option | Meaning |
|--------|---------|
| `--text TEXT` / `--file PATH` | The input: direct text, or a UTF-8 `.txt` script (exactly one) |
| `--voice ID` | A voice ID from section 5 (default `default` = `TTS_DEFAULT_VOICE`) |
| `--speed FLOAT` | 0.5–2.0. At 1.0 it reads about 177 words per minute; 0.85 gives about 150 |
| `--output PATH` | Output file. Missing folders are created |
| `--format wav\|mp3` | Defaults to the `--output` extension, then `wav`. Both must agree if both are given |
| `--overwrite` | Replace existing files. Without it, `name.wav` becomes `name_1.wav`, `name_2.wav`, ... |
| `--no-srt` | Skip the `.srt` file |
| `--keep-wav` | With MP3 output, also keep the WAV |
| `--pronunciations PATH` | Pronunciation dictionary (see below) |
| `--device auto\|cuda\|cpu` | Overrides `TTS_DEVICE` |
| `--verbose` / `--quiet` | Debug logging and tracebacks / warnings only |

**Exit codes:** `0` success · `1` generation error · `2` usage or validation error (bad option,
empty text, missing file, unknown voice) · `3` partial success (some chunks failed and were
skipped; the warning lists which) · `130` interrupted.

### Script controls

**Pauses.** Put a marker anywhere in the text:

```text
Welcome back. [pause] Today we visit the lighthouse. [pause 1.5s]

A blank line starts a new paragraph, which gets a longer pause. [pause 800ms] Short beat.
```

- `[pause]` = 1 s. `[pause 1.5s]` and `[pause 800ms]` are exact: the voice's own edge silence is
  trimmed, so the silence you hear is the length you asked for (measured within 30 ms).
- Malformed markers such as `[pause 2]` log a warning and use 1 s. Markers never appear in subtitles.
- Without markers, chunks are separated by `TTS_SENTENCE_PAUSE_MS` (250 ms) and paragraphs by
  `TTS_PARAGRAPH_PAUSE_MS` (700 ms). Both add to Kokoro's natural pauses, so a paragraph break
  sounds like about 1.5 s.

**Pronunciation dictionary.** A UTF-8 text file with one entry per line (example:
[`scripts/pronunciations.example.txt`](scripts/pronunciations.example.txt)):

```text
# word = how to say it
Kokoro = Koh-koh-roh
SQL = sequel
nginx/i = engine x            # /i: ignore case
Kokoro = /kˈOkəɹO/            # slashes: exact phonemes (misaki notation)
```

Matching is whole-word (`SQL` is not changed inside `SQLite`) and case-sensitive unless you add
`/i`. Entries change only what the voice says; **subtitles keep your original spelling**.

**Number and abbreviation fixes** (on by default, `TTS_NORMALIZE=false` turns them off). Kokoro
already reads years, currency, percentages, ordinals, decimals, `Dr.`/`Mr.`/`Mrs.`, and acronyms
correctly. These are added only where it was tested to fail:

| You write | Kokoro would say | Spoken as |
|-----------|------------------|-----------|
| `Main St.` · `5th Ave.` · `Elm Rd.` | "Main saint" · "av-ay" · "R D" | Main Street · 5th Avenue · Elm Road |
| `Prof. Kim` · `Lt. Dan` · `Mt. Everest` | "prof" · "L T" · "M T" | Professor · Lieutenant · Mount |
| `John Smith Jr.` · `Fig. 3` · `pp. 5` | "J R" · "fig" · "P P" | Junior · Figure 3 · pages 5 |
| `5 km` · `2 L` · `60 mph` · `70°F` | "K M" · "L" · "M P H" · "degrees F" | kilometers · liters · miles per hour · degrees Fahrenheit |
| `10:00 AM` · `3:05 pm` · `7am` | "ten zero zero" · "three: zero five" · "seven am" | 10 AM · 3 oh 5 PM · 7 AM |
| `555-1234` | "five hundred fifty five, twelve thirty four" | 5 5 5, 1 2 3 4 |
| `1-10` · `10–20` | "one-ten" · "ten twenty" | 1 to 10 · 10 to 20 |
| `World War II` · `Henry VIII` | "I I" · "V I I I" | World War 2 · Henry the 8th |
| `1/2 cup` · `5x faster` · `#1` | "one two" · "five ex" · "one" | one half · 5 times · number 1 |

## 7. Python API examples

```python
from tts_agent import TTSAgent

agent = TTSAgent()                          # reads settings from the environment / .env
result = agent.generate(
    text="Hello, welcome to my channel.",   # or file="scripts/video.txt"
    voice="default",
    speed=1.0,
    output_path="outputs/hello.wav",
    format="wav",                           # "wav" | "mp3"
    overwrite=False,
)
print(result.output_path, result.duration_seconds, result.subtitle_path)
```

`generate()` returns a `GenerationResult`:

| Field | Meaning |
|-------|---------|
| `output_path` | The file actually written (after any `_1` suffix) |
| `duration_seconds`, `sample_rate`, `format`, `voice` | About the audio |
| `generation_seconds`, `load_seconds` | Synthesis time, and the one-time model load before it |
| `chunk_count`, `failed_chunks` | How many chunks, and the 0-based indexes of any that were skipped |
| `subtitle_path`, `wav_path` | The `.srt` file and the kept WAV, if written |
| `timeline` | One `ChunkTiming(index, text, start, end, words)` per spoken chunk, with word times in seconds |

More options:

```python
from pathlib import Path
from tts_agent import Settings, TTSAgent, TTSError, UnsupportedVoiceError

# Settings in code instead of the environment (any field of Settings, see section 8)
agent = TTSAgent(settings=Settings(device="cpu", default_voice="bf_emma", paragraph_pause_ms=900))

result = agent.generate(
    file="scripts/long_form_sample.txt",
    voice="am_michael",
    output_path=Path("outputs/episode.mp3"),  # MP3 inferred from the extension
    keep_wav=True,                             # also outputs/episode.wav
    subtitles=True,                            # outputs/episode.srt (default: TTS_SUBTITLES)
    pronunciations="scripts/pronunciations.example.txt",  # default: TTS_PRONUNCIATIONS
    on_chunk_error="abort",                    # "skip" (default) or "abort"
)
if result.failed_chunks:
    print("skipped chunks:", result.failed_chunks)

for voice in agent.list_voices():             # VoiceInfo(id, name, language, gender, accent, description)
    print(voice.id, voice.accent, voice.gender)

try:
    agent.generate("Hi.", voice="nope")
except UnsupportedVoiceError as exc:           # every error is a TTSError subclass
    print(exc)
```

Errors: `EmptyTextError`, `InputFileError`, `UnsupportedVoiceError`, and `InvalidOptionError` are
input problems. `SynthesisError` (and its subclass `ModelLoadError`) and `OutputError` are runtime
problems. All of them derive from `TTSError`.

**Another engine.** `TTSAgent(engine=...)` accepts anything that implements
`tts_agent.TTSEngine` (`name`, `sample_rate`, `list_voices()`, `synthesize()`, and optionally
`synthesize_timed()` and `prepare()`). The tests use this to run without the model.

## 8. CPU/GPU configuration

The device is chosen by `TTS_DEVICE` (or `--device`):

| Value | Behavior |
|-------|----------|
| `auto` (default) | CUDA if `torch.cuda.is_available()`, otherwise CPU, with a `cuda_unavailable fallback=cpu` warning |
| `cuda` | Requires CUDA, and fails with a clear error if it is not available |
| `cpu` | Always CPU |

**Measured** on the development machine (RTX 5060 8 GB, Windows 11, `af_heart`, model already loaded):

| Workload | Device | Audio | Time | Real-time factor |
|----------|--------|-------|------|------------------|
| One 15-word sentence | CUDA | 5.6 s | 0.10 s | 0.018 |
| One 15-word sentence | CPU | 5.6 s | 0.85 s | 0.15 |
| 2,873-word script (58 chunks) | CUDA | 16.2 min | 13.7 s | 0.014 |
| 2,873-word script (58 chunks) | CPU | 16.2 min | 127.7 s | 0.131 |
| Model load (cached) | CUDA / CPU | — | 1.7 s / 2.0 s | — |

| Resource | Typical use |
|----------|-------------|
| VRAM | About 0.7 GB for a sentence, and about 1.1 GB peak for a 16-minute script |
| RAM | About 2–3 GB, plus about 5.8 MB per minute of audio held before writing (a 30-minute script is about 175 MB) |
| Output size | WAV about 2.9 MB/min (16-bit 24 kHz mono); MP3 about 1.2 MB/min (160 kbps) |

**All settings** (environment variables, `.env`, or `Settings(...)` fields; CLI flags win):

| Variable | Default | Meaning |
|----------|---------|---------|
| `TTS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `TTS_DEFAULT_VOICE` | `af_heart` | What `voice="default"` means |
| `TTS_DEFAULT_SPEED` | `1.0` | 0.5–2.0 |
| `TTS_OUTPUT_DIR` | `outputs` | Folder used when no output path is given |
| `TTS_MAX_CHUNK_CHARS` | `400` | Largest chunk sent to the model (at least 50) |
| `TTS_SENTENCE_PAUSE_MS` | `250` | Silence added between chunks |
| `TTS_PARAGRAPH_PAUSE_MS` | `700` | Silence added at paragraph breaks |
| `TTS_CHUNK_RETRIES` | `1` | Retries before a chunk counts as failed |
| `TTS_ON_CHUNK_ERROR` | `skip` | `skip` (continue, exit code 3) or `abort` |
| `TTS_PRONUNCIATIONS` | — | Pronunciation dictionary path |
| `TTS_NORMALIZE` | `true` | Number and abbreviation fixes |
| `TTS_SUBTITLES` | `true` | Write the `.srt` |
| `TTS_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `HF_HOME` | HF default | Model cache folder |
| `HF_HUB_OFFLINE` | — | `1` = never download, not even a voice used for the first time |

## 9. Troubleshooting

**CUDA is not detected (`cuda_unavailable fallback=cpu`, or `TTS_DEVICE=cuda but CUDA is not available`).**
Run `uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.
- If the version ends in `+cpu`, or `torch.version.cuda` is `None`, you have a CPU build. Run
  `uv sync` again, and check that `pyproject.toml` still points at the `pytorch-cu130` index.
- Update the NVIDIA driver. CUDA 13.0 wheels need a recent driver (this machine uses 610.62).
  `nvidia-smi` shows the driver version and the highest CUDA version it supports.

**RTX 50-series (Blackwell) errors such as `no kernel image is available for execution on the device`.**
Blackwell (sm_120) needs PyTorch built for **CUDA 12.8 or newer**. Older indexes (`cu121`, `cu124`)
install fine but cannot run on these cards. This project uses `cu130`. If you changed the index,
switch it back to `cu128` or newer, then run `uv lock` and `uv sync`.

**espeak-ng.** No system install is needed: the `espeakng-loader` wheel bundles it. Kokoro uses it
only for words missing from its dictionary, such as rare names or made-up words. If such words are
silently dropped from the audio, the bundled library did not load; run
`uv sync --reinstall-package espeakng-loader`. For specific words, the pronunciation dictionary is the
dependable fix.

**`Failed to load the Kokoro model ...` or `Could not load Kokoro voice ...` with "cannot find the
requested files in the local cache".** The file is not cached and could not be downloaded, either
because there is no connection or because `HF_HUB_OFFLINE=1` is set (the message says so). Run once
with a connection and without `HF_HUB_OFFLINE`, or use the pre-download command in section 4. If you
moved the cache, make sure `HF_HOME` is the same for both runs.

**MP3.**
- `MP3 output needs the 'lameenc' package` means you should run `uv sync` (or `uv add lameenc`).
  There is no silent WAV fallback.
- The bitrate is 160 kbps, not 192, because MP3 at Kokoro's 24 kHz cannot go higher, and audio is
  never resampled.
- The MP3 carries a standard LAME gapless header, so it decodes to exactly the WAV's length
  (checked with libsndfile/mpg123). Players that ignore the header add about 50–70 ms of silence,
  mostly at the start.

**Other.**
- **The first run is slow:** about 10 s of one-time library import (PyTorch, spaCy), plus the
  download. The CLI shows this as "plus N s loading".
- **A word is mispronounced:** add it to a pronunciation dictionary. Use a respelling, or
  `/phonemes/` for full control.
- **Garbled characters in the Windows console:** the CLI switches its output to UTF-8. When you pipe
  its output elsewhere, set `PYTHONIOENCODING=utf-8`.
- **Exit code 3:** some chunks failed after retrying and were skipped. The warning lists their
  numbers; rerun with `--verbose` to see the errors, or use `TTS_ON_CHUNK_ERROR=abort` to stop at
  the first failure.

## 10. Future improvements

- More languages (Kokoro also speaks Spanish, French, Italian, Portuguese, Hindi, Japanese, and Mandarin).
- Voice blending, for example `af_heart:60,am_michael:40`.
- A voice-cloning engine (for example Chatterbox) behind the same `TTSEngine` interface.
- Exact sentence and paragraph pauses (trimming Kokoro's own edge silence, as `[pause]` already does).
- Streaming audio to disk for very long scripts, and resuming after a crash.
- Batched or parallel chunk synthesis on the GPU.
- Loudness normalization (EBU R128) for consistent levels across videos.
- Word-level and karaoke-style captions (the word timings are already in `result.timeline`).
- A `voicify download` command that pre-caches the model and voices.
- Integration hooks for video-generation and YouTube automation pipelines.

---

## Development

```bash
uv run pytest                 # mock engine only: 403 tests, offline, about 5 s
uv run pytest -m model        # 10 real-Kokoro integration tests (needs the cached model)
uv run pytest -m "model or not model" --cov=tts_agent   # everything, with coverage (98%)
uv run ruff check . && uv run ruff format --check src tests
```

The build log, measurements, and the reasoning behind each decision are in [`docs/`](docs/), starting
with [phases.md](docs/phases.md) and [decisions.md](docs/decisions.md).

## License and credits

Speech is generated by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad. The
model, voices, `kokoro`, and the [misaki](https://github.com/hexgrad/misaki) G2P are Apache-2.0.

Libraries with other licenses, installed as wheels and used unmodified:

| Library | License | Used for |
|---------|---------|----------|
| [lameenc](https://github.com/chrisstaite/lameenc) (bundles LAME) | LGPL-3.0-or-later | MP3 encoding |
| `espeakng-loader` (bundles [espeak-ng](https://github.com/espeak-ng/espeak-ng)) | GPL-3.0 (espeak-ng) | Pronouncing out-of-dictionary words |
| PyTorch, NumPy, soundfile, spaCy | BSD / MIT | Inference, audio I/O, text analysis |

These licenses govern the software if you redistribute it. They are not a claim on the audio you
generate. Check your own distribution plans if you ship Voicify inside another product.
