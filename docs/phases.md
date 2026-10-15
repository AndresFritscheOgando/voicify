# Implementation Phases — Voicify (Local TTS Agent)

This is the build plan for the local text-to-speech agent. Each phase ends in a
working, testable state, and no phase starts until the previous one meets its
acceptance criteria.

Related context:

- [requirements.md](requirements.md): the full requirement set, condensed
- [decisions.md](decisions.md): answers to the clarifying questions and the decisions that follow from them
- [engine-evaluation.md](engine-evaluation.md): the Phase 1 research and engine choice
- [environment.md](environment.md): the target machine's hardware and tooling
- [architecture.md](architecture.md): modules, data flow, and public API shape

## Status overview

| # | Phase | Status |
|---|-------|--------|
| 1 | Research and design | Done (approved 2026-10-15) |
| 2 | Project setup | Done, pending a listening check of the smoke WAV |
| 3 | Basic speech generation (MVP) | Done (listening check passed 2026-10-15) |
| 4 | Multiple voices | Done (listening check passed 2026-10-15) |
| 5 | Long-form narration | Done (listening spot-check passed 2026-10-15) |
| 6 | CLI and Python API | Done (2026-10-15) |
| 7 | Output extras: MP3 + SRT | Done (listening sync check passed 2026-10-15) |
| 8 | Script controls | Done (2026-10-15) |
| 9 | Testing hardening and documentation | Done (2026-10-15) |

Rules that apply to every phase:

- MVP first. Nothing from a later phase gets built early.
- No feature is reported as working until it has been run and tested on this machine.
- Unit tests use a mock engine and never download the model.
- No web UI, database, auth, services, or cloud dependencies.
- User text never leaves the machine.

---

## Phase 1: Research and design

**Goal:** choose the TTS engine and settle the design before writing code.

**Tasks**

- [x] Evaluate Kokoro, Piper, Coqui XTTS-v2, Chatterbox and others on quality, local
      execution, resource use, Python compatibility, license, and ease of install.
- [x] Pick an engine and record why ([engine-evaluation.md](engine-evaluation.md)).
- [x] Settle the open questions with the user ([decisions.md](decisions.md)).
- [x] Sketch the module layout and public API ([architecture.md](architecture.md)).
- [x] Record the target hardware ([environment.md](environment.md)).

**Deliverables:** the documents in `docs/`.

**Acceptance criteria**

- [x] The user has reviewed and approved the engine choice and the decisions.

---

## Phase 2: Project setup

**Goal:** a reproducible Python 3.12 environment in which Kokoro loads and runs on the GPU.

**Tasks**

1. Create the project skeleton at the repo root (`D:\Workflow\repos\voicify`):
   `main.py`, `src/tts_agent/...`, `scripts/`, `outputs/`, `tests/`, `docs/`.
2. `pyproject.toml` managed by **uv**:
   - `requires-python = ">=3.11,<3.13"`, pinned to 3.12 with `.python-version`
     (the system Python is 3.14, which is too new; uv installs 3.12 itself).
   - Dependencies: `kokoro`, `torch`, `soundfile`, `numpy`, `lameenc`,
     `python-dotenv`. Dev dependencies: `pytest`, `pytest-cov`, `ruff`.
   - Point the PyTorch index at **CUDA 12.8 or newer wheels (`cu128`+)**. The RTX 5060
     (Blackwell, sm_120) is not supported by older CUDA builds.
   - Use a `src/` layout with `tts_agent` installed as an editable package.
3. `.env.example` listing every config variable (see [architecture.md](architecture.md#configuration)).
4. `.gitignore`: `.venv/`, `outputs/`, `__pycache__/`, `.env`, `*.wav`, `*.mp3`, `.pytest_cache/`.
5. `git init` and make the first commit.
6. **Smoke test** (a throwaway script, not part of the package):
   - `torch.cuda.is_available()` is `True` and the device name reports the RTX 5060.
   - `KPipeline(lang_code="a")` loads, and the model downloads into the Hugging Face cache.
   - One sentence synthesizes and is written to a WAV that plays correctly.
   - Rerun with `HF_HUB_OFFLINE=1` and confirm no download happens.
7. Check whether espeak-ng is needed. Kokoro's English G2P (`misaki`) falls back to
   espeak-ng for unknown words. Find out whether the bundled `espeakng-loader` covers this
   on Windows or whether the espeak-ng MSI must be installed. Write down what you find.

**Deliverables:** the skeleton, `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, and a git repo.

**Acceptance criteria**

- [x] `uv sync` builds the environment from scratch with no manual steps.
- [x] Kokoro synthesizes audio on CUDA, and on CPU when `TTS_DEVICE=cpu` is set.
- [x] A second run works offline from the cache.
- [x] The espeak-ng requirement is settled and documented.

**Phase 2 notes (2026-10-15)**

- **PyTorch index is `cu130`, not `cu128`.** The `cu128` index stops at torch 2.11. `cu130` has
  2.14.1, and driver 610.62 supports it. This still meets "CUDA 12.8 or newer". The source is scoped to
  Windows/Linux, so macOS resolves torch from PyPI.
- **`en-core-web-sm` is a pinned dependency** (a direct wheel URL from spaCy's GitHub releases).
  Without it, misaki tries to `pip install` the spaCy model at runtime. That fails in a uv venv and
  would also break offline use.
- **espeak-ng:** the system install is not needed. See [decisions.md](decisions.md#open-items-to-resolve-during-implementation).
- **Token timestamps exist** (`result.tokens[i].start_ts/end_ts`). Phase 7 decides whether to use them.
- **Offline:** with `HF_HUB_OFFLINE=1` set, the model and voice load from the cache. No
  `TRANSFORMERS_OFFLINE` is needed.
- **Things to handle in later phases:**
  - Kokoro/torch print three harmless deprecation warnings on load (`weight_norm`, `torch.jit.script`,
    LSTM dropout). Phase 3 logging should filter them.
  - The Windows console is cp1252, so printing phoneme strings raises `UnicodeEncodeError`. The Phase 6
    CLI should reconfigure stdout/stderr to UTF-8.
  - The first CUDA synthesis takes about 0.8 s of warm-up; later calls take about 0.1 s.
- Smoke WAVs: `outputs/smoke_cuda.wav`, `outputs/smoke_cpu.wav`, `outputs/smoke_offline.wav` (git-ignored).

---

## Phase 3: Basic speech generation (MVP)

**Goal:** text in, WAV out, through a clean engine abstraction.

**Tasks**

1. `engines/base.py` defines the abstract `TTSEngine` interface:
   - `synthesize(text, voice, speed) -> np.ndarray` (float32 mono)
   - `sample_rate` property
   - `list_voices() -> list[VoiceInfo]`
   - `name` property
2. `engines/kokoro_engine.py` implements it:
   - The `KPipeline` loads lazily on first use and is reused afterwards.
   - The device comes from config: `auto`, `cuda`, or `cpu`. `auto` falls back to CPU and logs a warning.
   - The audio segments that Kokoro yields for one input are joined into a single array.
3. `processing/audio_processor.py` provides `write_wav(audio, sample_rate, path)` using
   `soundfile`. The output is 16-bit PCM mono at 24 kHz.
4. `utils/file_utils.py` provides `resolve_output_path(path, overwrite)`:
   - creates any missing parent directories
   - **auto-suffixes** existing names (`narration.wav` → `narration_1.wav` → `narration_2.wav`)
     unless `overwrite=True`
5. `config.py` holds a `Settings` dataclass loaded from the environment and `.env`, with defaults.
6. Structured logging: a single `logging` setup with level from `TTS_LOG_LEVEL`. Keep it
   key=value friendly.
7. A minimal `agent.py` with `TTSAgent.generate(text, voice, output_path)` returning `GenerationResult`.
8. Exceptions in `tts_agent/errors.py`: `TTSError` as the base class, plus
   `EmptyTextError`, `InputFileError`, `UnsupportedVoiceError`, `InvalidOptionError`,
   `SynthesisError`, and `OutputError`.

**Deliverables:** a working end-to-end path for short text.

**Acceptance criteria**

- [x] The example sentence ("Welcome to our channel...") produces an audible WAV in `outputs/`.
- [x] Empty or whitespace-only text raises `EmptyTextError`.
- [x] Missing output directories are created, and existing files are never overwritten without `overwrite=True`.
- [x] Unit tests with a `MockEngine` (it produces a sine wave or silence) cover the above.

**Phase 3 notes (2026-10-15)**

- `scripts/example.txt` → `outputs/phase3_example.wav`: 12.9 s of audio, RMS 0.051, 16-bit PCM
  mono at 24 kHz. A second call auto-suffixed to `phase3_example_1.wav`. `TTS_DEVICE=cpu` also works.
- Timing on CUDA: the first `generate()` took 11.4 s, almost all of it importing kokoro/misaki/spaCy
  (model load itself is 0.9 s). The second call took 0.23 s (RTF 0.018). `generation_seconds`
  currently includes that first-call load; Phase 5 can report load and synthesis separately.
- The model is shared across one `KPipeline` per language code (`a` American, `b` British), chosen
  from the first letter of the voice id. Non-English voice ids raise `UnsupportedVoiceError`.
- Deferred to their phases: the voice catalog lists only `af_heart` (Phase 4); no voice or speed
  validation (Phase 4); no chunking, so `chunk_count` is always 1 (Phase 5); only `.wav` output, and
  any other extension raises `InvalidOptionError` (Phases 6 and 7); the CLI is unchanged (Phase 6).
- Logging: `setup_logging(level)` configures the `tts_agent` logger with `key=value` messages and
  filters the three Kokoro/torch load warnings. The library never configures logging on its own.
- Tests: 45 mock tests in about 3 s, plus `pytest -m model` for real Kokoro (passes with `HF_HUB_OFFLINE=1`).

---

## Phase 4: Multiple voices

**Goal:** discover voices, select one, and validate it.

**Tasks**

1. A voice catalog in the Kokoro engine (`VoiceInfo`: id, display name, language,
   gender, accent, and a quality note).
   - Ship the **English voices only**: American (`a*_` prefix) and British (`b*_` prefix).
   - The catalog is a static list kept in code, checked against Kokoro's `VOICES.md`
     during this phase.
2. `voice="default"` maps to a configurable default (`TTS_DEFAULT_VOICE`, initially `af_heart`).
3. Voice validation: an unknown voice raises `UnsupportedVoiceError`, and the error lists
   close matches (`difflib.get_close_matches`).
4. Voice blending, e.g. `af_heart:60,am_michael:40`, is **out of scope**. Note it as a future improvement.
5. Voices download on first use and are cached by Hugging Face. Confirm they also work
   offline once cached.
6. `speed` parameter: accepts 0.5–2.0, default 1.0. Anything outside that range is rejected.

**Acceptance criteria**

- [x] `agent.list_voices()` returns the English catalog.
- [x] At least three different voices produce distinguishable audio (checked by ear).
- [x] An unsupported voice fails with a helpful message.
- [x] Speed changes measurably affect duration (a test with real audio, marked `@pytest.mark.model`).

**Phase 4 notes (2026-10-15)**

- The catalog has all **28 English voices** (American: 11 female, 9 male; British: 4 female, 4 male).
  It matches both `VOICES.md` and the `voices/` files in `hexgrad/Kokoro-82M`. `VoiceInfo` gained
  an `accent` field. `description` holds the overall grade from `VOICES.md`, e.g. `Grade A`.
- Voice ids are case-insensitive. `"default"` or `None` resolves to `TTS_DEFAULT_VOICE`. An unknown id
  raises `UnsupportedVoiceError` with up to three `difflib` suggestions, for example
  `Unsupported voice 'af_hart'. Did you mean: af_heart?`. Validation happens in `TTSAgent` against
  `engine.list_voices()`, so it works for any engine, and it runs before anything is synthesized or written.
- Speed must be a number within 0.5–2.0. That applies to the `speed` argument and to `TTS_DEFAULT_SPEED`.
  Booleans, NaN, and infinity are rejected.
- Real audio, `af_heart`, at speeds 0.5 / 1.0 / 2.0 gave 17.07 s / 8.55 s / 4.45 s.
- All 28 voices were downloaded once (14.7 MB together), then each was rendered with
  `HF_HUB_OFFLINE=1`. Samples are in `outputs/phase4_voices/<id>.wav`, plus `speed_*.wav`.
- `pytest -m model` (3 tests: example sentence, speed vs. duration, distinct voices) passes offline.
- Voice blending such as `af_heart:60,am_michael:40` is still out of scope (see Future improvements).

---

## Phase 5: Long-form narration

**Goal:** 10–30 minute scripts turned into one consistent audio file.

**Tasks**

1. `processing/text_processor.py`:
   - Normalize: Unicode NFC, smart quotes and dashes to plain forms, collapse whitespace,
     and keep paragraph breaks (blank lines).
   - Split into **paragraphs**, then **sentences** (a regex-based splitter that
     handles abbreviations such as `Dr.`, `e.g.`, `U.S.`, decimals, and ellipses).
   - Pack sentences into **chunks** of at most `TTS_MAX_CHUNK_CHARS` (default ~400),
     never splitting mid-sentence. A single sentence longer than the limit is split at
     `;`, `:`, `,`, then at whitespace as a last resort.
   - Each chunk carries `index`, `text`, `paragraph_index`, and `ends_paragraph`.
2. Sequential generation in `agent.py`:
   - Synthesize the chunks in order on a single engine instance with fixed voice, speed, and sample rate.
   - Insert silence between chunks: about 250 ms between sentences and about 700 ms at
     paragraph breaks (both configurable).
   - Accumulate per-chunk start and end times, which Phase 7 uses for SRT.
3. Failure handling:
   - Retry a failed chunk once (`TTS_CHUNK_RETRIES`, default 1).
   - If it still fails, log it, record it in `result.failed_chunks`, and continue. This
     behaviour is controlled by `on_chunk_error`, which can be `skip` (the default) or `abort`.
   - If **every** chunk fails, raise `SynthesisError` and write no file.
4. Memory: keep the chunks as a list of numpy arrays and concatenate once at the end.
   30 minutes of float32 audio at 24 kHz mono is about 173 MB in RAM, which is acceptable.
   Streaming the output to disk is a future improvement.
5. Progress logging: "chunk 12/240 (5%)" at INFO level.
6. Integrity check: the concatenated chunk texts, ignoring whitespace, must equal the
   normalized input. This guards against duplicated or missing sentences.

**Acceptance criteria**

- [x] Unit tests cover sentence splitting, chunk-size limits, order preservation, no
      duplicates or omissions, oversized sentences, pause insertion, and partial-failure behaviour.
- [x] A real script of about 3,000 words produces one WAV whose duration is plausible
      (about 150 words per minute), with no audible seams or repeats when spot-checked by ear.
- [x] Generation time and the real-time factor are logged and recorded in `environment.md`.

**Phase 5 notes (2026-10-15)**

- **Real run:** `scripts/long_form_sample.txt` is an original 2,873-word, 32-paragraph narration
  script, which Phase 7 can reuse. It gave 58 chunks (the longest is 399 characters), 0 failures, and
  one WAV of 16.2 minutes (`outputs/phase5_long_form.wav`, 46.7 MB). CUDA took 13.7 s (RTF 0.014);
  CPU took 127.7 s (RTF 0.131). Both produced the same duration. See [environment.md](environment.md).
- **Pace is 177 words per minute**, faster than the ~150 the plan assumed. That is Kokoro's natural
  rate for `af_heart` at speed 1.0. For a slower read, set `TTS_DEFAULT_SPEED=0.85`, which gives
  about 150 wpm.
- **Dashes, checked against misaki's phonemes:** `—` and a spaced ` - ` are both read as a pause.
  `--` loses the pause, and turning a range dash `10–20` into `10-20` makes misaki say "tentwenty".
  So normalization maps `--` and a spaced `–` to `—`, and keeps `—` and range dashes as they are.
  Smart quotes and `…` become plain `"`, `'`, and `...`.
- **Sentence splitting** needs terminal punctuation, then whitespace, then a capital letter, digit, or
  opening quote/bracket. A known abbreviation (`Dr.`, `e.g.`, `U.S.`, ...) or a single capital
  initial does not end a sentence. Abbreviations that are also everyday words (`No.`, `Sun.`,
  `Mar.`) are left out, because a missed split costs more than an extra one.
- `GenerationResult.timeline` lists a `ChunkTiming(index, text, start, end)` for every chunk that
  was synthesized; Phase 7 builds the SRT from it. Failed chunks get no entry.
- A skipped chunk leaves no audio and no pause of its own. The pause before the next chunk is the
  paragraph pause if a paragraph break came anywhere since the last chunk that succeeded.
- A model that fails to load raises the new `ModelLoadError` (a `SynthesisError`) at once, instead
  of being retried and skipped chunk by chunk.
- `TTS_MAX_CHUNK_CHARS` must be at least 50.
- `generate(..., on_chunk_error=...)` overrides `TTS_ON_CHUNK_ERROR` for a single call.
- Tests: 136 mock tests in about 3 s, including a 500-sentence run.

---

## Phase 6: CLI and Python API

**Goal:** both interfaces are complete and the inputs are validated.

**Python API** (`from tts_agent import TTSAgent`)

```python
agent = TTSAgent()                       # reads Settings from env/.env
result = agent.generate(
    text="Hello, welcome to my channel.",  # or file="scripts/video.txt"
    voice="default",
    speed=1.0,
    output_path="outputs/hello.wav",
    format="wav",                          # "wav" | "mp3" (Phase 7)
    overwrite=False,
)
result.output_path, result.duration_seconds, result.sample_rate,
result.voice, result.generation_seconds, result.chunk_count,
result.failed_chunks, result.subtitle_path  # (Phase 7)
```

- `TTSAgent(engine=...)` accepts any `TTSEngine`, so tests and other apps can inject one.
- `agent.list_voices()`.

**CLI** (`python main.py ...`, also installed as the `voicify` console script)

| Option | Behaviour |
|--------|-----------|
| `--text TEXT` | Direct text input (mutually exclusive with `--file`) |
| `--file PATH` | Read a UTF-8 `.txt` file. Error if it is missing, empty, or not `.txt` |
| `--voice ID` | Default `default`. Validated against the catalog |
| `--speed FLOAT` | 0.5–2.0 |
| `--output PATH` | Default `outputs/narration.<format>` |
| `--format wav\|mp3` | Default `wav`. If `--output` has an extension, it must agree with this |
| `--overwrite` | Replace an existing file instead of auto-suffixing |
| `--no-srt` | Skip subtitle generation (Phase 7) |
| `--keep-wav` | With `--format mp3`, also keep the WAV (Phase 7) |
| `--pronunciations PATH` | Pronunciation dictionary file (Phase 8) |
| `--device auto\|cuda\|cpu` | Override `TTS_DEVICE` |
| `--list-voices` | Print a table of voices and exit |
| `--verbose / --quiet` | Log level |
| `--help` | argparse help with examples |

- Built on `argparse` alone, with no CLI framework.
- Exit codes: `0` success, `1` generation error, `2` usage or validation error, `3` partial success (some chunks failed).
- Errors print one readable line with no traceback unless `--verbose` is set.
- On success the CLI prints the output path, duration, voice, and generation time.

**Acceptance criteria**

- [x] All three example commands from the spec work as written.
- [x] CLI tests (calling `main(argv)` with a mock engine) cover each validation error.
- [x] The API example from the spec runs unchanged.

**Phase 6 notes (2026-10-15)**

- **The spec's three example commands are not in the repo.** These three were run on real Kokoro
  (offline, CUDA), through both entry points, and all exited 0:
  `python main.py --text "Hello, welcome to my channel." --output outputs/phase6/hello.wav`,
  `voicify --file scripts/example.txt --voice am_michael --speed 0.9 --output outputs/phase6/example.wav`,
  and `voicify --list-voices`. The user accepted these three in place of the spec's exact
  commands and marked the criterion done (2026-10-15).
- The API example above ran unchanged against real Kokoro. A copy of it, using the mock engine, is
  in `test_agent.py`.
- **Options deferred to their phases:** `--no-srt` and `--keep-wav` (Phase 7) and `--pronunciations`
  (Phase 8). `--format` accepts only `wav` until Phase 7 adds `mp3` to `OUTPUT_FORMATS`.
- **API:** `generate(text=None, *, file=None, voice="default", speed=None, output_path=None,
  format=None, overwrite=False, on_chunk_error=None)`. Everything after `text` is keyword-only.
  `format` defaults to the `output_path` extension, then `wav`. A file name without an extension gets
  `.<format>` appended (so `episode.v2` would become `episode.v2.wav`).
- **Input files** (`read_text_file`): a missing file, a directory, a non-`.txt` name, an unreadable
  file, or invalid UTF-8 raises `InputFileError`. A whitespace-only file raises `EmptyTextError`.
  A UTF-8 byte-order mark is accepted.
- **CLI:**
  - Errors print as one `error: ...` line, plus a traceback with `--verbose`.
  - Exit codes are 0, 1, 2, and 3 as planned, plus 130 for Ctrl+C.
  - Usage errors (exit 2) are: argparse errors, `EmptyTextError`, `InputFileError`,
    `InvalidOptionError`, and `UnsupportedVoiceError`.
  - `--quiet` keeps warnings, so skipped chunks are still reported.
  - stdout/stderr are switched to UTF-8 for the Windows console.
  - `--list-voices` marks the configured default with `*`.
- **Load time is reported separately.** The new optional `TTSEngine.prepare(voice)` hook loads the
  model before the timer starts. `GenerationResult.load_seconds` holds that time, and
  `generation_seconds` is now synthesis only. On CUDA, a CLI run prints
  `Generated: 0.7 s (1 chunk, plus 11.8 s loading)`. The 11.8 s is mostly the one-time import of
  kokoro, misaki, and spaCy.
- **Fixed:** `.env` was searched for from `config.py`'s own folder rather than the working directory,
  so an installed package would never have found it. It now uses `find_dotenv(usecwd=True)`.
- Tests: 202 mock tests in about 3 s, including 42 CLI tests. The `pytest -m model` tests pass offline.

---

## Phase 7: Output extras (MP3 and SRT)

**Goal:** MP3 export and subtitle files for the video pipeline.

**MP3**

- Encode with `lameenc`, a pip wheel that needs no ffmpeg: 192 kbps CBR, mono, 24 kHz.
- When `--format mp3` is given, the agent writes MP3 directly. The WAV is not kept unless `--keep-wav` is set.
- If `lameenc` cannot be imported, fail with a clear message telling the user to install it.
  Never fall back silently.

**SRT**

- Write `<output_stem>.srt` next to the audio by default. `--no-srt` / `subtitles=False` turns it off.
- Use the **sentence-level** timings from the Phase 5 chunk timeline. Where a chunk holds
  several sentences, split its span between them in proportion to character count. (Kokoro
  can give token timestamps for English; if Phase 2 shows they are reliable through
  `KPipeline`, use them instead.)
- Wrap long sentences to at most 2 lines of about 42 characters, and split cues longer than
  about 7 seconds.
- `[pause]` markers from Phase 8 never appear in subtitle text.
- Pronunciation-dictionary substitutions (Phase 8) are **not** applied to subtitle text. Subtitles show the original spelling.

**Acceptance criteria**

- [x] The MP3 plays and its duration is within 50 ms of the WAV.
- [x] The SRT parses, timestamps are monotonic and non-overlapping, and the last cue ends
      no later than the audio duration.
- [x] Spot check: subtitles stay in sync with the audio within about 0.5 s on a 5-minute script.

**Phase 7 notes (2026-10-15)**

- **The MP3 is 160 kbps, not 192.** 24 kHz MP3 is MPEG-2, where 160 kbps is the maximum, and
  lameenc silently clamps 192 down to 160. 160 kbps at Kokoro's native 24 kHz avoids resampling and
  is transparent for speech. Audio at 32 kHz or higher would get 192 kbps (MPEG-1).
- **The MP3 has a gapless header.** lameenc writes bare frames, so decoders kept LAME's 576-sample
  start delay and end padding. That made the MP3 56-72 ms longer than the WAV, which failed the
  50 ms criterion. The writer now adds a Xing `Info` frame with the LAME delay/padding fields.
  libsndfile/mpg123 then decode it to the **exact** WAV length, sample-aligned. That was checked
  with clicks at known positions at 22.05, 24, 44.1, and 48 kHz, and by a unit test.
- **Real checks** use a 5.1-minute excerpt of `scripts/long_form_sample.txt` (910 words, 18 chunks):
  - The MP3 and the WAV are both 305.450 s (0.0 ms difference), with waveform correlation 0.999.
    The MP3 is 6.1 MB and the WAV is 14.7 MB.
  - The SRT has 89 cues, ordered and non-overlapping. The last cue ends at 305.350 s, inside the
    audio. The longest line is 44 characters and the longest cue is 5.3 s.
  - **Sync, measured objectively:** for the 79 cues that follow a pause, the cue start is a median
    0.07 s from the speech onset in the audio (worst 0.12 s), and none is more than 0.5 s off.
  - The full 16.2-minute script also gives a 0.0 ms MP3/WAV difference.
  - Listening files: `outputs/phase7/excerpt.mp3` + `excerpt.srt` (+ `excerpt.wav`), and
    `outputs/phase7/long_form.mp3` + `.srt`.
- **SRT timing uses Kokoro's word timestamps, not character proportions.** Splitting each chunk's
  span by character count put sentence starts a median 0.34 s off, with a p95 of 0.92 s and 30%
  over 0.5 s, which fails the criterion. misaki's token timestamps come from the model's own
  predicted durations, and all of them matched the text. The engine API gained
  `synthesize_timed()` and `WordTiming`; `ChunkTiming.words` carries the times on the output
  clock. Engines without timings fall back to the proportional split.
- **Cue layout:** one cue per sentence. A sentence over 84 characters or 7 s is split by a small
  optimizer that balances cue lengths and prefers breaks after `, ; :` or a dash. This avoids
  stranded tails like "…that most / maps forget to name." Lines are wrapped into two balanced
  lines of about 42 characters. Cues shorter than 0.3 s are stretched when there is room. An
  unvoiced sentence with no room joins the previous cue, so subtitle text is never dropped.
- **File naming:** `<stem>.srt`, plus `<stem>.wav` with `--keep-wav`, sit next to the audio. When
  auto-suffixing, a name counts as taken if any file in the set exists, so the audio, SRT, and WAV
  always share one stem. `--keep-wav` without MP3 output is a usage error.
- **API:** `generate(..., subtitles=None, keep_wav=False)`. `subtitles=None` follows `TTS_SUBTITLES`.
  `GenerationResult` gained `wav_path`. **CLI:** `--format mp3`, `--no-srt`, `--keep-wav`, and new
  `Subtitles:` and `WAV copy:` lines in the output.
- If `lameenc` is missing, MP3 output raises `OutputError` (CLI exit 1) with an install hint. There
  is no silent WAV fallback.
- Tests: 262 mock tests in about 4.5 s (18 SRT unit tests, 20 output integration tests, MP3
  alignment tests, and 9 more CLI tests), plus 4 `-m model` tests including real word timings.

---

## Phase 8: Script controls

**Goal:** give narrators control over pauses and pronunciation.

1. **Pause markers:** inline `[pause]` (default 1.0 s), `[pause 1.5s]`, or `[pause 800ms]`.
   - These are parsed out before chunking. A marker forces a chunk boundary and inserts exact silence.
   - Malformed markers log a warning and are treated as `[pause]`.
2. **Paragraph pauses:** blank lines mark paragraph breaks and get a longer pause
   (`TTS_PARAGRAPH_PAUSE_MS`, default 700). This is formalized from Phase 5 and made configurable.
3. **Pronunciation dictionary:** `pronunciations.txt` (or `.toml`), one entry per line,
   for example `Kokoro = Koh-koh-roh` or `SQL = sequel`.
   - Matching is whole-word and case-sensitive by default, with an optional case-insensitive flag per entry.
   - Applied only to the text sent to the engine, never to the subtitles.
   - The path comes from `--pronunciations PATH` or `TTS_PRONUNCIATIONS`.
   - Optional advanced form: Kokoro/misaki inline phoneme syntax `[word](/fəˈnɛtɪk/)`.
     Check during this phase that it works through `KPipeline`.
4. **Number and abbreviation normalization:**
   - First, test what misaki already gets right: years, ordinals, currency, percentages,
     decimals, `Dr.`/`Mr.`/`St.`, and acronyms.
   - Only write rules for the cases it gets wrong, and back each rule with a test.
   - A config flag turns it off.

**Acceptance criteria**

- [x] Unit tests cover marker parsing, dictionary substitution (whole-word, not inside
      other words), and each normalization rule.
- [x] A real-audio check confirms pause lengths are within ±50 ms of the requested value.

**Phase 8 notes (2026-10-15)**

- **Pause markers, real audio:** a script with `[pause 1s]` at the start, then 1 s, 1.5 s, 800 ms,
  2 s, 300 ms and 3 s markers, and `[pause 500ms]` at the end, was rendered with `af_heart`,
  `am_michael`, and `bf_emma` (`outputs/phase8/pauses_*.wav`). The silence measured at each marker
  (5 ms RMS windows at −40 dBFS) is within **0–30 ms** of the request; the worst is 30 ms on
  `am_michael`. At a stricter −50 dBFS threshold, every marker, the lead-in, and the tail measure
  exactly as requested.
- **Exact pauses need trimming.** Kokoro adds its own silence: about 0.2–0.3 s before speech and
  some after. Next to a marker, that edge silence is cut (RMS over 0.01, with a 5 ms fade so there
  is no click) and the exact marker silence goes in its place. Ordinary sentence and paragraph
  pauses still keep Kokoro's silence, as approved in Phase 5. So a 700 ms paragraph pause
  *sounds* like about 1.4–1.8 s. Making those exact as well would be a small change, left for a
  later decision.
- **Marker rules:**
  - Markers are case-insensitive, and consecutive markers add up.
  - A marker replaces the sentence or paragraph pause at its boundary.
  - Markers at the start or end of the text give leading or trailing silence.
  - A marker next to a chunk that failed still applies.
  - `[pause 2]` (no unit) and other malformed markers log `malformed_pause_marker` and use 1 s.
  - Other bracketed text, such as `[briefly]`, is spoken as written.
- **Pronunciation dictionary** (`processing/pronunciations.py`, example in
  `scripts/pronunciations.example.txt`):
  - The file has one `word = respelling` per line, with `#` for comments.
  - `/i` after the word ignores case.
  - A value in slashes (`Kokoro = /kˈOkəɹO/`) becomes misaki's inline phoneme syntax
    `[Kokoro](/kˈOkəɹO/)`. **That syntax works through `KPipeline`**: "Kokoro" was read
    `kəkˈɔɹO` by default and exactly as given with it.
  - Matching is whole-word, using lookarounds so keys like `C++` and `Dr. Who` work. Longer
    keys win.
  - Errors name the line (`words.txt:3: ...`), and a later duplicate wins with a warning.
  - `.toml` is not supported; the plain format covers the need.
- **Normalization** (`processing/normalizer.py`, `TTS_NORMALIZE=false` turns it off):
  - misaki was probed on years, ordinals, currency, percentages, decimals, titles, acronyms,
    times, units, phone numbers, ranges, Roman numerals, and fractions.
  - **Left alone (already correct):** years, ordinals, `$`/`£`/`€`, `%`, decimals,
    `Dr.`/`Mr.`/`Mrs.`/`Ms.`, `St.` before a name, `No.`, `vs.`, acronyms (NASA, FBI, SQL),
    `lbs`, `3 pm`, and negative numbers.
  - **Rules added, one per observed failure:**
    - `St.` after a name becomes Street; `Ave.`/`Rd.`/`Blvd.` are expanded.
    - Titles before a name are expanded: `Prof.`, `Gov.`, `Gen.`, `Capt.`, `Lt.`, `Sgt.`,
      `Rev.`, `Hon.`, `Mt.`, `Ft.`.
    - `Jr.`/`Sr.` are expanded.
    - So are `Fig.`, `Vol.`, `Ch.`, `Sec.`, `Eq.`, `pp.`, `Dept.`, `Univ.`, `Approx.`,
      `Inc.`, and `Corp.`. A sentence-final period is kept.
    - 21 units after a number (km, kg, ml, L, ft, mi, mph, km/h, GB, °C, ...), singular after
      1 or "a".
    - Clock times: `3:30 pm` becomes "3 30 PM", `9:00` "9 o'clock", `3:05` "3 oh 5", and
      `7am` "7 AM".
    - Phone numbers are read digit by digit.
    - Numeric ranges: `1-10` and `10–20` become "to".
    - Roman numerals II–XXXIX after a capitalized word: "World War 2", but "Henry the 8th".
    - Common fractions (`1/2`, `3/4`, ...), `5x` as "5 times", and `#1` as "number 1".
  - Left out on purpose, as ambiguous: `5 in`, `5 m`, dates, `4/5`, ISO dates, and version
    numbers.
  - Each rule has a test, plus a "left alone" list. A `-m model` test checks misaki's phonemes
    before and after for five of the rules.
- **Original spelling in subtitles:** the dictionary and normalization rewrite only what the engine
  reads (`processing/spoken.py`). Each rewrite records its offsets, so the engine's word timings
  are mapped back onto the original words. The SRT shows `Main St.`, `3:30 pm`, and `555-1234`,
  timed correctly. The dictionary is applied first, so its entries win over normalization.
- `generate(..., pronunciations=PATH)`, `--pronunciations PATH` / `TTS_PRONUNCIATIONS`,
  `TTS_NORMALIZE`, and `--help` documents the markers. A bad dictionary is a usage error (exit 2)
  raised before any synthesis.
- Tests: 386 mock tests in about 4.5 s (124 new for script controls, including 48 normalization
  cases and 18 "left alone" cases) and 9 `-m model` tests.

---

## Phase 9: Testing hardening and documentation

**Goal:** meet the Definition of Done.

**Tests:** must cover every item from the spec.

| Required test | Where |
|---------------|-------|
| Empty text input | `test_agent.py`, `test_cli.py` |
| Invalid file paths | `test_cli.py`, `test_file_utils.py` |
| Unsupported voices | `test_agent.py` |
| Text chunking | `test_text_processor.py` |
| Correct chunk ordering | `test_text_processor.py`, `test_agent.py` |
| Audio file generation | `test_audio_processor.py` |
| Output directory creation | `test_file_utils.py` |
| Long-form narration | `test_agent.py` (mock engine, about 500 sentences) |
| Audio concatenation | `test_audio_processor.py` |
| Generation failures | `test_agent.py` (`FailingMockEngine`) |
| MP3 / SRT / markers / dictionary | the matching test modules |

- The default `pytest` run uses mocks only and must pass offline in seconds.
- `pytest -m model` runs the real-Kokoro integration tests (opt-in, needs the cached model).
- `ruff check` is clean.

**README.md:** the 10 required sections

1. Project overview
2. Prerequisites: Windows/Linux/macOS, Python 3.12 through uv, an optional NVIDIA GPU, and espeak-ng if Phase 2 found it necessary
3. Local installation (`uv sync`), with CUDA and CPU-only variants
4. Model installation and setup (first-run download, cache location, offline mode)
5. Available voices (table)
6. CLI examples
7. Python API examples
8. CPU/GPU configuration and the hardware/memory table
9. Troubleshooting (CUDA not detected, Blackwell needing cu128+, espeak-ng, offline errors, MP3)
10. Future improvements

**Definition of Done checklist**

- [x] Installs and runs locally from a clean clone.
- [x] Text input works both directly and from a file.
- [x] Generates audible speech.
- [x] Voices are selectable.
- [x] Long scripts produce one audio file.
- [x] Audio is saved locally (WAV/MP3, plus SRT).
- [x] No paid API and no network access after the first model download.
- [x] The Python interface is usable from other apps.
- [x] Automated tests pass.
- [x] The README is complete.

**Phase 9 notes (2026-10-15): evidence for each item**

| Definition of Done | Evidence |
|--------------------|----------|
| Clean clone | `git clone`, then `uv sync` (9 s with a warm uv cache), gives `torch 2.14.1+cu130`, CUDA available, all tests passing, and ruff clean. `voicify --file scripts/example.txt --format mp3` and `python main.py --text ...` both produced audio offline (exit 0). The first run in a fresh environment spent 42.6 s loading (bytecode compilation). |
| Text and file input | `--text` / `--file` and `generate(text=...)` / `generate(file=...)`, with CLI and API tests and the README examples |
| Audible speech | Phases 3–8 listening checks, plus the `-m model` RMS check |
| Voices selectable | 28 English voices, `--voice` / `voice=`, three checked by ear in Phase 4 |
| Long scripts → one file | 2,873 words become one 16.2-minute file (Phase 5) |
| WAV/MP3 + SRT saved | Phase 7: MP3 is the exact WAV length, and the SRT is in sync (confirmed by ear) |
| No network after download | **Fixed in this phase.** Kokoro called `hf_hub_download` on every load, which made three metadata requests to huggingface.co per run unless `HF_HUB_OFFLINE=1` was set. `KokoroEngine` now resolves cached files itself (`model_file()`: `try_to_load_from_cache`, and downloads only when a file is missing). With every socket connect blocked, generation succeeds with **0 connection attempts**, with and without `HF_HUB_OFFLINE`. This is pinned by a `-m model` test. No paid API is used anywhere. |
| Python interface | `from tts_agent import TTSAgent`, with engine injection. Both README Python examples run as written. |
| Automated tests | 403 mock tests in about 5 s (with `HF_HUB_OFFLINE=1`), 10 `-m model` tests, and ruff clean. Coverage is 95% for the mock suite and 98% with the model tests included. |
| README complete | All 10 required sections, plus Development and License and credits. All 8 `voicify` commands and both Python blocks in it were executed as written (exit 0). |

- **Test gaps filled:** `concatenate()` and `silence()` (the "audio concatenation" row had no direct
  test), MP3 and SRT write errors, default Kokoro engine construction, an over-long single word and
  an unvoiced first cue in subtitles, token skipping in `map_words`, a non-UTF-8 dictionary, and the
  cache-or-download logic.
- **The CPU-only variant was verified** by following the README in a fresh clone: switching the index
  URL to `/whl/cpu`, then `uv lock && uv sync` (23 s), gave `torch 2.14.1+cpu`, `cuda_unavailable
  fallback=cpu`, and audio generated. The environment is **0.9 GB, versus 3.3 GB** with CUDA.
- **Offline error message:** a missing file under `HF_HUB_OFFLINE=1` now adds "unset HF_HUB_OFFLINE
  and run once with a network connection" to huggingface_hub's generic message.
- **Licenses:**
  - The model, voices, kokoro, and misaki are Apache-2.0.
  - `lameenc` (bundles LAME) is LGPL-3.0-or-later.
  - `espeakng-loader` bundles espeak-ng, which is GPL-3.0.
  - These are software licenses on unmodified wheels. They do not change decision D2, which is
    about the model and voices used to generate audio. The README states this, together with a
    note for anyone redistributing the software.

---

## Future improvements (out of scope for now)

- Additional languages: Spanish, French, Italian, Portuguese, Hindi, Japanese, Mandarin
  (Kokoro supports them; some need espeak-ng or extra G2P packages).
- Voice blending (weighted mixes of Kokoro voices).
- A pluggable voice-cloning engine (e.g., Chatterbox) behind the same `TTSEngine` interface.
- Streaming output to disk for very long scripts, and resuming after a crash.
- Batched or parallel chunk synthesis on the GPU.
- Loudness normalization (EBU R128) for consistent levels across videos.
- Word-level subtitle timing and karaoke-style captions.
- Integration hooks for the video-generation and YouTube automation pipelines.
