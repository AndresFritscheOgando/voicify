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
| 4 | Multiple voices | Not started |
| 5 | Long-form narration | Not started |
| 6 | CLI and Python API | Not started |
| 7 | Output extras: MP3 + SRT | Not started |
| 8 | Script controls | Not started |
| 9 | Testing hardening and documentation | Not started |

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

- [ ] `agent.list_voices()` returns the English catalog.
- [ ] At least three different voices produce distinguishable audio (checked by ear).
- [ ] An unsupported voice fails with a helpful message.
- [ ] Speed changes measurably affect duration (a test with real audio, marked `@pytest.mark.model`).

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

- [ ] Unit tests cover sentence splitting, chunk-size limits, order preservation, no
      duplicates or omissions, oversized sentences, pause insertion, and partial-failure behaviour.
- [ ] A real script of about 3,000 words produces one WAV whose duration is plausible
      (about 150 words per minute), with no audible seams or repeats when spot-checked by ear.
- [ ] Generation time and the real-time factor are logged and recorded in `environment.md`.

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

- [ ] All three example commands from the spec work as written.
- [ ] CLI tests (calling `main(argv)` with a mock engine) cover each validation error.
- [ ] The API example from the spec runs unchanged.

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

- [ ] The MP3 plays and its duration is within 50 ms of the WAV.
- [ ] The SRT parses, timestamps are monotonic and non-overlapping, and the last cue ends
      no later than the audio duration.
- [ ] Spot check: subtitles stay in sync with the audio within about 0.5 s on a 5-minute script.

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

- [ ] Unit tests cover marker parsing, dictionary substitution (whole-word, not inside
      other words), and each normalization rule.
- [ ] A real-audio check confirms pause lengths are within ±50 ms of the requested value.

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

- [ ] Installs and runs locally from a clean clone.
- [ ] Text input works both directly and from a file.
- [ ] Generates audible speech.
- [ ] Voices are selectable.
- [ ] Long scripts produce one audio file.
- [ ] Audio is saved locally (WAV/MP3, plus SRT).
- [ ] No paid API and no network access after the first model download.
- [ ] The Python interface is usable from other apps.
- [ ] Automated tests pass.
- [ ] The README is complete.

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
