# Requirements Summary

A condensed version of the original project brief, with the decisions from
[decisions.md](decisions.md) applied. When this file and the brief disagree, the
decisions win.

## Purpose

A **local, offline-capable Python text-to-speech agent** that turns text (short or a
10–30 minute script) into natural narration. It is reusable as a CLI and as a Python
library by other tools, such as a video-generation pipeline, YouTube automation, or
other AI agents.

## Functional requirements

### Input
- Direct text (`--text`) and UTF-8 `.txt` files (`--file`).
- Long-form text, Unicode, and punctuation.
- English first. The architecture leaves room for other languages.

### Speech
- Realistic speech from a local open-source model (Kokoro-82M).
- Multiple voices, voice discovery (`--list-voices`), and voice selection.
- Natural pauses from punctuation, plus explicit pause markers and paragraph pauses.
- Adjustable speed.

### Output
- WAV (default) and MP3 (`lameenc`).
- Configurable filename and directory, with missing directories created automatically.
- **Never overwrite by accident:** existing names get a suffix unless `--overwrite` is set.
- Report the final file path.
- An SRT subtitle file next to the audio (on by default).

### Long-form
- Chunk at sentence boundaries and keep the order.
- Same voice, speed, and sample rate across chunks.
- No duplicated or missing sentences (enforced by an integrity check).
- Pauses between chunks.
- Handle a failed chunk without losing the whole run (retry, skip, report).

### Interfaces
- CLI: `--text`, `--file`, `--voice`, `--speed`, `--output`, `--format`, `--list-voices`,
  `--help`, plus `--overwrite`, `--no-srt`, `--device`, `--pronunciations`, and `--verbose`/`--quiet`.
- Python: `TTSAgent().generate(...)` returns a `GenerationResult` containing the output path,
  duration, sample rate, voice, generation time, chunk count, failed chunks, and subtitle path.

### Script controls
- `[pause]`, `[pause 1.5s]`, `[pause 800ms]` markers.
- A pronunciation dictionary file.
- Number and abbreviation normalization, covering only what Kokoro's G2P misses.
- Longer pauses at paragraph breaks.

## Non-functional requirements

- Python 3.12 (uv), type hints, `pathlib`, structured logging, and clear exceptions.
- CPU inference supported; CUDA used when available (RTX 5060, CUDA 12.8+ torch).
- Model files cached locally and not downloaded on every run. Offline after the first download.
- **No user text sent to external services.**
- Lightweight: no web UI, database, auth, microservices, or cloud services.
- `pytest` suite that runs on mock engines, so no model download is needed for unit tests.
- Commercial-safe licensing for every model and dependency.

## Definition of Done

See the checklist at the end of [phases.md](phases.md#phase-9-testing-hardening-and-documentation).
