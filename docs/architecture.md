# Architecture

This is a deliberately small design: plain classes and functions, one engine
abstraction, and no frameworks.

## Layout

```
voicify/                         # repo root (decision D8)
├── main.py                      # thin CLI entry → tts_agent.cli:main
├── src/tts_agent/
│   ├── __init__.py              # exports TTSAgent, GenerationResult, VoiceInfo, errors
│   ├── agent.py                 # TTSAgent: orchestration (validate → process → synthesize → write)
│   ├── cli.py                   # argparse CLI, exit codes, user-facing messages
│   ├── config.py                # Settings dataclass from env/.env
│   ├── errors.py                # TTSError hierarchy
│   ├── logging_setup.py         # structured logging config
│   ├── engines/
│   │   ├── base.py              # TTSEngine ABC, VoiceInfo
│   │   ├── kokoro_engine.py     # Kokoro implementation (lazy model load, device select)
│   │   └── mock_engine.py       # deterministic engine used by tests (not shipped as a default)
│   ├── processing/
│   │   ├── text_processor.py    # normalize, markers, sentences, chunks, pronunciations
│   │   ├── audio_processor.py   # silence, concat, WAV/MP3 writing, duration
│   │   └── subtitles.py         # chunk timeline → SRT
│   └── utils/
│       └── file_utils.py        # input file reading, output path resolution/auto-suffix
├── scripts/example.txt
├── outputs/                     # git-ignored
├── tests/
├── docs/
├── pyproject.toml / uv.lock / .python-version
├── .env.example / .gitignore
└── README.md
```

Two files have been added to the brief's suggested layout: `cli.py` keeps `main.py` trivial, and
`subtitles.py` is needed for decision D12. The spec's `engines/local_engine.py` is named
`kokoro_engine.py` to make its purpose clear.

## Data flow

```
text / .txt file
   │  read + validate (EmptyTextError, InputFileError)
   ▼
TextProcessor
   normalize → extract [pause] markers → split paragraphs → sentences
   → pack into chunks (≤ max chars, sentence boundaries) → apply pronunciations (engine text only)
   ▼
list[Chunk]  (index, engine_text, display_text, pause_after_ms)
   │
   ▼
TTSAgent loop (sequential, one engine instance, fixed voice/speed)
   for chunk: engine.synthesize() → retry once on failure → skip/abort policy
   append audio + silence(pause_after_ms); record (start, end) timeline
   ▼
AudioProcessor: concatenate → write WAV or MP3 to resolved path
Subtitles:      timeline + display_text → .srt
   ▼
GenerationResult
```

## Key interfaces

```python
class TTSEngine(ABC):
    name: str
    sample_rate: int
    def list_voices(self) -> list[VoiceInfo]: ...
    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray: ...  # float32 mono

@dataclass(frozen=True)
class VoiceInfo:
    id: str            # "af_heart"
    name: str          # "Heart"
    language: str      # "en-us" / "en-gb"
    gender: str        # "female" / "male"
    description: str = ""

@dataclass
class GenerationResult:
    output_path: Path
    duration_seconds: float
    sample_rate: int
    voice: str
    generation_seconds: float
    chunk_count: int
    failed_chunks: list[int]
    subtitle_path: Path | None
    format: str
```

## Error hierarchy

```
TTSError
├── EmptyTextError
├── InputFileError         # missing / unreadable / wrong extension / not UTF-8
├── UnsupportedVoiceError  # includes suggestions
├── InvalidOptionError     # speed out of range, format mismatch
├── SynthesisError         # all chunks failed, or model failed to load
└── OutputError            # cannot write / encode (e.g., lameenc missing)
```

## Configuration

Read from environment variables or `.env`. CLI flags take precedence.

| Variable | Default | Meaning |
|----------|---------|---------|
| `TTS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `TTS_DEFAULT_VOICE` | `af_heart` | What `voice="default"` resolves to |
| `TTS_DEFAULT_SPEED` | `1.0` | 0.5–2.0 |
| `TTS_OUTPUT_DIR` | `outputs` | Used when `--output` is not given |
| `TTS_MAX_CHUNK_CHARS` | `400` | Chunk size limit |
| `TTS_SENTENCE_PAUSE_MS` | `250` | Silence between chunks |
| `TTS_PARAGRAPH_PAUSE_MS` | `700` | Silence at paragraph breaks |
| `TTS_CHUNK_RETRIES` | `1` | Retries per failed chunk |
| `TTS_ON_CHUNK_ERROR` | `skip` | `skip` / `abort` |
| `TTS_PRONUNCIATIONS` | _(none)_ | Path to the pronunciation dictionary |
| `TTS_SUBTITLES` | `true` | Write the SRT next to the audio |
| `TTS_LOG_LEVEL` | `INFO` | Logging level |
| `HF_HOME` | HF default | Model cache location (standard Hugging Face variable) |
| `HF_HUB_OFFLINE` | _(unset)_ | Set to `1` to forbid network access after the first download |

## Privacy

Kokoro runs entirely in-process. The only network traffic is the one-time download of model and
voice files from Hugging Face; the text is never sent anywhere. Setting `HF_HUB_OFFLINE=1` enforces
this after the first run.
