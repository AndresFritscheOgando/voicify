# Phase 1: TTS Engine Evaluation

**Decision: Kokoro-82M** (`kokoro` Python package, weights at `hexgrad/Kokoro-82M` on Hugging Face).

## Criteria

From the project spec, roughly in order of weight for this project:

1. Natural, non-robotic English narration quality
2. Fully local inference with offline use after the first download
3. Resource requirements (works on CPU, faster on an 8 GB consumer GPU)
4. License that allows **commercial** use (monetized YouTube)
5. Python 3.11/3.12 compatibility and ease of installation on Windows
6. Multiple voices, speed control, and active maintenance

## Candidates

| | **Kokoro-82M** | **Piper** | **Coqui XTTS-v2** | **Chatterbox** |
|---|---|---|---|---|
| Quality (English narration) | Very natural; near the top of open-weight TTS rankings for its size | Clear but noticeably synthetic | Very expressive; occasional artifacts and hallucinations on long text | Very natural, with emotion control |
| Model size | ~82M params, ~330 MB | ~20–60 MB per voice | ~1.8 GB | ~0.5B params, multiple GB |
| CPU speed | Faster than real time on modern CPUs | Much faster than real time, even on a Raspberry Pi | Slow, often slower than real time | Slow |
| GPU benefit | Large, many times faster than real time | Small | Needed in practice | Needed in practice |
| VRAM | ~1–2 GB | n/a | ~4–6 GB | ~4–8 GB |
| Voices | ~50+ presets across 9 languages (~28 English) | Many voices, one model file each | Voice cloning from a ~6 s sample | Voice cloning |
| Speed control | Yes (`speed=`) | Yes (`length_scale`) | Limited | Limited |
| License | **Apache-2.0** (code + weights) | MIT code; the newer runtime is GPL-3.0; each voice has its own dataset license | **Coqui Public Model License, non-commercial** | MIT |
| Install on Windows | `pip install kokoro` + torch | Easy (onnxruntime) | Heavy; the original Coqui company has shut down, so maintenance depends on community forks | Heavy torch stack |
| Maintenance | Active | Active (moved to OHF-Voice) | Community forks only | Active |

## Why Kokoro

- **Quality per resource:** it is the most natural-sounding option that still runs well on CPU,
  and on the RTX 5060 it should be many times faster than real time. A 30-minute script becomes a
  wait of a few minutes rather than an hour (to be measured in Phase 5).
- **License:** Apache-2.0 for both code and weights, so it is safe for monetized content (decision D2).
  XTTS-v2 is ruled out on licensing alone.
- **Simple setup:** a single pip package. Model and voice files download once from Hugging Face
  and are cached locally (`~/.cache/huggingface`). After that, inference is offline and no user
  text leaves the machine.
- **Fits the requirements:** multiple voices, a speed parameter, and punctuation-aware prosody.
  Its pipeline already splits text internally, and we add our own chunking on top for 30-minute scripts.
- **Room to grow:** the same package covers Spanish, French, Italian, Portuguese, Hindi, Japanese,
  and Mandarin when other languages are needed.

## Known limitations of Kokoro (accepted)

- **No voice cloning.** That's fine: decision D5 says preset voices only.
- **Limited emotion and style control** compared with Chatterbox or XTTS. Narration style is mostly
  set by the choice of voice.
- **Context window of about 510 phoneme tokens per pass.** Handled by our sentence-level chunking (Phase 5).
- **Some voices are clearly better than others.** Kokoro's `VOICES.md` grades them; we keep the
  higher-quality English ones visible and set a good default.
- **Grapheme-to-phoneme edge cases:** rare names and acronyms can be mispronounced. Handled by the
  pronunciation dictionary (Phase 8). Unknown words may need espeak-ng as a fallback (checked in Phase 2).
- **Python 3.13+ is not yet safe** for the full dependency chain (misaki/spaCy/torch), so we pin 3.12.

## Rejected alternatives

- **Piper:** a good fallback for very low-end hardware, but its quality is a step below Kokoro and
  this machine has a capable GPU. It could be added later as a second `TTSEngine` for a "fast draft" mode.
- **Coqui XTTS-v2:** its non-commercial license rules it out (D2), and it is heavy and unmaintained upstream.
- **Chatterbox:** excellent with an MIT license, but its main advantage (cloning and emotion) isn't
  needed (D5), and it costs far more compute. It is the leading candidate if cloning is ever added.
- **Others considered briefly:** F5-TTS (non-commercial weights), StyleTTS2 (Kokoro is built on the
  same architecture and is better packaged), Bark (slow, unpredictable for long narration), and
  MeloTTS (lower English quality than Kokoro).

> Note: the speed and VRAM figures above are typical published or community numbers and are
> approximate. The real numbers for this machine will be measured in Phases 2 and 5 and recorded in
> [environment.md](environment.md). Nothing here is claimed as verified until then.
