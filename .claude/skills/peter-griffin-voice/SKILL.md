---
name: peter-griffin-voice
description: Speak a line of text in the Peter Griffin voice cloned from the user's own sample clip, using Voicify's Chatterbox voice cloning. Use when the user runs /peter-griffin-voice with the text to say.
argument-hint: <text to say>
disable-model-invocation: true
allowed-tools: Read, Write, Bash(uv run voicify:*), Bash(ls:*)
---

# Peter Griffin voice

Turn the text in `$ARGUMENTS` into an MP3 spoken in the voice of `samples/peter-griffin.wav`, using
Voicify's voice cloning (`--voice-sample`). Run everything from the repository root.

## Settings

| Setting | Value | Why |
|---------|-------|-----|
| Voice sample | `samples/peter-griffin.wav` | The user supplies this clip; it is git-ignored |
| `--exaggeration` | `0.8` | A cartoon character needs a bigger, more animated delivery than narration (0.5) |
| `--cfg-weight` | `0.3` | High exaggeration makes Chatterbox rush; a lower weight keeps the pacing natural |
| Output folder | `outputs/peter-griffin/` | Keeps these clips apart from regular narration |

If the user asks for a calmer or wilder take, adjust `--exaggeration` (0.25–2.0) for that run.

## Steps

1. **Check the text.** If `$ARGUMENTS` is empty, ask the user what Peter should say and stop.

2. **Check the voice sample.** If `samples/peter-griffin.wav` does not exist, stop and tell the
   user to add one: about 10 seconds of a single speaker, clean (no laugh track, music or other
   characters), saved as WAV at that path. Only the first 10 seconds are used, so the best 10
   seconds matter more than length. Do not search for, download or extract audio yourself; the
   clip is the user's to provide.

3. **Write the text to a file.** Save `$ARGUMENTS` verbatim to
   `outputs/peter-griffin/<slug>.txt`, where `<slug>` is the first 5 or so words, lowercased, with
   non-alphanumerics replaced by hyphens (for example `holy-crap-lois-look-at`). Going through a
   file instead of `--text` means quotes, apostrophes and line breaks in the text cannot break the
   shell command.

4. **Generate.** This takes about 20–30 s to load the model plus a few seconds per sentence:

   ```bash
   uv run voicify --file outputs/peter-griffin/<slug>.txt --voice-sample samples/peter-griffin.wav --exaggeration 0.8 --cfg-weight 0.3 --output outputs/peter-griffin/<slug>.mp3 --no-srt --quiet
   ```

   Voicify never overwrites: if `<slug>.mp3` exists, it saves `<slug>_1.mp3` and prints the real
   name on the `Saved:` line.

5. **Report** the `Saved:` path and the duration from the output, as a clickable link to the file.

## When it fails

| Error | Fix |
|-------|-----|
| `Voice cloning needs the optional Chatterbox dependencies` | Run `uv sync --extra clone`, then retry |
| `Voice sample ... needs at least 5 s` / `Could not read voice sample` | The clip is too short or not a readable WAV; ask the user for a better one |
| `TTS_DEVICE=cuda but CUDA is not available` | Retry with `--device cpu` (much slower) |
| Mumbling, repeated words, or odd delivery | Chatterbox samples randomly; just run it again, or lower `--exaggeration` to 0.6 |

The first run ever downloads the cloning model (3.2 GB), so it can take several minutes.
