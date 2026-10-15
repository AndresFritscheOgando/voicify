# Step-by-step guide: from a text script to narration audio

This guide takes you from a written script to a finished voice-over (WAV or MP3), plus a subtitle
file. It uses the `voicify` command, and every step can be done from a terminal.
For the full reference (all options, all voices, troubleshooting), see the [README](../README.md).

> **The short version**, once everything is set up:
>
> ```bash
> uv run voicify --file scripts/my_script.txt --output outputs/my_script.mp3
> ```

---

## Step 1: Install the tools (once)

1. **Install uv**, the tool that manages Python for this project.
   - Windows (PowerShell):
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - macOS / Linux:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   Close and reopen the terminal afterwards, so the `uv` command is found.

2. **Get the project and install it.** In a terminal, go to the project folder and run:
   ```bash
   uv sync
   ```
   This downloads Python 3.12 and everything else the project needs into a `.venv` folder. It takes
   a few minutes the first time, mostly for PyTorch (about 3 GB).

3. **Check that it works:**
   ```bash
   uv run voicify --help
   ```
   You should see the list of options. If you have an NVIDIA graphics card, also check that it
   is detected:
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```
   `True` means the GPU will be used, which is about 70 times faster than real time. `False` is
   fine too: it runs on the CPU, about 7 times faster than real time.

## Step 2: Do a first test run (downloads the voice model once)

```bash
uv run voicify --text "Hello, welcome to my channel." --output outputs/hello.wav
```

- The first run downloads the Kokoro model (about 313 MB) and takes 30–60 seconds. Later runs
  start in about 10–15 seconds.
- Open `outputs/hello.wav` and listen. If you hear the sentence, everything works.
- Your text never leaves your computer. The only downloads are this model, and each voice
  (about 0.5 MB) the first time you use it. After that, everything is read from the local cache
  and no internet connection is needed.

## Step 3: Prepare your script

1. **Save the script as a plain text file** with the `.txt` extension, encoded as **UTF-8**. In
   Notepad, choose *Save as → Encoding: UTF-8*. VS Code uses UTF-8 by default. A good place is the
   `scripts/` folder, for example `scripts/my_script.txt`.

2. **Write it the way it should be read:**
   - Separate paragraphs with a **blank line**. Each paragraph break gets a longer pause.
   - Use normal punctuation. Commas, periods, question marks and dashes (`—`) shape the delivery.
   - Long scripts (10–30 minutes) are fine. They are split into small pieces automatically and
     joined back into one file.

3. **Add pauses where you want them** (optional). Type a marker anywhere in the text:

   ```text
   Welcome back to the channel. [pause] Today we visit the lighthouse at Gull Point.

   It was built in 1912. [pause 1.5s] And it has never once gone dark.
   ```

   | Marker | Pause |
   |--------|-------|
   | `[pause]` | 1 second |
   | `[pause 1.5s]` | 1.5 seconds |
   | `[pause 800ms]` | 0.8 seconds |

   The markers are not spoken, and they do not appear in the subtitles.

4. **You do not need to spell out numbers and abbreviations.** Things like `$3.50`, `1999`,
   `50%`, `Dr. Smith`, `Main St.`, `3:30 pm`, `5 km`, `1-10`, `World War II` and phone numbers are
   read correctly. Subtitles keep exactly what you wrote.

## Step 4: Choose a voice

1. **See the voices:**
   ```bash
   uv run voicify --list-voices
   ```
   There are 28 English voices, American (`a…`) and British (`b…`), female (`…f_`) and male
   (`…m_`). The QUALITY column is Kokoro's own grade; A is best. The default is `af_heart` (marked `*`).

2. **Audition a few** with the first lines of your script:
   ```bash
   uv run voicify --text "Paste the first two sentences of your script here." --voice af_heart --output outputs/try_af_heart.wav
   ```
   ```bash
   uv run voicify --text "Paste the first two sentences of your script here." --voice am_michael --output outputs/try_am_michael.wav
   ```
   ```bash
   uv run voicify --text "Paste the first two sentences of your script here." --voice bf_emma --output outputs/try_bf_emma.wav
   ```
   Good ones to start with are `af_heart`, `af_bella`, `af_nicole`, `bf_emma`, `am_michael`, and `am_fenrir`.

3. **Pick a speed** (optional). `--speed 1.0` is about 177 words per minute. For a calmer
   narration, try `--speed 0.9`, or `--speed 0.85` for about 150 words per minute. Anything from
   0.5 to 2.0 works.

## Step 5: Generate the narration

```bash
uv run voicify --file scripts/my_script.txt --voice af_heart --output outputs/my_script.wav
```

While it runs, you see progress such as `chunk 12/58 (20%)`. When it finishes it prints:

```
Saved:     outputs\my_script.wav
Subtitles: outputs\my_script.srt
Duration:  16:13.8 (973.8 s)
Voice:     af_heart
Generated: 13.7 s (58 chunks, plus 11.9 s loading)
```

You now have:
- **`my_script.wav`**: the full narration as one audio file.
- **`my_script.srt`**: subtitles timed to the audio, one sentence per caption.

If a file with that name already exists, it is **not** overwritten. The new one is saved as
`my_script_1.wav` (the printed `Saved:` line always shows the real name). Add `--overwrite` to
replace the old file instead.

## Step 6: Check the result

1. **Listen** to the audio. Skip around, and pay attention to:
   - names or words that sound wrong (fix them in step 7),
   - pacing: too fast, too slow, or a missing pause,
   - the start of each paragraph.
2. **Check the subtitles**: open the audio in [VLC](https://www.videolan.org/), then choose
   *Subtitle → Add Subtitle File…* and pick the `.srt`. The captions should follow the voice.
3. **Read the exit message.** If it ends with a line like
   `warning: 1 of 58 chunks failed and were skipped: 12`, part of the script is missing from the
   audio. Run it again with `--verbose` to see why, then see step 7.

## Step 7: Fix what you heard

| Problem | Fix |
|---------|-----|
| A name or word is mispronounced | Add it to a pronunciation file (below) |
| Too fast or too slow | Change `--speed` (for example `0.9`) |
| You need a longer or shorter pause somewhere | Add `[pause 1s]`, `[pause 500ms]`, … in the script |
| Paragraph breaks are too long or short | Set `TTS_PARAGRAPH_PAUSE_MS` (default 700) in `.env` (step 9) |
| A number or abbreviation sounds odd | Write it out in words, or add it to the pronunciation file |
| Chunks were skipped | Run again with `--verbose`; see Troubleshooting in the README |

**Pronunciation file.** Copy the example and edit it:

```bash
cp scripts/pronunciations.example.txt scripts/my_words.txt
```

(On Windows PowerShell, `cp` works too; or use `Copy-Item`.) Write one word per line, followed by
how to say it:

```text
# word = how to say it
Kokoro = Koh-koh-roh
Nguyen = Win
SQL = sequel
nginx/i = engine x        # /i: match any capitalization
```

Then generate again with the file:

```bash
uv run voicify --file scripts/my_script.txt --pronunciations scripts/my_words.txt --output outputs/my_script.wav --overwrite
```

Only the voice changes; the subtitles still show the original spelling. Matching is whole-word, so
`SQL` does not change `SQLite`.

## Step 8: Export for your video

- **MP3** (smaller, for uploading):
  ```bash
  uv run voicify --file scripts/my_script.txt --output outputs/episode1.mp3
  ```
- **MP3 plus a WAV copy** (the WAV is better for editing in a video editor):
  ```bash
  uv run voicify --file scripts/my_script.txt --output outputs/episode1.mp3 --keep-wav
  ```
- **No subtitle file:** add `--no-srt`.

The `.srt` file can be imported into most video editors, or uploaded to YouTube as captions: in
YouTube Studio, open the video's *Subtitles* page, add a language, and upload the file
*with timing*.

## Step 9: Save your favorite settings (optional)

So you don't have to type the same options every time, create a `.env` file in the project folder:

```bash
cp .env.example .env
```

Open `.env` in a text editor and change what you need, for example:

```ini
TTS_DEFAULT_VOICE=bf_emma
TTS_DEFAULT_SPEED=0.9
TTS_PARAGRAPH_PAUSE_MS=900
TTS_PRONUNCIATIONS=scripts/my_words.txt
```

After that, the short command uses those settings automatically:

```bash
uv run voicify --file scripts/my_script.txt --output outputs/episode2.mp3
```

Options you type on the command line always win over `.env`.

## Step 10: Use it from your own Python code (optional)

To automate narration from another tool, such as a video pipeline or a batch of scripts:

```python
from tts_agent import TTSAgent

agent = TTSAgent()  # uses your .env settings
for name in ["episode1", "episode2"]:
    result = agent.generate(
        file=f"scripts/{name}.txt",
        voice="af_heart",
        output_path=f"outputs/{name}.mp3",
    )
    print(result.output_path, result.subtitle_path, f"{result.duration_seconds / 60:.1f} min")
```

Run it with `uv run python my_batch.py` from the project folder. The README describes everything
`generate()` accepts and returns.

---

## Cheat sheet

| I want to… | Command |
|------------|---------|
| Narrate a script | `uv run voicify --file scripts/my_script.txt` |
| Choose the output name | `... --output outputs/name.wav` (or `.mp3`) |
| Change voice / speed | `... --voice bf_emma --speed 0.9` |
| See all voices | `uv run voicify --list-voices` |
| Fix pronunciations | `... --pronunciations scripts/my_words.txt` |
| Replace an existing file | `... --overwrite` |
| MP3 and keep a WAV | `... --format mp3 --keep-wav` |
| Skip subtitles | `... --no-srt` |
| See details when something fails | `... --verbose` |
| All options | `uv run voicify --help` |

When something goes wrong, the error is printed as a single line starting with `error:`. The
[README's Troubleshooting section](../README.md#9-troubleshooting) covers the common ones, such as
the GPU not being detected, offline mode, and MP3 problems.
