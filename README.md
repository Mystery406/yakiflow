# YakiFlow

[简体中文](README.zh-CN.md) | English

YakiFlow turns a local video, audio file, web video, or live stream into an SRT
subtitle file. It transcribes with `whisper.cpp`, translates with Codex or
Claude Code, and lets you review the result with an interactive Agent before it
is saved.

Jobs are resumable: if a run is interrupted, YakiFlow keeps its completed work
and prints the command needed to continue.

## Quick start

### 1. Install the required tools

You need:

- Python 3.12 or newer;
- `ffmpeg`;
- a recent [whisper.cpp](https://github.com/ggml-org/whisper.cpp) build whose
  `whisper-cli` executable is on `PATH`;
- either the `codex` or `claude` CLI, installed and signed in.

Install `yt-dlp` as well if you want to process URLs. Live streams additionally
need `whisper-server`. `tmux` and `mpv` are optional conveniences for subtitle
review and video preview.

### 2. Install YakiFlow

From this repository:

```console
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 3. Save your usual settings

Create `yakiflow.toml` in the directory where you will run YakiFlow:

```toml
source_language = "auto"
target_language = "zh-CN"
translation_backend = "codex"
output_dir = "./subtitles"
output_mode = "bilingual"
```

Use `translation_backend = "claude"` if you signed in with Claude Code. You
can also omit this file and pass the equivalent options on every run.

### 4. Download the default model and check the setup

```console
yakiflow models fetch
yakiflow doctor --translation-backend codex
```

Replace `codex` with `claude` when appropriate. Fix every reported `FAIL` that
applies to your input; `yt-dlp` is only needed for URLs and `whisper-server` is
only needed for `--stream`.

### 5. Create subtitles

```console
yakiflow run video.mp4
```

Without `yakiflow.toml`, provide the three required settings explicitly:

```console
yakiflow run video.mp4 \
  --source-language auto \
  --target-language zh-CN \
  --translation-backend codex \
  --output-dir ./subtitles
```

YakiFlow shows transcription and translation progress. In an interactive
terminal, it then opens the selected Agent for final review. Tell the Agent any
changes you want, then end the review session when you are satisfied. The final
subtitle path is printed when the job finishes.

## Common tasks

### Process a local file

Any audio or video format supported by `ffmpeg` can be used:

```console
yakiflow run ./recordings/interview.mkv \
  --source-language en \
  --target-language zh-CN \
  --translation-backend codex \
  --output-dir ./subtitles
```

### Process a web video

```console
yakiflow run 'https://example.com/watch?v=123' \
  --source-language ja \
  --target-language en \
  --translation-backend claude \
  --download-dir ./downloads \
  --output-dir ./subtitles
```

`--download-dir` keeps the downloaded media; omit it to use a temporary copy.
Set `--output-dir` for URL jobs so the subtitle location is predictable.
Playlists are not downloaded.

### Process a live stream

```console
yakiflow run 'https://example.com/live' \
  --stream \
  --source-language auto \
  --target-language zh-CN \
  --translation-backend codex \
  --output-dir ./subtitles
```

Live subtitles are provisional. After capture ends, YakiFlow transcribes the
complete recording again before publishing the final SRT. Press Ctrl-C once to
stop capture and finalize what has been received.

### Choose the subtitle format

Use `--output-mode` or set `output_mode` in TOML:

| Value | Result for an input named `video.mp4` |
| --- | --- |
| `bilingual` | `video.en-zh-cn.srt`, with translation above source text (default) |
| `source` | `video.source.srt` |
| `translated` | `video.translated.srt` |
| `all` | All three variants |

With automatic language detection, the bilingual filename uses `auto`, for
example `video.auto-zh-cn.srt`.

### Resume an interrupted job

The first Ctrl-C stops acquisition cleanly; a second Ctrl-C forces exit. After
an interruption or failure, YakiFlow prints the preserved work directory.
Resume it with:

```console
yakiflow resume /path/to/workdir
```

Keep the entire work directory intact. `resume` uses the original input and
settings and does not accept replacement run options. To choose its location in
advance, start the job with `--work-dir ./work/video-job`.

## Configuration guide

Settings are read in this order, from highest to lowest priority:

1. command-line options;
2. the selected profile in the file passed to `--config`, or in
   `yakiflow.toml` in the current directory;
3. the selected profile in the user configuration file;
4. base settings in the project configuration file;
5. base settings in the user configuration file;
6. built-in defaults.

On Linux, the user configuration file is
`~/.config/yakiflow/config.toml` (or
`$XDG_CONFIG_HOME/yakiflow/config.toml`). Put normal settings directly at the
TOML document root. Relative paths are resolved from the directory in which you
start the job.

### Configuration profiles

Put reusable variants under `[profiles.NAME]` and select one with `--profile`
on `run`, `doctor`, or `models fetch`:

```toml
source_language = "auto"
target_language = "zh-CN"
translation_backend = "codex"

[profiles.stream]
stream = true
translation_batch_size = 5
draft_codex_options = ["-c", "service_tier=fast"]
```

```console
yakiflow run 'https://example.com/live' --profile stream
yakiflow doctor --profile stream
yakiflow models fetch --profile stream
```

If both user and project files define the selected name, their profile settings
merge field by field, with project values replacing user values. Lists replace
earlier lists rather than being appended. A profile is applied after both base
files, so even a user profile overrides project base settings; explicit CLI
values still win. Unselected profiles have no effect.

Selecting an unknown profile is an error. `profiles` and each named profile
must be TOML tables, and profiles cannot contain nested tables or inherit from
other profiles. YakiFlow saves the fully resolved settings with each job, so a
resume does not depend on the profile or configuration files still existing.
There are no built-in profiles and selecting no profile preserves the normal
defaults.

### Common settings

Most users only need these settings:

| TOML key | CLI option | What to use it for |
| --- | --- | --- |
| `source_language` | `--source-language` | Source language code, or `auto` |
| `target_language` | `--target-language` | Translation language or locale |
| `translation_backend` | `--translation-backend` | `codex` or `claude` |
| `output_dir` | `--output-dir` | Directory for final SRT files |
| `output_mode` | `--output-mode` | `source`, `translated`, `bilingual`, or `all` |
| `download_dir` | `--download-dir` | Keep media downloaded from a URL |
| `whisper_model` | `--whisper-model` | Use an existing custom whisper.cpp model |
| `vad_model` | `--vad-model` | Enable a local Whisper-compatible VAD model |
| `memory` | `--memory` | Choose the persistent terminology/style memory file |
| `context_files` | `--context-file` | Copy reference files into the work directory for interactive review; repeat the option for multiple files |
| `agent_workers` | `--agent-workers` | Limit parallel draft translations (default: 4) |
| `draft_model` | `--draft-model` | Override the backend's draft model |
| `final_model` | `--final-model` | Override the interactive review model |
| `keep_workdir` | `--keep-workdir` | Keep intermediate files after success |

The default Codex models are `gpt-5.6-terra` for draft translation and
`gpt-5.6-sol` for review. The Claude defaults are `sonnet` and `opus`. Override
them if those names are unavailable to your account.

For example, pass a danmaku export and a speaker note to the interactive review
Agent as read-only references:

```console
yakiflow run video.mp4 --context-file danmaku.xml --context-file speakers.txt
```

YakiFlow snapshots each file under `context/` in the job's work directory, so a
resumed job does not depend on the original reference files.

### Advanced settings

The tables below complete the TOML reference. Defaults generally work well;
change these settings only when you need the described behavior. A dash in the
CLI column means the setting is TOML-only. Run `yakiflow run --help` for full
command-line usage.

#### Job control

| TOML key | CLI option | Default | What it controls |
| --- | --- | --- | --- |
| `stream` | `--stream` / `--no-stream` | `false` | Treat a URL as a live stream |
| `work_dir` | `--work-dir` | temporary directory | Place resumable job state and intermediate files in a chosen directory |

#### Alignment

| TOML key | CLI option | Default | What it controls |
| --- | --- | --- | --- |
| `alignment_backend` | `--alignment-backend` | `vad` | Timing adjustment backend: `vad` or `whisperx` |
| `alignment_device` | `--alignment-device` | `auto` | WhisperX device: `auto`, `cpu`, or `cuda` |
| `alignment_model` | `--alignment-model` | automatic | Override the language-specific WhisperX alignment model |

#### Translation and Agent tuning

| TOML key | CLI option | Default | What it controls |
| --- | --- | --- | --- |
| `draft_effort` | `--draft-effort` | `low` | Reasoning effort for draft translation |
| `final_effort` | `--final-effort` | `high` | Reasoning effort for interactive review |
| `draft_codex_options` | — | `[]` | Extra Codex argv tokens for structured draft translation calls |
| `final_codex_options` | — | `[]` | Extra Codex argv tokens for interactive review and memory-conflict sessions |
| `draft_claude_options` | — | `[]` | Extra Claude argv tokens for structured draft translation calls |
| `final_claude_options` | — | `[]` | Extra Claude argv tokens for interactive review and memory-conflict sessions |
| `translation_batch_size` | — | `20` | Maximum subtitle cues sent in each draft request |
| `translation_context` | — | `10` | Number of preceding cues supplied as translation context |
| `draft_agent_timeout_seconds` | `--draft-agent-timeout-seconds` | `600` | Timeout for each draft Agent attempt |
| `agent_max_attempts` | `--agent-max-attempts` | `3` | Maximum attempts for a failed draft request |
| `agent_retry_delay_seconds` | `--agent-retry-delay-seconds` | `1` | Delay between draft request attempts |

The supported effort values are `minimal`, `low`, `medium`, `high`, and
`xhigh`. Agent option settings must be TOML arrays containing only strings.
YakiFlow passes each string directly as one argv token, after its managed flags
and before the prompt or stdin sentinel; it does not shell-parse, combine, or
filter them. Invalid or conflicting options are reported by Codex or Claude.

#### Review display

| TOML key | CLI option | Default | What it controls |
| --- | --- | --- | --- |
| `review_display_mode` | `--review-display-mode` | `split` | Use a tmux subtitle pane (`split`) or an external program (`open`) |
| `review_open_command` | `--review-open-command` | unset | Command template used by `open` mode; `{file}` is replaced with the staged SRT path |
| `video_open_command` | `--video-open-command` | built-in `mpv` command | Command template for opening media; supports `{file}` and `{subtitle}` |

`review_open_command` is required when `review_display_mode = "open"`.

#### Live-stream tuning

| TOML key | CLI option | Default | What it controls |
| --- | --- | --- | --- |
| `stream_chunk_seconds` | — | `15` | Duration of each live transcription chunk |
| `stream_context_seconds` | — | `5` | Audio overlap retained before each live chunk for context |

#### External commands

| TOML key | CLI option | Default | What it controls |
| --- | --- | --- | --- |
| `ffmpeg` | — | `ffmpeg` | FFmpeg executable name or path |
| `yt_dlp` | — | `yt-dlp` | yt-dlp executable name or path |
| `whisper_cli` | — | `whisper-cli` | whisper.cpp batch executable name or path |
| `whisper_server` | — | `whisper-server` | whisper.cpp server executable name or path |

Set an external command to its full path if it is not on `PATH`.

## Review and translation memory

When run from a terminal, YakiFlow starts an interactive review after batch
translation. The Agent checks the complete subtitle file, applies agreed edits,
and can propose reusable names, terminology, or style preferences. Memory is
only updated after you approve a proposal.

By default YakiFlow uses `tmux` to show a read-only subtitle pane next to the
Agent when possible. In that pane, press `O` to open the media with the current
subtitles in `mpv`. To use another player:

```toml
video_open_command = "vlc --sub-file={subtitle} {file}"
```

To open the staged SRT in an editor instead of using the split preview:

```toml
review_display_mode = "open"
review_open_command = "code --reuse-window {file}"
```

In CI or with redirected input/output, interactive review is skipped and the
batch result is finalized directly.

## Optional: improve subtitle timing

### VAD

A Silero VAD model helps Whisper skip silence and improves cue starts. Download
a model supported by your whisper.cpp build, then configure its path:

```toml
vad_model = "/absolute/path/to/ggml-silero-v6.2.0.bin"
```

`yakiflow models fetch` does not download VAD models. The
[whisper.cpp VAD guide](https://github.com/ggml-org/whisper.cpp#voice-activity-detection-vad)
has the model download commands.

### WhisperX forced alignment

WhisperX is optional and only adjusts timing; transcription still uses
whisper.cpp. Python 3.12 or 3.13 is required for the current extra. Install
exactly one variant with uv:

```console
# CPU
uv sync --extra whisperx-cpu

# NVIDIA CUDA 12.8
uv sync --extra whisperx-cuda
```

Then enable and verify it:

```toml
alignment_backend = "whisperx"
alignment_device = "auto" # auto, cpu, or cuda
```

```console
yakiflow doctor --alignment-backend whisperx
```

The first run may download NLTK data and a language-specific alignment model.
If alignment setup fails interactively, YakiFlow lets you retry or fall back to
VAD; a non-interactive run preserves the job for later resumption.

## Troubleshooting

- Start with `yakiflow doctor --translation-backend codex` (or `claude`).
- If the default Whisper model is missing, run `yakiflow models fetch`.
- If a custom model path is configured, YakiFlow expects that file to already
  exist and will not replace it.
- `whisper-server` failures only affect `--stream`; `yt-dlp` failures only
  affect URL inputs.
- If review has no subtitle pane, install `tmux` or configure
  `review_display_mode = "open"`.
- Keep the printed work directory after a failure. It contains the job journal
  and process logs needed to diagnose or resume the run.

## Development

```console
uv sync --extra dev
uv run --extra dev pytest
```
