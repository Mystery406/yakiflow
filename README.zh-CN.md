# YakiFlow

简体中文 | [English](README.md)

YakiFlow 可以把本地视频、音频、网页视频或直播流转换成 SRT 字幕。它使用
`whisper.cpp` 转录，通过 Codex 或 Claude Code 翻译，并在保存前让你与交互式
Agent 一起审校结果。

任务支持断点续跑：如果运行中断，YakiFlow 会保留已经完成的工作，并打印继续
任务所需的命令。

## 快速开始

### 1. 安装必需工具

你需要：

- Python 3.12 或更高版本；
- `ffmpeg`；
- 较新版本的 [whisper.cpp](https://github.com/ggml-org/whisper.cpp)，且能在
  `PATH` 中找到 `whisper-cli`；
- `codex` 或 `claude` CLI，二选一安装并登录。

处理 URL 时还要安装 `yt-dlp`，处理直播流时还要有 `whisper-server`。`tmux` 和
`mpv` 是可选工具，分别用于分屏审校和视频预览。

### 2. 安装 YakiFlow

在本仓库中运行：

```console
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 3. 保存常用设置

在准备运行 YakiFlow 的目录中创建 `yakiflow.toml`：

```toml
source_language = "auto"
target_language = "zh-CN"
translation_backend = "codex"
output_dir = "./subtitles"
output_mode = "bilingual"
```

如果使用 Claude Code，把后端改为 `translation_backend = "claude"`。也可以不创建
配置文件，每次运行时都传入对应参数。

### 4. 下载默认模型并检查环境

```console
yakiflow models fetch
yakiflow doctor --translation-backend codex
```

使用 Claude Code 时把 `codex` 换成 `claude`。请修复所有与当前输入有关的
`FAIL`；`yt-dlp` 只影响 URL，`whisper-server` 只影响 `--stream`。

### 5. 生成字幕

```console
yakiflow run video.mp4
```

如果没有 `yakiflow.toml`，需要明确传入三个必填设置：

```console
yakiflow run video.mp4 \
  --source-language auto \
  --target-language zh-CN \
  --translation-backend codex \
  --output-dir ./subtitles
```

YakiFlow 会显示转录和翻译进度。在交互式终端中，批处理结束后还会打开所选
Agent 做最终审校。直接告诉 Agent 需要怎样修改，满意后结束审校会话。任务完成
时，终端会打印最终字幕路径。

## 常用操作

### 处理本地文件

只要 `ffmpeg` 支持，就可以使用相应格式的音频或视频：

```console
yakiflow run ./recordings/interview.mkv \
  --source-language en \
  --target-language zh-CN \
  --translation-backend codex \
  --output-dir ./subtitles
```

### 处理网页视频

```console
yakiflow run 'https://example.com/watch?v=123' \
  --source-language ja \
  --target-language en \
  --translation-backend claude \
  --download-dir ./downloads \
  --output-dir ./subtitles
```

`--download-dir` 会保留下载的媒体；不设置时使用临时副本。URL 任务建议设置
`--output-dir`，以便确定字幕保存位置。YakiFlow 不会下载整个播放列表。

### 处理直播流

```console
yakiflow run 'https://example.com/live' \
  --stream \
  --source-language auto \
  --target-language zh-CN \
  --translation-backend codex \
  --output-dir ./subtitles
```

直播字幕是临时结果。采集结束后，YakiFlow 会重新转录完整录制内容，再发布最终
SRT。按一次 Ctrl-C 可以停止采集，并完成已经收到的内容。

### 选择字幕格式

使用 `--output-mode`，或在 TOML 中设置 `output_mode`：

| 值 | 输入为 `video.mp4` 时的结果 |
| --- | --- |
| `bilingual` | `video.en-zh-cn.srt`，每条先放译文、再放原文（默认） |
| `source` | `video.source.srt` |
| `translated` | `video.translated.srt` |
| `all` | 同时生成以上三种文件 |

自动识别源语言时，双语文件名会使用 `auto`，例如
`video.auto-zh-cn.srt`。

### 恢复中断的任务

第一次按 Ctrl-C 会正常停止采集，第二次会强制退出。任务中断或失败后，YakiFlow
会打印保留下来的工作目录。运行以下命令恢复：

```console
yakiflow resume /path/to/workdir
```

请完整保留工作目录。`resume` 使用原来的输入和设置，不接受替换运行参数。如果
想预先确定工作目录，可在创建任务时加上 `--work-dir ./work/video-job`。

## 配置指南

YakiFlow 按以下顺序读取配置，越靠前优先级越高：

1. 命令行参数；
2. `--config` 指定的文件（或当前目录中的 `yakiflow.toml`）里选中的配置档；
3. 用户配置文件里选中的配置档；
4. 项目配置文件的基础设置；
5. 用户配置文件的基础设置；
6. 内置默认值。

Linux 用户配置文件位于 `~/.config/yakiflow/config.toml`（设置了
`XDG_CONFIG_HOME` 时则在 `$XDG_CONFIG_HOME/yakiflow/config.toml`）。普通设置
直接写在 TOML 文档根部。相对路径以启动任务时所在的目录为基准。

### 配置档

可复用的变体写在 `[profiles.NAME]` 下，并在 `run`、`doctor` 或
`models fetch` 中用 `--profile` 选择：

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

如果用户与项目文件都定义了所选名称，两处配置档会按字段合并，项目值覆盖用户
值；列表会整体替换，而不是追加。配置档在两个基础配置文件之后应用，因此用户
配置档也会覆盖项目基础设置；显式命令行参数仍有最高优先级。未选中的配置档完全
不生效。

选择不存在的配置档会报错。`profiles` 和每个命名配置档都必须是 TOML 表，配置档
不能包含嵌套表，也不能继承其他配置档。YakiFlow 会把完全解析后的有效设置随任务
保存，因此恢复任务不依赖原配置档或配置文件继续存在。YakiFlow 不提供内置配置档；
不选择配置档时仍使用原有默认行为。

### 常用设置

大多数用户只需要以下设置：

| TOML 键 | 命令行参数 | 用途 |
| --- | --- | --- |
| `source_language` | `--source-language` | 源语言代码，或 `auto` |
| `target_language` | `--target-language` | 翻译目标语言或区域 |
| `translation_backend` | `--translation-backend` | `codex` 或 `claude` |
| `output_dir` | `--output-dir` | 最终 SRT 保存目录 |
| `output_mode` | `--output-mode` | `source`、`translated`、`bilingual` 或 `all` |
| `download_dir` | `--download-dir` | 保留从 URL 下载的媒体 |
| `whisper_model` | `--whisper-model` | 使用已有的自定义 whisper.cpp 模型 |
| `vad_model` | `--vad-model` | 启用本地 Whisper 兼容 VAD 模型 |
| `memory` | `--memory` | 指定持久化术语和风格记忆文件 |
| `context_files` | `--context-file` | 将参考文件复制到工作目录供交互式审校使用；多个文件可重复指定此参数 |
| `agent_workers` | `--agent-workers` | 限制草稿翻译并发数，默认为 4 |
| `draft_model` | `--draft-model` | 覆盖后端的草稿模型 |
| `final_model` | `--final-model` | 覆盖交互式审校模型 |
| `keep_workdir` | `--keep-workdir` | 成功后仍保留中间文件 |

Codex 默认使用 `gpt-5.6-terra` 做草稿翻译、`gpt-5.6-sol` 做审校；Claude 默认
使用 `sonnet` 和 `opus`。如果当前账号不能使用这些名称，请覆盖相应模型设置。

例如，可以把弹幕导出文件和说话人备注作为只读参考交给交互式审校 Agent：

```console
yakiflow run video.mp4 --context-file danmaku.xml --context-file speakers.txt
```

YakiFlow 会将每个文件快照到任务工作目录的 `context/` 下，因此恢复任务时不再依赖
原始参考文件。

### 高级设置

以下表格列出了其余所有 TOML 设置。内置默认值通常已经合适，只有在需要对应行为时
才应修改。命令行参数栏中的破折号表示该设置仅支持 TOML。完整的命令行用法可运行
`yakiflow run --help` 查看。

#### 任务控制

| TOML 键 | 命令行参数 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `stream` | `--stream` / `--no-stream` | `false` | 将 URL 作为直播流处理 |
| `work_dir` | `--work-dir` | 临时目录 | 将可恢复任务状态和中间文件放在指定目录 |

#### 对齐

| TOML 键 | 命令行参数 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `alignment_backend` | `--alignment-backend` | `vad` | 时间轴调整后端：`vad` 或 `whisperx` |
| `alignment_device` | `--alignment-device` | `auto` | WhisperX 设备：`auto`、`cpu` 或 `cuda` |
| `alignment_model` | `--alignment-model` | 自动选择 | 覆盖对应语言的 WhisperX 对齐模型 |

#### 翻译与 Agent 调优

| TOML 键 | 命令行参数 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `draft_effort` | `--draft-effort` | `low` | 草稿翻译的推理强度 |
| `final_effort` | `--final-effort` | `high` | 交互式审校的推理强度 |
| `draft_codex_options` | — | `[]` | 仅传给 Codex 结构化草稿翻译调用的额外 argv token |
| `final_codex_options` | — | `[]` | 传给 Codex 交互式审校和记忆冲突会话的额外 argv token |
| `draft_claude_options` | — | `[]` | 仅传给 Claude 结构化草稿翻译调用的额外 argv token |
| `final_claude_options` | — | `[]` | 传给 Claude 交互式审校和记忆冲突会话的额外 argv token |
| `translation_batch_size` | — | `20` | 每次草稿请求最多发送的字幕条数 |
| `translation_context` | — | `10` | 作为翻译上下文提供的前文字幕条数 |
| `draft_agent_timeout_seconds` | `--draft-agent-timeout-seconds` | `600` | 每次草稿 Agent 尝试的超时时间 |
| `agent_max_attempts` | `--agent-max-attempts` | `3` | 草稿请求失败后的最大尝试次数 |
| `agent_retry_delay_seconds` | `--agent-retry-delay-seconds` | `1` | 草稿请求两次尝试之间的等待秒数 |

推理强度可设为 `minimal`、`low`、`medium`、`high` 或 `xhigh`。Agent 选项设置
必须是只含字符串的 TOML 数组。YakiFlow 会把每个字符串直接作为一个 argv token，
放在自身管理的参数之后、提示词或标准输入哨兵之前；不会进行 shell 解析、拼接或
过滤。无效或冲突的参数由 Codex 或 Claude 报错。

#### 审校显示

| TOML 键 | 命令行参数 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `review_display_mode` | `--review-display-mode` | `split` | 使用 tmux 字幕窗格（`split`）或外部程序（`open`） |
| `review_open_command` | `--review-open-command` | 未设置 | `open` 模式使用的命令模板；`{file}` 会替换为暂存 SRT 路径 |
| `auto_open_video` | `--auto-open-video` / `--no-auto-open-video` | `false` | 交互式审校开始时自动打开源视频 |
| `video_open_command` | `--video-open-command` | 内置 `mpv` 命令 | 打开媒体的命令模板；支持 `{file}` 和 `{subtitle}` |

当 `review_display_mode = "open"` 时，必须设置 `review_open_command`。

#### 直播流调优

| TOML 键 | 命令行参数 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `stream_chunk_seconds` | — | `15` | 每个直播转录分块的时长 |
| `stream_context_seconds` | — | `5` | 每个直播分块前保留的上下文音频重叠时长 |

#### 外部命令

| TOML 键 | 命令行参数 | 默认值 | 用途 |
| --- | --- | --- | --- |
| `ffmpeg` | — | `ffmpeg` | FFmpeg 可执行文件名或路径 |
| `yt_dlp` | — | `yt-dlp` | yt-dlp 可执行文件名或路径 |
| `whisper_cli` | — | `whisper-cli` | whisper.cpp 批处理可执行文件名或路径 |
| `whisper_server` | — | `whisper-server` | whisper.cpp 服务端可执行文件名或路径 |

如果外部命令不在 `PATH` 中，请将对应设置改为它的完整路径。

## 审校与翻译记忆

从终端运行时，YakiFlow 会在批量翻译后启动交互式审校。Agent 会检查完整字幕，
应用双方确认的修改，并可建议复用人名、术语或风格偏好。只有在你同意后，它才会
更新记忆文件。

默认情况下，YakiFlow 会尽量通过 `tmux` 在 Agent 旁显示只读字幕窗格。在该窗格
按 `O`，可以用 `mpv` 打开媒体并加载当前字幕。要使用其他播放器：

```toml
video_open_command = "vlc --sub-file={subtitle} {file}"
```

如需在交互式审校开始时自动打开源视频，请启用 `auto_open_video`（或传入
`--auto-open-video`）。程序会使用配置的 `video_open_command`，并在播放器支持时
附加暂存字幕。

如果希望用编辑器打开暂存 SRT，而不是使用分屏预览：

```toml
review_display_mode = "open"
review_open_command = "code --reuse-window {file}"
```

在 CI 或标准输入输出被重定向时，YakiFlow 会跳过交互式审校，直接保存批处理
结果。

## 可选：改善字幕时间轴

### VAD

Silero VAD 模型可以帮助 Whisper 跳过静音，并改善字幕起点。下载与你的
whisper.cpp 构建兼容的模型，然后配置路径：

```toml
vad_model = "/absolute/path/to/ggml-silero-v6.2.0.bin"
```

`yakiflow models fetch` 不会下载 VAD 模型。下载命令请参考
[whisper.cpp VAD 指南](https://github.com/ggml-org/whisper.cpp#voice-activity-detection-vad)。

### WhisperX 强制对齐

WhisperX 是可选组件，只调整时间轴；转录仍由 whisper.cpp 完成。当前 extra 需要
Python 3.12 或 3.13。使用 uv 安装以下版本之一，不能同时安装：

```console
# CPU
uv sync --extra whisperx-cpu

# NVIDIA CUDA 12.8
uv sync --extra whisperx-cuda
```

然后启用并检查：

```toml
alignment_backend = "whisperx"
alignment_device = "auto" # auto、cpu 或 cuda
```

```console
yakiflow doctor --alignment-backend whisperx
```

首次运行可能会下载 NLTK 数据和对应语言的对齐模型。如果交互式运行时初始化
失败，YakiFlow 会让你选择重试或回退 VAD；非交互运行会保留任务，供之后恢复。

## 故障排查

- 先运行 `yakiflow doctor --translation-backend codex`（或 `claude`）。
- 默认 Whisper 模型缺失时，运行 `yakiflow models fetch`。
- 如果配置的是自定义模型路径，YakiFlow 要求该文件已经存在，不会自动替换。
- `whisper-server` 失败只影响 `--stream`，`yt-dlp` 失败只影响 URL 输入。
- 审校时没有字幕窗格，可安装 `tmux`，或设置
  `review_display_mode = "open"`。
- 失败后请保留终端打印的工作目录；其中的任务日志和进程日志可用于排错或恢复。

## 开发

```console
uv sync --extra dev
uv run --extra dev pytest
```
