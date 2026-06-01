# Bass Extractor Pro

这是一个面向商用流程的 bass 音轨提取工具。它使用 Demucs 做源分离，提供桌面 GUI、命令行、批量脚本入口和质量报告。

先说真实边界：从已经混在一起的成品歌曲里提取 bass，不可能做到数学意义上的“精准无误”。母带里 bass、底鼓、低频合成器、房间混响和压缩器副作用会互相重叠。这个软件的目标是商业可用：使用成熟模型、保留高质量 WAV/FLAC 输出、生成可审查的 QC 报告，并让最终交付前可以人工复核。

## 功能

- 提取 `bass` stem，默认使用 `htdemucs_ft` 高质量模型。
- 支持 `wav`、`flac`、`mp3` 输出；商用母带建议用 32-bit float `wav`。
- 支持 GUI 和 CLI。
- 支持 `studio`、`balanced`、`fast` 三档质量配置。
- 可选导出 `no_bass` 版本。
- 可选 kick-aware 清理：检测固定音高、重复节奏的 kick，在 bass stem 的对应瞬态窗口中压低 kick 泄漏。
- 可选生成 bass 五线谱 MusicXML：自动估计 BPM、调号和按小节推断的和弦标记。
- 自动生成 `*.quality.json`，记录峰值、RMS、裁剪风险、低频能量比例、时长匹配等指标。

## 安装

在 PowerShell 中运行：

```powershell
cd "C:\Users\ck\Documents\New project\bass-extractor"
.\install.ps1
```

首次真正分离歌曲时，Demucs 会下载模型权重。WAV/FLAC 由内置 Python 音频后端读取；处理 MP3、M4A、AAC、OGG 等压缩格式时，系统还需要安装 `ffmpeg` 并放入 PATH。没有 `ffmpeg` 时，请优先输入 WAV。

如果安装脚本发现独立虚拟环境里的 PyTorch DLL 无法加载，会自动尝试 `.venv-system`，复用系统中已安装且可运行的 PyTorch。

## 启动 GUI

```powershell
cd "C:\Users\ck\Documents\New project\bass-extractor"
.\run-gui.ps1
```

## 命令行使用

```powershell
cd "C:\Users\ck\Documents\New project\bass-extractor"
.\run-cli.ps1 "C:\music\song.wav" -OutputPath "C:\music\song_bass.wav" -Profile studio -Format wav
```

如果歌曲里 kick 是固定音高且节奏规律，可以打开 kick 清理：

```powershell
.\run-cli.ps1 "C:\music\song.wav" -OutputPath "C:\music\song_bass.wav" -Profile studio -Format wav -KickClean -KickStrength 0.7
```

同时生成 bass 五线谱：

```powershell
.\run-cli.ps1 "C:\music\song.wav" -OutputPath "C:\music\song_bass.wav" -Profile studio -Format wav -KickClean -Score -ScoreTempo 120 -ScoreKey "A minor"
```

生成的 `*.musicxml` 可以用 MuseScore、Logic、Dorico、Finale 等软件打开。BPM 和调号会写入谱面；和弦来自 bass 音符和调性推断，不等于完整和声听写。

也可以直接调用 Python：

```powershell
.\.venv\Scripts\python.exe -m bass_extractor.cli "C:\music\song.wav" -o "C:\music\song_bass.wav" --profile studio --device auto --format wav
```

Python CLI 对应参数：

```powershell
.\.venv\Scripts\python.exe -m bass_extractor.cli "C:\music\song.wav" -o "C:\music\song_bass.wav" --profile studio --kick-clean --kick-strength 0.7
```

Python CLI 生成谱面：

```powershell
.\.venv\Scripts\python.exe -m bass_extractor.cli "C:\music\song.wav" -o "C:\music\song_bass.wav" --profile studio --kick-clean --score --score-tempo 120 --score-key "A minor"
```

环境诊断：

```powershell
.\.venv\Scripts\python.exe -m bass_extractor.cli --doctor
```

## 质量档位

- `studio`：`htdemucs_ft`、更多 shift averaging，适合最终交付，速度较慢。
- `balanced`：速度和质量折中。
- `fast`：预览用，不建议作为最终商用成品。

## 商用交付建议

- 优先使用无损输入：WAV 或 FLAC。
- 输出使用 WAV，避免 MP3 二次损伤。
- 对每首歌保留 `*.quality.json`。
- 对低频复杂、bass 与 kick 重叠严重的歌曲，必须人工听检。
- kick-aware 清理适合固定音高 kick；如果 kick 有长 808 尾音、滑音或和 bass 同音高同节奏，强度不要过高。
- 只靠 bass 音频无法唯一确定完整和弦，谱面里的和弦是基于 bass 根音和调号的推断，正式出版前需要人工校对。
- 如果要批量处理正式曲库，建议使用 CUDA GPU，并统一记录模型名、profile、日期和操作员。

## 打包成 exe

```powershell
cd "C:\Users\ck\Documents\New project\bass-extractor"
.\build-exe.ps1
```

输出位置：

```text
C:\Users\ck\Documents\New project\bass-extractor\dist\BassExtractorPro.exe
```
