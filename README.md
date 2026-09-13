# 本地 AI 照片关键词写入工具（面向 Windows + NVIDIA CUDA）

文档版本：2026-09-13

完整的配置、使用、恢复和故障处理说明请阅读 `使用手册.md`。

这个项目使用本地显卡运行，推荐显存至少16G且面向 Windows + NVIDIA CUDA
本项目模型为`Qwen3-VL-8B-Instruct` 的官方 `Q8_0` GGUF 模型，为照片生成中文描述与关键词。

首次存量相册处理完成后，日常只需使用 `--incremental`。它按照片相对于 Photos
根目录的“文件夹 + 文件主名”判断是否为新媒体组，不依赖拍摄日期、文件修改时间或
时间线位置，所以后来补传的旧月份照片也能被发现。

## 写入规则

- 所有静态照片写入 `XMP-dc:Subject`，这是群晖 Photos 的主要“常规标签”字段。
- JPG/JPEG 同时写入 `IPTC:Keywords`，提高兼容性。
- 描述写入 `XMP-dc:Description`；支持的格式也写入 `EXIF:ImageDescription`，JPG/JPEG 额外写入群晖 Photos 可搜索的 `IPTC:Caption-Abstract`。
- 已有人工关键词和人工描述会保留，不会被 AI 覆盖。
- 对群晖中已经入库的照片，仅修改文件和重建索引并不能可靠更新 Photos 数据库；可用 `--synology-sync` 通过与网页端相同的 Photos 接口同步标签和描述。
- JPG/JPEG、HEIC/HEIF、PNG、DNG 都会处理；HEIC 保持原格式，只在内存中解码供 AI 识别。
- iPhone 实况照片的 MOV 会抽取 25%、50%、75% 三帧参与识别，但 MOV 文件本身绝不修改。
- 独立 MOV/MP4 跳过，因为群晖 Photos 不读取视频中的常规标签。

公开版默认照片目录是项目内的 `image` 测试目录。处理群晖时必须用 `--image-dir "\\NAS_IP\home\Photos"` 明确指定个人空间根目录。

## 第一次准备

在 PyCharm 中把项目解释器设为：

```text
.venv\Scripts\python.exe
```

安装 Python 依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

然后单独运行一次 `setup_tools.py`。它会从官方发布页下载 ExifTool 和 llama.cpp CUDA 版，
核对 SHA-256 后放进 `tools/`。模型不会在这一步下载。

## 推荐的首次测试

先仅检查整个个人空间的扫描和实况配对，不加载 AI：

```powershell
.\.venv\Scripts\python.exe index_photos.py --scan-only
```

首次 AI 预览建议只处理 3 张；第一次会通过 ModelScope 下载约 9.5 GB 的官方模型：

```powershell
.\.venv\Scripts\python.exe index_photos.py --limit 3
```

这一步只生成 `data/tag_manifest.json` 和运行报告，不修改照片。确认输出合理后，再写入同一批：

```powershell
.\.venv\Scripts\python.exe index_photos.py --limit 3 --write
```

写入前，每个原文件都会完整备份到 `data/metadata_backups/<批次时间>/`。写入后程序会：

1. 回读 XMP/IPTC/描述，确认关键词完整；
2. 对比拍摄时间、GPS、方向、相机信息和 Apple 实况标识；
3. 失败时自动恢复该照片组的全部静态文件。

确认小批量无误后，处理全部照片：

```powershell
.\.venv\Scripts\python.exe index_photos.py --image-dir "\\NAS_IP\home\Photos" --write --synology-sync --synology-sync-all --synology-url "https://NAS_IP:5001" --synology-user "你的用户名" --synology-insecure
```

结束时程序会单独显示成功数和失败数；每个失败文件会打印相对于 `Photos` 根目录的路径及错误原因，完整内容同时保存在 `data/reports/`。

群晖中已经入库的旧照片，请先只测试一张。下面的 `--cached-only` 保证只复用现有识别结果，不启动 Qwen；命令会在控制台安全询问密码和两步验证码：

```powershell
.\.venv\Scripts\python.exe index_photos.py --image-dir "\\NAS_IP\home\Photos\MobileBackup\iPhone\2026\09" --match "IMG_1234*" --write --synology-sync --synology-url "https://NAS_IP:5001" --synology-user "你的用户名" --synology-insecure --cached-only
```

把 `IMG_1234*` 换成一张已经成功识别过的照片名称。`--synology-insecure` 只用于局域网中按 IP 访问、自签名证书无法验证的情况。程序只增加标签，并保护 Photos 数据库中已有的人工描述；同步后立即回读验证。默认最多同步一张，确认网页端显示和搜索都正常后才可加 `--synology-sync-all` 批量运行。

此操作不改文件名、不移动照片。对于没有内嵌拍摄日期的旧 JPG/PNG，程序会把写入前的文件时间固化为拍摄日期，并在群晖同步后恢复原文件时间，避免 Photos 把它们显示为处理当天。密码和验证码只存在于本次进程内，不写入配置、清单或报告。

如果旧版本已经造成时间线变化，可先运行 `--repair-dates` 只读预览，再运行 `--repair-dates --write`：程序会先备份当前含 AI 标签的文件，然后补入原日期，不重新运行 AI。完成后在群晖中重新建立照片索引。

`--synology-refresh` 旧参数仍保留，但实测即使刷新文件修改时间并重建索引，Photos 也可能继续使用旧数据库内容；它不能代替 `--synology-sync`。

任务结束后如果有失败项，可以只重试该报告中的失败照片，避免再次处理全部相册：

```powershell
.\.venv\Scripts\python.exe index_photos.py --image-dir "\\NAS_IP\home\Photos" --write --synology-sync --synology-sync-all --synology-url "https://NAS_IP:5001" --synology-user "你的用户名" --synology-insecure --retry-report "data\reports\运行报告.json"
```

日常手机同步后使用 `--incremental`，程序仍会递归扫描整个 Photos，但只选择
`tag_manifest.json` 中从未出现过的新媒体组。判断依据是相对文件夹和文件主名，
与拍摄日期、文件修改时间及时间线位置无关，因此后来补传的旧月份照片也会被处理：

```powershell
.\.venv\Scripts\python.exe index_photos.py --image-dir "\\NAS_IP\home\Photos" --incremental --write --synology-sync --synology-sync-all --synology-url "https://NAS_IP:5001" --synology-user "你的用户名" --synology-insecure
```

不要给日常增量命令添加 `--force`。如果没有新增照片，扫描结束后会显示选择 0 张，
不会启动 AI，也不会调用 Photos 同步接口。

扫描期间会依次显示“遍历目录”“整理媒体文件”“读取媒体文件状态”和“缓存核对”进度。
增量模式不会为旧照片重新计算完整文件签名。所有慢速准备完成后程序才询问密码，输入密码后
会立即提示当前六位 OTP，避免动态码在扫描期间过期。密码和 OTP 输入时不回显，也不会保存。

注意：同一文件夹内用相同主文件名覆盖旧照片，不属于“新媒体组”，`--incremental` 会跳过；
确实需要重做时使用 `--match "文件名*" --force`。不要删除 `tag_manifest.json`，否则所有照片
都会被当作首次出现。


中断后重新运行会根据 `tag_manifest.json` 继续，已完成且未变化的照片不会重复识别。
为避免超长批次中 llama.cpp 的 CUDA/提示缓存持续累积，程序默认每完成 500 张 AI
识别就自动重启一次模型服务；连接意外中断时会自动重启并只重试当前照片一次。
可用 `--model-restart-interval` 调整间隔，通常不建议关闭。

默认高精度参数为：主照片最长边 2048 像素、实况抽帧最长边 1024 像素、
12288 token 上下文、每张图 1024～3072 个视觉 token，并对第一轮结果进行第二轮看图复核。
若更看重速度，可用 `--single-pass` 跳过复核；它会降低耗时，但也更容易保留误识别内容：

```powershell
.\.venv\Scripts\python.exe index_photos.py --limit 3 --single-pass
```

## 按名称测试与恢复

可重复使用 `--match` 精确挑选照片：

```powershell
.\.venv\Scripts\python.exe index_photos.py --match "IMG_5006*" --match "IMG_4813*"
```

恢复某次写入前的完整原文件：

```powershell
.\.venv\Scripts\python.exe index_photos.py --restore-backup "data\metadata_backups\批次目录名"
```

