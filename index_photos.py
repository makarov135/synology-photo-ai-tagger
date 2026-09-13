from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from photo_tagger.config import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_FRAME_RATIOS,
    DEFAULT_IMAGE_DIR,
    DEFAULT_IMAGE_MAX_TOKENS,
    DEFAULT_IMAGE_MIN_TOKENS,
    DEFAULT_LIVE_FRAME_MAX_EDGE,
    DEFAULT_PRIMARY_MAX_EDGE,
    ProjectPaths,
)
from photo_tagger.date_repair import repair_missing_capture_dates
from photo_tagger.metadata import restore_backup_run
from photo_tagger.pipeline import TaggingOptions, run_tagging


def _parse_ratios(raw: str) -> tuple[float, ...]:
    if not raw.strip():
        return ()
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("抽帧位置必须是逗号分隔的 0～1 小数") from exc
    if any(not 0 <= value <= 1 for value in values):
        raise argparse.ArgumentTypeError("抽帧位置必须是逗号分隔的 0～1 小数")
    return values


def _failed_asset_keys(report_path: Path) -> tuple[str, ...]:
    path = report_path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"重试报告不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取重试报告：{path}：{exc}") from exc
    items = payload.get("items", []) if isinstance(payload, dict) else []
    keys = tuple(
        dict.fromkeys(
            str(item["asset"])
            for item in items
            if isinstance(item, dict) and item.get("error") and item.get("asset")
        )
    )
    if not keys:
        raise SystemExit(f"报告中没有失败项目：{path}")
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用本地 AI 识别照片，并把中文关键词写入群晖 Photos 可搜索的元数据"
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="照片目录")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="清单、报告、模型和备份目录")
    parser.add_argument("--model-dir", type=Path, help="Qwen3-VL GGUF 模型目录")
    parser.add_argument("--model-file", type=Path, help="自定义 GGUF 主模型")
    parser.add_argument("--mmproj-file", type=Path, help="自定义 GGUF 视觉编码器")
    parser.add_argument("--llama-server", type=Path, help="llama-server.exe 路径")
    parser.add_argument("--write", action="store_true", help="确认写入照片；不加时只生成预览和清单")
    parser.add_argument(
        "--synology-refresh",
        action="store_true",
        help="写入后更新文件修改时间，触发群晖 Photos 重新读取元数据；必须与 --write 同用",
    )
    parser.add_argument(
        "--synology-sync",
        action="store_true",
        help="把缓存的关键词和描述同步到 Synology Photos 数据库；首次默认仅允许 1 张",
    )
    parser.add_argument("--synology-url", help="群晖 DSM 地址，例如 https://NAS_IP:5001")
    parser.add_argument("--synology-user", help="群晖用户名；不填写时运行中询问")
    parser.add_argument(
        "--synology-photo-root",
        help="图片目录在个人空间 Photos 下的路径，例如 /MobileBackup/iPhone/2026/09",
    )
    parser.add_argument(
        "--synology-insecure",
        action="store_true",
        help="允许本地群晖使用未受信任的 HTTPS 证书",
    )
    parser.add_argument(
        "--synology-sync-all",
        action="store_true",
        help="解除群晖数据库同步的单张测试限制；确认单张成功后才使用",
    )
    parser.add_argument("--force", action="store_true", help="忽略缓存，重新进行 AI 识别")
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="只使用已有识别缓存；缺少缓存时停止，绝不加载 AI 模型",
    )
    parser.add_argument("--scan-only", action="store_true", help="只扫描照片和实况配对，不加载模型")
    parser.add_argument("--limit", type=int, help="最多处理多少组照片，适合首次小批量测试")
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        metavar="GLOB",
        help="只处理匹配名称/相对路径的照片，可重复；例如 IMG_5006*",
    )
    parser.add_argument(
        "--retry-report",
        type=Path,
        help="只重试指定运行报告中的失败照片，避免再次遍历处理所有成功照片",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="只处理识别清单中从未出现过的新照片，与照片拍摄日期无关",
    )
    parser.add_argument(
        "--frame-ratios",
        type=_parse_ratios,
        default=DEFAULT_FRAME_RATIOS,
        help="实况 MOV 抽帧位置；传空字符串表示不抽帧",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=DEFAULT_PRIMARY_MAX_EDGE,
        help="主照片送入 AI 的最长边像素，默认 2048",
    )
    parser.add_argument(
        "--live-frame-max-edge",
        type=int,
        default=DEFAULT_LIVE_FRAME_MAX_EDGE,
        help="实况 MOV 抽帧最长边像素，默认 1024",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help="模型上下文长度，默认 12288",
    )
    parser.add_argument("--max-new-tokens", type=int, default=640, help="AI 最大输出 token 数")
    parser.add_argument(
        "--model-restart-interval",
        type=int,
        default=500,
        help="每识别多少张后重启本地 AI 释放 CUDA 缓存；默认 500，0 表示关闭",
    )
    parser.add_argument(
        "--image-min-tokens",
        type=int,
        default=DEFAULT_IMAGE_MIN_TOKENS,
        help="每张图最少视觉 token，默认 1024",
    )
    parser.add_argument(
        "--image-max-tokens",
        type=int,
        default=DEFAULT_IMAGE_MAX_TOKENS,
        help="每张图最多视觉 token，默认 3072",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="只生成一次关键词，跳过默认的第二次纠错核验",
    )
    parser.add_argument("--exiftool", type=Path, help="ExifTool 可执行文件路径")
    parser.add_argument("--ffmpeg", type=Path, help="ffmpeg 可执行文件路径")
    parser.add_argument("--ffprobe", type=Path, help="ffprobe 可执行文件路径")
    parser.add_argument("--no-auto-download", action="store_true", help="本地模型不存在时不要自动下载")
    parser.add_argument(
        "--restore-backup",
        type=Path,
        help="从指定的 metadata_backups 批次恢复原文件；此模式不会运行 AI",
    )
    parser.add_argument(
        "--repair-dates",
        action="store_true",
        help="修复因群晖写回导致时间线变化的无内嵌日期照片；不加 --write 时只预览",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()
    paths = ProjectPaths(args.image_dir.resolve(), args.data_dir.resolve())

    if args.restore_backup:
        restored = restore_backup_run(args.restore_backup, paths.image_dir)
        print(f"恢复完成：{restored} 个文件。再次运行时会根据文件变化重新检查。")
        return 0
    if args.repair_dates:
        stats = repair_missing_capture_dates(
            paths,
            write=args.write,
            patterns=tuple(args.match),
            limit=args.limit,
            exiftool_path=args.exiftool.resolve() if args.exiftool else None,
            progress=lambda current, total, message: print(
                (f"[{current:>3}/{total}] " if total else "") + message,
                flush=True,
            ),
        )
        action = "已修复" if args.write else "待修复"
        print(
            f"日期修复{'执行' if args.write else '预览'}完成：扫描 {stats.scanned_files} 个文件，"
            f"{action} {stats.repaired if args.write else stats.candidates} 个，失败 {stats.failed}。"
        )
        if not args.write and stats.candidate_files:
            print("待修复文件（仅显示前 20 个）：")
            for filename in stats.candidate_files[:20]:
                print(f"  - {filename}")
        if stats.failed_files:
            print("失败文件：")
            for filename, error in zip(
                stats.failed_files, stats.failure_details, strict=True
            ):
                print(f"  - {filename}：{error}")
        if stats.backup_run:
            print(f"日期修复前备份：{stats.backup_run}")
        if stats.report_path:
            print(f"日期修复报告：{stats.report_path}")
        if args.write and stats.repaired:
            print("请在群晖中重新建立照片索引，使 Photos 按补入的拍摄日期恢复时间线。")
        return 0 if stats.failed == 0 else 2
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit 不能小于 0")
    if args.synology_refresh and not args.write:
        raise SystemExit("--synology-refresh 必须与 --write 同时使用")
    if args.synology_sync and not args.write:
        raise SystemExit("--synology-sync 必须与 --write 同时使用")
    if args.synology_sync_all and not args.synology_sync:
        raise SystemExit("--synology-sync-all 必须与 --synology-sync 同时使用")
    if args.cached_only and args.force:
        raise SystemExit("--cached-only 不能与 --force 同时使用")
    if args.retry_report and args.match:
        raise SystemExit("--retry-report 不能与 --match 同时使用")
    if args.retry_report and args.incremental:
        raise SystemExit("--retry-report 不能与 --incremental 同时使用")
    if args.cached_only and args.incremental:
        raise SystemExit("--cached-only 不能与 --incremental 同时使用")
    if args.max_edge < 448:
        raise SystemExit("--max-edge 不能小于 448")
    if args.live_frame_max_edge < 448:
        raise SystemExit("--live-frame-max-edge 不能小于 448")
    if args.context_size < 4096:
        raise SystemExit("--context-size 不能小于 4096")
    if args.model_restart_interval < 0:
        raise SystemExit("--model-restart-interval 不能小于 0")
    if args.image_min_tokens < 256:
        raise SystemExit("--image-min-tokens 不能小于 256")
    if args.image_max_tokens < args.image_min_tokens:
        raise SystemExit("--image-max-tokens 不能小于 --image-min-tokens")

    def report(current: int, total: int, message: str) -> None:
        prefix = f"[{current:>3}/{total}] " if total else ""
        print(prefix + message, flush=True)

    synology_user = None
    synology_password = None
    synology_password_prompt = None
    synology_otp_prompt = None
    if args.synology_sync:
        synology_user = args.synology_user or input("群晖用户名：").strip()
        if not synology_user:
            raise SystemExit("群晖用户名不能为空")

        def synology_password_prompt() -> str:
            return getpass.getpass("群晖密码（不会保存）：")

        def synology_otp_prompt() -> str | None:
            return (
                getpass.getpass(
                    "请输入验证器当前显示的 6 位动态码（输入时不回显；程序不会发送验证码；未开启则回车）："
                ).strip()
                or None
            )

    retry_asset_keys = (
        _failed_asset_keys(args.retry_report) if args.retry_report else ()
    )

    stats = run_tagging(
        paths,
        TaggingOptions(
            model_dir=args.model_dir.resolve() if args.model_dir else None,
            model_file=args.model_file.resolve() if args.model_file else None,
            mmproj_file=args.mmproj_file.resolve() if args.mmproj_file else None,
            llama_server=args.llama_server.resolve() if args.llama_server else None,
            frame_ratios=args.frame_ratios,
            max_edge=args.max_edge,
            live_frame_max_edge=args.live_frame_max_edge,
            context_size=args.context_size,
            max_new_tokens=args.max_new_tokens,
            model_restart_interval=args.model_restart_interval,
            image_min_tokens=args.image_min_tokens,
            image_max_tokens=args.image_max_tokens,
            verify_tags=not args.single_pass,
            write=args.write,
            synology_refresh=args.synology_refresh,
            synology_sync=args.synology_sync,
            synology_url=args.synology_url,
            synology_user=synology_user,
            synology_password=synology_password,
            synology_otp=None,
            synology_photo_root=args.synology_photo_root,
            synology_insecure=args.synology_insecure,
            synology_sync_all=args.synology_sync_all,
            force=args.force,
            scan_only=args.scan_only,
            cached_only=args.cached_only,
            auto_download=not args.no_auto_download,
            limit=args.limit,
            patterns=tuple(args.match),
            retry_asset_keys=retry_asset_keys,
            incremental=args.incremental,
            exiftool=args.exiftool.resolve() if args.exiftool else None,
            ffmpeg=args.ffmpeg.resolve() if args.ffmpeg else None,
            ffprobe=args.ffprobe.resolve() if args.ffprobe else None,
        ),
        report,
        synology_otp_prompt=synology_otp_prompt,
        synology_password_prompt=synology_password_prompt,
    )
    mode = "写入" if args.write else "预览"
    print(
        f"完成（{mode}）：扫描媒体组 {stats.scanned}，选择照片 {stats.selected}，"
        f"AI 识别 {stats.analyzed}，使用缓存 {stats.cached}，"
        f"写入照片组 {stats.written_assets}/{stats.written_files} 个文件，"
        f"同步群晖数据库 {stats.synology_synced} 张，"
        f"成功 {stats.succeeded}，失败 {stats.failed}，"
        f"纯视频跳过 {stats.skipped_video_only}，警告 {stats.warnings}。"
    )
    if stats.failed_files:
        print("失败文件（可在运行报告中查看完整信息）：")
        for path, error in zip(stats.failed_files, stats.failure_details, strict=True):
            print(f"  - {path}：{error}")
    if not args.write and not args.scan_only:
        print("本次没有修改照片；确认预览后，加 --write 才会写入。")
    if stats.backup_run:
        print(f"本次原文件备份：{stats.backup_run}")
    return 0 if stats.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
