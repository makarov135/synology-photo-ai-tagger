from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from .ai import (
    MODEL_PROMPT_VERSION,
    LlamaCppPhotoTagger,
    LocalAIServerError,
    PhotoTags,
    ensure_local_model,
)
from .config import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_IMAGE_MAX_TOKENS,
    DEFAULT_IMAGE_MIN_TOKENS,
    DEFAULT_LIVE_FRAME_MAX_EDGE,
    DEFAULT_MODEL_ID,
    DEFAULT_PRIMARY_MAX_EDGE,
    ProjectPaths,
)
from .manifest import ManifestStore
from .media import AssetCandidate, extract_video_frames, load_still_image, refreshed_asset, scan_assets
from .metadata import (
    ExifTool,
    FileTagState,
    create_backups,
    extension_supports_date_fallback,
    restore_files,
    sha256_file,
)
from .synology import (
    SynologyPhotosClient,
    infer_personal_photo_root,
    infer_synology_url,
    normalize_photo_folder,
)


Progress = Callable[[int, int, str], None]
MODEL_IDENTITY = DEFAULT_MODEL_ID + ":Q8_0+mmproj-Q8_0"


@dataclass(slots=True)
class TaggingOptions:
    model_dir: Path | None = None
    model_file: Path | None = None
    mmproj_file: Path | None = None
    llama_server: Path | None = None
    frame_ratios: tuple[float, ...] = (0.25, 0.50, 0.75)
    max_edge: int = DEFAULT_PRIMARY_MAX_EDGE
    live_frame_max_edge: int = DEFAULT_LIVE_FRAME_MAX_EDGE
    context_size: int = DEFAULT_CONTEXT_SIZE
    max_new_tokens: int = 640
    image_min_tokens: int = DEFAULT_IMAGE_MIN_TOKENS
    image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS
    verify_tags: bool = True
    model_restart_interval: int = 500
    write: bool = False
    synology_refresh: bool = False
    synology_sync: bool = False
    synology_url: str | None = None
    synology_user: str | None = None
    synology_password: str | None = None
    synology_otp: str | None = None
    synology_photo_root: str | None = None
    synology_insecure: bool = False
    synology_sync_all: bool = False
    force: bool = False
    scan_only: bool = False
    cached_only: bool = False
    auto_download: bool = True
    limit: int | None = None
    patterns: tuple[str, ...] = ()
    retry_asset_keys: tuple[str, ...] = ()
    incremental: bool = False
    exiftool: Path | None = None
    ffmpeg: Path | None = None
    ffprobe: Path | None = None


@dataclass(slots=True)
class TaggingStats:
    scanned: int = 0
    selected: int = 0
    live_photos: int = 0
    analyzed: int = 0
    cached: int = 0
    written_assets: int = 0
    written_files: int = 0
    synology_synced: int = 0
    skipped_video_only: int = 0
    skipped_heic_files: int = 0
    succeeded: int = 0
    failed: int = 0
    failed_files: list[str] = field(default_factory=list)
    failure_details: list[str] = field(default_factory=list)
    warnings: int = 0
    backup_run: str | None = None


def run_tagging(
    paths: ProjectPaths,
    options: TaggingOptions,
    progress: Progress,
    synology_otp_prompt: Callable[[], str | None] | None = None,
    synology_password_prompt: Callable[[], str] | None = None,
) -> TaggingStats:
    paths.ensure_output_dirs()
    manifest = ManifestStore(paths.manifest)
    progress(0, 0, "正在扫描相册目录；完成缓存核对后才会询问密码和动态验证码…")

    def scan_progress(stage: str, current: int, total: int) -> None:
        if total:
            percent = current * 100 // total
            filled = percent * 20 // 100
            bar = "█" * filled + "░" * (20 - filled)
            progress(current, total, f"相册扫描 [{bar}] {percent}%：{stage}")
        else:
            progress(0, 0, f"相册扫描中：{stage}，已发现 {current} 个媒体文件，请继续等待…")

    assets = scan_assets(
        paths.image_dir,
        scan_progress,
        (lambda key: manifest.get(key) is None) if options.incremental else None,
    )
    stats = TaggingStats(scanned=len(assets))
    video_only = [
        asset for asset in assets if not asset.still_paths and not asset.ignored_paths
    ]
    stats.skipped_video_only = len(video_only)
    stats.skipped_heic_files = sum(len(asset.ignored_paths) for asset in assets)
    retry_keys = {key.casefold() for key in options.retry_asset_keys}
    candidates = [
        asset
        for asset in assets
        if asset.still_paths and _matches(asset, paths.image_dir, options.patterns)
        and (not retry_keys or asset.key.casefold() in retry_keys)
        and (not options.incremental or manifest.get(asset.key) is None)
    ]
    if options.limit is not None:
        candidates = candidates[: max(0, options.limit)]
    stats.selected = len(candidates)
    stats.live_photos = sum(asset.is_live_photo for asset in candidates)

    if options.synology_sync and len(candidates) > 1 and not options.synology_sync_all:
        raise ValueError(
            "首次群晖数据库同步默认只允许 1 张；请加 --limit 1 或用 --match 精确选择。"
            "确认单张成功后，才加 --synology-sync-all。"
        )

    progress(
        0,
        len(candidates),
        f"扫描到 {len(assets)} 组媒体；选择 {len(candidates)} 张照片；"
        f"其中实况 {stats.live_photos}；HEIC/HEIF {sum(path.suffix.casefold() in {'.heic', '.heif'} for asset in candidates for path in asset.still_paths)} 个；"
        f"跳过纯视频 {len(video_only)}",
    )
    if options.scan_only or not candidates:
        return stats

    model_identity = _model_identity(options)
    inference_profile = _inference_profile(model_identity, options)
    progress(
        0,
        len(candidates),
        "正在核对识别缓存；部分照片可能需要从 NAS 读取并校验，完成后再询问动态验证码…",
    )
    cache: dict[str, PhotoTags | None] = {}
    for cache_index, asset in enumerate(candidates, start=1):
        cache[asset.key] = _cached_tags(
            manifest.get(asset.key) or {}, asset, options, inference_profile
        )
        if cache_index == 1 or cache_index % 100 == 0 or cache_index == len(candidates):
            percent = cache_index * 100 // len(candidates)
            filled = percent * 20 // 100
            bar = "█" * filled + "░" * (20 - filled)
            progress(
                cache_index,
                len(candidates),
                f"等待登录准备 [{bar}] {percent}%：正在核对识别缓存",
            )
    cached_count = sum(tags is not None for tags in cache.values())
    progress(
        0,
        len(candidates),
        f"缓存核对完成：可复用 {cached_count} 张，需要 AI 识别 {len(candidates) - cached_count} 张；即将询问动态验证码",
    )
    if options.cached_only and any(tags is None for tags in cache.values()):
        missing = [asset.display_name for asset in candidates if cache[asset.key] is None]
        raise ValueError(
            "以下照片没有可用的当前识别缓存，--cached-only 已阻止启动 AI："
            + "、".join(missing[:5])
        )
    exiftool = ExifTool(options.exiftool) if options.write else None
    synology: SynologyPhotosClient | None = None
    photo_root: str | None = None
    if options.synology_sync:
        if not options.synology_user:
            raise ValueError("群晖数据库同步缺少用户名。")
        password = options.synology_password
        if password is None and synology_password_prompt is not None:
            password = synology_password_prompt()
        if password is None:
            raise ValueError("群晖数据库同步缺少密码。")
        progress(
            len(candidates),
            len(candidates),
            "等待动态验证码 [████████████████████] 100%：准备完成，请立即输入当前 OTP",
        )
        synology_url = options.synology_url or infer_synology_url(paths.image_dir)
        photo_root = normalize_photo_folder(
            options.synology_photo_root or infer_personal_photo_root(paths.image_dir)
        )
        otp_code = (
            synology_otp_prompt()
            if synology_otp_prompt is not None
            else options.synology_otp
        )
        synology = SynologyPhotosClient(
            synology_url,
            options.synology_user,
            password,
            otp_code,
            verify_tls=not options.synology_insecure,
        )
        progress(0, len(candidates), f"正在登录 Synology Photos：{synology.endpoint}")
        if options.synology_insecure:
            progress(0, len(candidates), "提示：本次仅为局域网群晖关闭 HTTPS 证书校验")
        synology.login()
        progress(0, len(candidates), "正在读取 Synology Photos 的文件夹、照片和标签索引…")
        target_folders = {
            normalize_photo_folder(
                photo_root
                if asset.primary_path.relative_to(paths.image_dir).parent.as_posix() == "."
                else f"{photo_root}/{asset.primary_path.relative_to(paths.image_dir).parent.as_posix()}"
            )
            for asset in candidates
        }
        synology.load_catalog(target_folders)
    tagger: LlamaCppPhotoTagger | None = None
    if any(tags is None for tags in cache.values()):
        progress(0, len(candidates), "正在准备 Qwen3-VL 8B Q8_0（首次运行会下载约 9.5 GB）…")
        if options.model_file or options.mmproj_file:
            if not options.model_file or not options.mmproj_file:
                raise ValueError("--model-file 和 --mmproj-file 必须同时提供")
            model_path = options.model_file.resolve()
            mmproj_path = options.mmproj_file.resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"自定义主模型不存在：{model_path}")
            if not mmproj_path.is_file():
                raise FileNotFoundError(f"自定义视觉编码器不存在：{mmproj_path}")
        else:
            model_path, mmproj_path = ensure_local_model(
                options.model_dir or paths.model_dir,
                auto_download=options.auto_download,
            )
        progress(0, len(candidates), "正在把 Q8 高精度模型载入 RTX 4080 SUPER…")
        tagger = LlamaCppPhotoTagger(
            model_path,
            mmproj_path,
            server_executable=options.llama_server,
            log_path=paths.data_dir / "llama-server.log",
            context_size=options.context_size,
            max_new_tokens=options.max_new_tokens,
            image_min_tokens=options.image_min_tokens,
            image_max_tokens=options.image_max_tokens,
            verify_tags=options.verify_tags,
        )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_run = paths.backups / run_id
    report_items: list[dict[str, object]] = []
    analyses_since_model_restart = 0

    try:
        for index, asset in enumerate(candidates, start=1):
            record = manifest.get(asset.key) or {}
            current_path = (
                asset.primary_path.relative_to(paths.image_dir).as_posix()
                if asset.primary_path is not None
                else asset.key
            )
            progress(index, len(candidates), f"正在处理：{current_path}")
            try:
                tags = cache[asset.key]
                timings: dict[str, object] = {}
                if tags is not None:
                    stats.cached += 1
                    restored_state = _matching_record_file_state(record, asset)
                    if (
                        record.get("signature") != asset.signature
                        and restored_state in {"before", "after"}
                    ):
                        record = {**record, "signature": asset.signature}
                        if restored_state == "before":
                            record.update(
                                {
                                    "status": "analyzed",
                                    "synology_refreshed_at": None,
                                    "synology_synced_at": None,
                                }
                            )
                            progress(
                                index,
                                len(candidates),
                                f"{asset.display_name}：检测到文件恢复为写入前原版，复用 AI 缓存并重新写入",
                            )
                        manifest.set(asset.key, record)
                        manifest.save()
                    progress(index, len(candidates), f"{asset.display_name}：使用已缓存识别结果")
                else:
                    assert tagger is not None
                    if not tagger.is_alive:
                        progress(
                            index,
                            len(candidates),
                            f"{asset.display_name}：检测到本地 AI 已退出，正在自动重启",
                        )
                        tagger.restart()
                        analyses_since_model_restart = 0
                    elif (
                        options.model_restart_interval > 0
                        and analyses_since_model_restart >= options.model_restart_interval
                    ):
                        progress(
                            index,
                            len(candidates),
                            f"{asset.display_name}：已连续识别 {analyses_since_model_restart} 张，正在定期重启 AI 释放显存缓存",
                        )
                        tagger.restart()
                        analyses_since_model_restart = 0
                    progress(index, len(candidates), f"{asset.display_name}：AI 正在识别")
                    images = _load_asset_images(
                        asset, options, stats, progress, index, len(candidates)
                    )
                    try:
                        try:
                            tags = tagger.analyze(images)
                        except LocalAIServerError as exc:
                            progress(
                                index,
                                len(candidates),
                                f"{asset.display_name}：本地 AI 连接中断，正在重启并重试当前照片一次：{exc}",
                            )
                            tagger.restart()
                            analyses_since_model_restart = 0
                            tags = tagger.analyze(images)
                        timings = dict(tagger.last_timings)
                    finally:
                        for image in images:
                            image.close()
                    analyses_since_model_restart += 1
                    stats.analyzed += 1
                    record = {
                        **record,
                        "signature": asset.signature,
                        "model": model_identity,
                        "inference_profile": inference_profile,
                        "prompt_version": MODEL_PROMPT_VERSION,
                        "tags": tags.to_dict(),
                        "status": "analyzed",
                        "synology_synced_at": None,
                        "analyzed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    }
                    manifest.set(asset.key, record)
                    manifest.save()

                progress(
                    index,
                    len(candidates),
                    f"{asset.display_name}：{tags.description}｜{', '.join(tags.keywords)}",
                )

                write_results: list[dict[str, object]] = []
                synology_result: dict[str, object] | None = None
                needs_synology_refresh = (
                    options.synology_refresh and not record.get("synology_refreshed_at")
                )
                if options.write and (
                    record.get("status") != "written" or needs_synology_refresh
                ):
                    assert exiftool is not None
                    backups = create_backups(asset.still_paths, paths.image_dir, backup_run)
                    old_file_records = (
                        record.get("files") if isinstance(record.get("files"), dict) else {}
                    )
                    try:
                        for still_path in asset.still_paths:
                            relative = still_path.relative_to(paths.image_dir).as_posix()
                            old_file = (
                                old_file_records.get(relative, {})
                                if isinstance(old_file_records, dict)
                                else {}
                            )
                            previous = FileTagState.from_dict(
                                old_file.get("state") if isinstance(old_file, dict) else None
                            )
                            result = exiftool.write(
                                still_path,
                                tags,
                                backups[still_path],
                                previous,
                                synology_refresh=options.synology_refresh,
                            )
                            write_results.append(result.to_dict())
                    except Exception:
                        restore_files(backups)
                        raise

                    written_at = datetime.now().astimezone().isoformat(timespec="seconds")
                    record = {
                        **record,
                        "signature": refreshed_asset(asset, paths.image_dir).signature,
                        "status": "written",
                        "written_at": written_at,
                        "synology_refreshed_at": written_at if options.synology_refresh else None,
                        "backup_run": str(backup_run),
                        "files": {
                            Path(str(item["path"])).relative_to(paths.image_dir).as_posix(): item
                            for item in write_results
                        },
                    }
                    manifest.set(asset.key, record)
                    manifest.save()
                    stats.written_assets += 1
                    stats.written_files += len(write_results)
                    stats.backup_run = str(backup_run)
                    refresh_message = "并已触发群晖刷新" if options.synology_refresh else ""
                    progress(
                        index,
                        len(candidates),
                        f"{asset.display_name}：已写入、回读验证{refresh_message}",
                    )

                if options.synology_sync and not record.get("synology_synced_at"):
                    assert synology is not None and photo_root is not None
                    assert exiftool is not None
                    record, timeline_guard_added = _ensure_timeline_guard(
                        exiftool,
                        asset,
                        record,
                        paths,
                        backup_run,
                    )
                    if timeline_guard_added:
                        stats.backup_run = str(backup_run)
                        progress(
                            index,
                            len(candidates),
                            f"{asset.display_name}：原文件没有内嵌拍摄日期，已用写入前文件时间固化日期并备份",
                        )
                    manifest.set(asset.key, record)
                    manifest.save()
                    progress(index, len(candidates), f"{asset.display_name}：正在同步群晖数据库")
                    previous_ai_description = record.get("synology_ai_description")
                    try:
                        result = synology.sync_asset(
                            asset,
                            tags,
                            image_dir=paths.image_dir,
                            photo_root=photo_root,
                            previous_ai_description=(
                                str(previous_ai_description)
                                if previous_ai_description is not None
                                else None
                            ),
                        )
                    finally:
                        record = _restore_timeline_and_refresh_hashes(
                            asset, record, paths.image_dir
                        )
                        manifest.set(asset.key, record)
                        manifest.save()
                    synology_result = result.to_dict()
                    if result.skipped_tags:
                        stats.warnings += 1
                        progress(
                            index,
                            len(candidates),
                            f"{asset.display_name}：群晖不支持或拒绝部分标签，已跳过："
                            + "、".join(result.skipped_tags),
                        )
                    synced_at = datetime.now().astimezone().isoformat(timespec="seconds")
                    record = {
                        **record,
                        "synology_synced_at": synced_at,
                        "synology_ai_description": tags.description,
                        "synology_item_id": result.item_id,
                        "synology_result": synology_result,
                    }
                    manifest.set(asset.key, record)
                    manifest.save()
                    stats.synology_synced += 1
                    progress(
                        index,
                        len(candidates),
                        f"{asset.display_name}：群晖数据库同步并回读验证成功",
                    )
                elif options.synology_sync:
                    previous_result = record.get("synology_result")
                    synology_result = (
                        previous_result if isinstance(previous_result, dict) else None
                    )
                    progress(index, len(candidates), f"{asset.display_name}：群晖数据库已同步，跳过")

                report_items.append(
                    {
                        "asset": asset.key,
                        "primary": str(asset.primary_path),
                        "live_photo": asset.is_live_photo,
                        "tags": tags.to_dict(),
                        "timings": timings,
                        "write_results": write_results,
                        "synology_result": synology_result,
                    }
                )
                stats.succeeded += 1
            except Exception as exc:
                stats.failed += 1
                failed_path = (
                    asset.primary_path.relative_to(paths.image_dir).as_posix()
                    if asset.primary_path is not None
                    else asset.key
                )
                stats.failed_files.append(failed_path)
                stats.failure_details.append(str(exc))
                progress(index, len(candidates), f"{failed_path}：失败：{exc}")
                report_items.append(
                    {
                        "asset": asset.key,
                        "primary": str(asset.primary_path) if asset.primary_path else None,
                        "error": str(exc),
                    }
                )
    finally:
        if tagger is not None:
            tagger.close()
        if synology is not None:
            try:
                synology.logout()
            except Exception as exc:
                stats.warnings += 1
                progress(0, len(candidates), f"群晖会话注销失败（不影响已完成同步）：{exc}")

    manifest.save()
    report_path = paths.reports / f"{run_id}.json"
    report_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "options": _serializable_options(options),
                "stats": asdict(stats),
                "items": report_items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return stats


def _cached_tags(
    record: dict[str, object],
    asset: AssetCandidate,
    options: TaggingOptions,
    inference_profile: str,
) -> PhotoTags | None:
    if options.force:
        return None
    if (
        record.get("signature") != asset.signature
        and _matching_record_file_state(record, asset) not in {"before", "after"}
    ):
        return None
    if (
        record.get("inference_profile") != inference_profile
        or record.get("prompt_version") != MODEL_PROMPT_VERSION
    ):
        return None
    tags = record.get("tags")
    return PhotoTags.from_dict(tags) if isinstance(tags, dict) else None


def _matching_record_file_state(
    record: dict[str, object], asset: AssetCandidate
) -> str | None:
    files = record.get("files")
    if not isinstance(files, dict) or not asset.still_paths:
        return None
    entries: dict[str, dict[str, object]] = {}
    for relative, raw in files.items():
        if isinstance(raw, dict):
            entries[Path(str(relative)).name.casefold()] = raw
    current_hashes: list[tuple[dict[str, object], str]] = []
    for path in asset.still_paths:
        entry = entries.get(path.name.casefold())
        if entry is None or not path.is_file():
            return None
        current_hashes.append((entry, _sha256_file(path)))
    for state, field in (("after", "after_sha256"), ("before", "before_sha256")):
        if all(entry.get(field) == digest for entry, digest in current_hashes):
            return state
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_timeline_guard(
    exiftool: ExifTool,
    asset: AssetCandidate,
    record: dict[str, object],
    paths: ProjectPaths,
    backup_run: Path,
) -> tuple[dict[str, object], bool]:
    raw_files = record.get("files")
    if not isinstance(raw_files, dict):
        return record, False
    files = {str(key): value for key, value in raw_files.items()}
    changed = False
    for still_path in asset.still_paths:
        if not extension_supports_date_fallback(still_path.suffix):
            continue
        relative = still_path.relative_to(paths.image_dir).as_posix()
        raw_entry = files.get(relative)
        if not isinstance(raw_entry, dict) or raw_entry.get("capture_time_source"):
            continue
        entry = dict(raw_entry)
        original_mtime_ns = int(entry.get("file_mtime_before_ns") or 0)
        if original_mtime_ns <= 0:
            raise ValueError(f"{still_path.name} 缺少写入前文件时间，无法安全保护相册顺序。")
        if exiftool.has_embedded_capture_time(still_path):
            entry["capture_time_source"] = "embedded"
            files[relative] = entry
            continue
        backups = create_backups([still_path], paths.image_dir, backup_run)
        capture_time = exiftool.ensure_capture_time(
            still_path,
            original_mtime_ns,
            backups[still_path.resolve()],
        )
        entry.update(
            {
                "capture_time_source": "original_file_mtime",
                "capture_time_added": capture_time,
                "capture_time_backup_path": str(backups[still_path.resolve()]),
                "after_sha256": sha256_file(still_path),
                "file_mtime_after_ns": still_path.stat().st_mtime_ns,
            }
        )
        files[relative] = entry
        changed = True
    updated = {**record, "files": files}
    if changed:
        updated["signature"] = refreshed_asset(asset, paths.image_dir).signature
        updated["timeline_guarded_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
    return updated, changed


def _restore_timeline_and_refresh_hashes(
    asset: AssetCandidate,
    record: dict[str, object],
    image_dir: Path,
) -> dict[str, object]:
    raw_files = record.get("files")
    if not isinstance(raw_files, dict):
        return record
    files = {str(key): value for key, value in raw_files.items()}
    for still_path in asset.still_paths:
        relative = still_path.relative_to(image_dir).as_posix()
        raw_entry = files.get(relative)
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        original_mtime_ns = int(entry.get("file_mtime_before_ns") or 0)
        if original_mtime_ns > 0 and still_path.is_file():
            current_stat = still_path.stat()
            os.utime(
                still_path,
                ns=(current_stat.st_atime_ns, original_mtime_ns),
            )
            entry["after_sha256"] = sha256_file(still_path)
            entry["file_mtime_after_ns"] = still_path.stat().st_mtime_ns
            files[relative] = entry
    return {
        **record,
        "files": files,
        "signature": refreshed_asset(asset, image_dir).signature,
    }


def _model_identity(options: TaggingOptions) -> str:
    if not options.model_file and not options.mmproj_file:
        return MODEL_IDENTITY
    digest = hashlib.sha256()
    for raw_path in (options.model_file, options.mmproj_file):
        if raw_path is None:
            digest.update(b"<missing>\n")
            continue
        path = raw_path.resolve()
        try:
            stat = path.stat()
            identity = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        except OSError:
            identity = f"{path}\0<unavailable>\n"
        digest.update(identity.encode("utf-8"))
    return f"custom-gguf:{digest.hexdigest()}"


def _inference_profile(model_identity: str, options: TaggingOptions) -> str:
    return ":".join(
        (
            model_identity,
            f"primary-{options.max_edge}",
            f"live-{options.live_frame_max_edge}",
            f"ctx-{options.context_size}",
            f"vision-{options.image_min_tokens}-{options.image_max_tokens}",
            "verified" if options.verify_tags else "single-pass",
        )
    )


def _load_asset_images(
    asset: AssetCandidate,
    options: TaggingOptions,
    stats: TaggingStats,
    progress: Progress,
    current: int,
    total: int,
) -> list[Image.Image]:
    assert asset.primary_path is not None
    images = [load_still_image(asset.primary_path, max_edge=options.max_edge)]
    if asset.is_live_photo and options.frame_ratios:
        motion = next(path for path in asset.motion_paths if path.suffix.casefold() == ".mov")
        try:
            frames = extract_video_frames(
                motion,
                options.frame_ratios,
                max_edge=options.live_frame_max_edge,
                ffmpeg_path=options.ffmpeg,
                ffprobe_path=options.ffprobe,
            )
            images.extend(image for _, image in frames)
        except Exception as exc:
            stats.warnings += 1
            progress(current, total, f"{asset.display_name}：实况抽帧失败，改用静态主图：{exc}")
    return images


def _matches(asset: AssetCandidate, image_root: Path, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    candidates = [asset.display_name, asset.key]
    candidates.extend(path.relative_to(image_root).as_posix() for path in asset.still_paths)
    lowered = [value.casefold() for value in candidates]
    return any(
        fnmatch.fnmatchcase(value, pattern.casefold())
        for pattern in patterns
        for value in lowered
    )


def _serializable_options(options: TaggingOptions) -> dict[str, object]:
    values = asdict(options)
    values["synology_password"] = "<redacted>" if options.synology_password else None
    values["synology_otp"] = "<redacted>" if options.synology_otp else None
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return values
