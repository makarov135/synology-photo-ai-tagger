from __future__ import annotations

import base64
import io
import json
import re
import secrets
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .config import (
    DEFAULT_LLAMA_SERVER,
    DEFAULT_MMPROJ_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_MODEL_ID,
    PROMPT_VERSION,
)


class LocalAIServerError(RuntimeError):
    """The local llama.cpp process stopped or became unreachable."""


SYSTEM_PROMPT = (
    "你是私人相册的严谨中文编目员。只描述画面中能够确认的内容，不猜测姓名、"
    "具体地址、人物关系、民族、宗教、健康或其他敏感属性。"
)

TAGGING_PROMPT = """请把所有输入图像视为同一项相册内容：第一张是主照片，其余图像（如果有）是同一张 iPhone 实况照片在不同时刻的画面。

请生成便于中文搜索的照片元数据：
1. description：一句自然、具体、客观的中文描述，建议 20～80 个汉字。
2. keywords：8～20 个简体中文词或短语，覆盖可确认的主要人物数量（不要猜姓名）、动物、物体、场景、活动、环境、天气、昼夜、显著颜色和拍摄类型。
3. 截图、票据、二维码、招牌或文档应加入清晰可辨且有搜索价值的文字、应用名或品牌名；不能确认的文字不要猜。
4. 不要加入“照片、图片、实况照片、好看、生活”等空泛词，不要使用 #，不要重复近义词。
5. ocr_text：最多 5 段清晰可辨且有搜索价值的原文；没有则为空数组。

只输出 JSON，不要 Markdown 和解释：
{"description":"……","keywords":["……"],"ocr_text":["……"]}
"""

VERIFICATION_PROMPT = """请重新仔细检查输入画面，审校上一轮 JSON，在准确率和可搜索性之间取得平衡：
1. 上一轮中有画面、清晰文字、标志或外观特征支持的具体词必须保留，包括品牌、型号、物种、颜色和数值；不要仅因它很具体就删除。
2. 只有在内容与画面明确矛盾，或画面完全没有可见依据时，才删除猜测得到的地点、品牌、型号、物种、人物属性、动作或文字。
3. 保留清晰可见且有检索价值的 OCR 原文；纠正明显的错字。局部可辨但可能不完整的文字可以保留，不要擅自补全。
4. 补充画面中明显但遗漏的主体、场景、活动和关键物体，同时保留便于宽泛搜索的上位词，例如“哈苏相机”应同时保留“哈苏”和“相机”。
5. description 必须客观、自然；keywords 保持 8～20 个，实在没有足够内容时可以少于 8 个；避免重复近义词和空泛词。
6. 只输出最终修正后的 JSON，不要 Markdown、解释或评价：
{"description":"……","keywords":["……"],"ocr_text":["……"]}
"""

RECOVERY_PROMPT = """上一轮输出过长、重复或 JSON 不完整。请重新查看图像并精简输出：
1. description 只写一句不超过 60 个汉字的客观描述。
2. keywords 只保留 6～12 个最重要、互不重复的搜索词。
3. ocr_text 最多 3 段，每段不超过 30 个字符；忽略重复、乱码和无意义字符。
4. 必须输出完整合法的单行 JSON，不要解释：
{"description":"……","keywords":["……"],"ocr_text":["……"]}
"""


@dataclass(frozen=True, slots=True)
class PhotoTags:
    description: str
    keywords: tuple[str, ...]
    ocr_text: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["keywords"] = list(self.keywords)
        result["ocr_text"] = list(self.ocr_text)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PhotoTags":
        return normalize_tags(
            value.get("description", ""),
            value.get("keywords", []),
            value.get("ocr_text", []),
        )


def normalize_tags(description: object, keywords: object, ocr_text: object) -> PhotoTags:
    clean_description = _clean_text(description)
    if not clean_description:
        raise ValueError("模型没有返回照片描述")
    clean_description = clean_description[:160]

    keyword_values = _as_text_list(keywords)
    ocr_values = _as_text_list(ocr_text)
    clean_ocr = tuple(_deduplicate(item for item in ocr_values if 1 < len(item) <= 80)[:5])
    combined = [*keyword_values, *(item for item in clean_ocr if len(item) <= 30)]
    clean_keywords = _deduplicate(
        item
        for item in combined
        if 0 < len(item) <= 40 and item not in {"照片", "图片", "实况照片", "生活"}
    )[:24]
    if len(clean_keywords) < 3:
        raise ValueError("模型返回的有效关键词少于 3 个")
    return PhotoTags(clean_description, tuple(clean_keywords), clean_ocr)


def parse_model_response(raw: str) -> PhotoTags:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"模型输出中没有 JSON：{raw[:200]}")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型输出不是合法 JSON：{raw[:300]}") from exc
    if not isinstance(value, dict):
        raise ValueError("模型输出的 JSON 顶层必须是对象")
    return normalize_tags(
        value.get("description", ""),
        value.get("keywords", []),
        value.get("ocr_text", []),
    )


def _parse_completed_core_fields(raw: str) -> PhotoTags:
    """Recover complete description/keywords when only trailing OCR was truncated."""
    decoder = json.JSONDecoder()

    def field(name: str) -> object:
        match = re.search(rf'"{re.escape(name)}"\s*:\s*', raw)
        if match is None:
            raise ValueError(f"模型输出缺少 {name}")
        value, _ = decoder.raw_decode(raw[match.end() :])
        return value

    return normalize_tags(field("description"), field("keywords"), [])


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[,，、;；\n]+", value)
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value if item is not None]
    else:
        values = []
    return [cleaned for item in values if (cleaned := _clean_text(item))]


def _clean_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.strip("# ,，、;；。\"'[]")


def _deduplicate(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def ensure_local_model(model_dir: Path, *, auto_download: bool) -> tuple[Path, Path]:
    model_path = (model_dir / DEFAULT_MODEL_FILE).resolve()
    mmproj_path = (model_dir / DEFAULT_MMPROJ_FILE).resolve()
    if model_path.is_file() and mmproj_path.is_file():
        return model_path, mmproj_path
    if not auto_download:
        raise FileNotFoundError(
            f"模型文件不存在：{model_dir}\n请去掉 --no-auto-download，或手动放入官方 GGUF 文件。"
        )
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("自动下载模型需要安装 modelscope") from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        model_id=DEFAULT_MODEL_ID,
        local_dir=str(model_dir),
        allow_file_pattern=[DEFAULT_MODEL_FILE, DEFAULT_MMPROJ_FILE],
        max_workers=4,
    )
    if not model_path.is_file() or not mmproj_path.is_file():
        raise FileNotFoundError("模型下载结束，但没有找到 Q8_0 主模型或 Q8_0 视觉编码器")
    return model_path, mmproj_path


def locate_llama_server(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(DEFAULT_LLAMA_SERVER)
    discovered = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "没有找到 llama-server.exe。请先运行 setup_tools.py 安装项目所需工具。"
    )


class LlamaCppPhotoTagger:
    """Run the official Qwen 8B Q8 GGUF through a localhost-only llama.cpp server."""

    def __init__(
        self,
        model_path: Path,
        mmproj_path: Path,
        *,
        server_executable: str | Path | None = None,
        log_path: Path | None = None,
        context_size: int = 12288,
        max_new_tokens: int = 640,
        image_min_tokens: int = 1024,
        image_max_tokens: int = 3072,
        verify_tags: bool = True,
    ) -> None:
        self.server_executable = locate_llama_server(server_executable)
        self.model_path = model_path.resolve()
        self.mmproj_path = mmproj_path.resolve()
        self.context_size = max(4096, int(context_size))
        self.max_new_tokens = max(128, int(max_new_tokens))
        self.image_min_tokens = max(256, int(image_min_tokens))
        self.image_max_tokens = max(self.image_min_tokens, int(image_max_tokens))
        self.verify_tags = bool(verify_tags)
        self.last_timings: dict[str, Any] = {}
        self._log_path = (log_path or model_path.parent / "llama-server.log").resolve()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None
        self._has_started = False
        self.restart_count = 0
        self._start_server()

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _server_command(self) -> list[str]:
        return [
            str(self.server_executable),
            "--model",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--n-gpu-layers",
            "99",
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            "1",
            "--batch-size",
            "2048",
            "--ubatch-size",
            "512",
            "--flash-attn",
            "on",
            "--image-min-tokens",
            str(self.image_min_tokens),
            "--image-max-tokens",
            str(self.image_max_tokens),
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--api-key",
            self._api_key,
            "--no-ui",
        ]

    def _start_server(self) -> None:
        self._port = _free_local_port()
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._api_key = secrets.token_urlsafe(32)
        mode = "a" if self._has_started else "w"
        self._log_handle = self._log_path.open(mode, encoding="utf-8")
        if self._has_started:
            self._log_handle.write(
                f"\n=== PhotoAI restart {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            self._log_handle.flush()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            self._server_command(),
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )
        try:
            self._wait_until_ready(timeout=180)
            self._has_started = True
        except Exception:
            self.close()
            raise

    def restart(self) -> None:
        """Fully recreate llama-server so CUDA and prompt caches are released."""
        self.close()
        self.restart_count += 1
        self._start_server()

    def analyze(self, images: Sequence[Image.Image]) -> PhotoTags:
        if not images:
            raise ValueError("至少需要一张图像")
        content: list[dict[str, Any]] = []
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image)},
                }
            )
        content.append({"type": "text", "text": TAGGING_PROMPT})
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        response = self._complete(messages)
        raw_first_timings = response.get("timings", {})
        first_timings: dict[str, Any] = (
            dict(raw_first_timings)
            if isinstance(raw_first_timings, dict)
            else {"initial": raw_first_timings}
        )
        try:
            first_pass = parse_model_response(_response_text(response))
        except ValueError as exc:
            retry_content = [*content[:-1], {"type": "text", "text": RECOVERY_PROMPT}]
            retry_response = self._complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": retry_content},
                ]
            )
            first_timings.update(
                {
                    "recovery": retry_response.get("timings", {}),
                    "recovery_used": True,
                    "initial_error": str(exc),
                }
            )
            retry_text = _response_text(retry_response)
            try:
                first_pass = parse_model_response(retry_text)
            except ValueError as recovery_exc:
                # OCR-heavy photos occasionally repeat text until max_tokens even
                # after the compact retry. Description and keywords precede OCR,
                # so salvage them only when both fields are complete JSON values.
                recovered = None
                for candidate_text in (retry_text, _response_text(response)):
                    try:
                        recovered = _parse_completed_core_fields(candidate_text)
                        break
                    except (ValueError, json.JSONDecodeError):
                        continue
                if recovered is None:
                    raise recovery_exc
                first_timings["partial_core_recovery"] = True
                first_pass = recovered
        if not self.verify_tags:
            self.last_timings = {"first_pass": first_timings}
            return first_pass

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(first_pass.to_dict(), ensure_ascii=False),
                },
                {"role": "user", "content": VERIFICATION_PROMPT},
            ]
        )
        verified_response = self._complete(messages)
        self.last_timings = {
            "first_pass": first_timings,
            "verification": verified_response.get("timings", {}),
        }
        try:
            return parse_model_response(_response_text(verified_response))
        except ValueError as exc:
            # The first pass is already valid. A long OCR-heavy verification can hit
            # max_tokens before closing its JSON, so retain the usable result instead
            # of failing the entire photo.
            self.last_timings["verification_fallback"] = True
            self.last_timings["verification_error"] = str(exc)
            return first_pass

    def _complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": "Qwen3-VL-8B-Instruct-Q8_0",
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "seed": 3407,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        return self._post_json("/v1/chat/completions", payload, timeout=300)

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_handle = getattr(self, "_log_handle", None)
        if log_handle is not None and not log_handle.closed:
            log_handle.close()

    def __enter__(self) -> "LlamaCppPhotoTagger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self._log_handle.flush()
                tail = _tail_text(self._log_path, 5000)
                raise RuntimeError(f"llama.cpp 启动失败（退出码 {self._process.returncode}）：\n{tail}")
            try:
                request = urllib.request.Request(
                    self._base_url + "/health",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = str(exc)
            time.sleep(0.5)
        self._log_handle.flush()
        raise TimeoutError(
            f"llama.cpp 在 {timeout:.0f} 秒内没有就绪：{last_error}\n{_tail_text(self._log_path, 3000)}"
        )

    def _post_json(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._base_url + path,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            error_type = LocalAIServerError if exc.code >= 500 else RuntimeError
            raise error_type(
                f"本地 AI 请求失败（HTTP {exc.code}）：{details[:1000]}"
            ) from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            raise LocalAIServerError(f"无法连接本地 AI：{exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LocalAIServerError("本地 AI 连接中断，返回内容不完整。") from exc


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _response_text(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"llama.cpp 返回结构异常：{str(response)[:500]}") from exc


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _tail_text(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


MODEL_PROMPT_VERSION = PROMPT_VERSION
