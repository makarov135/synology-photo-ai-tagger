from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from .ai import PhotoTags
from .media import AssetCandidate


class SynologyError(RuntimeError):
    """Raised when Synology Photos cannot be queried or updated safely."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        api: str | None = None,
        method: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.api = api
        self.method = method


@dataclass(frozen=True)
class SynologySyncResult:
    item_id: int
    folder: str
    filename: str
    tags: list[str]
    description: str
    description_action: str
    skipped_tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "folder": self.folder,
            "filename": self.filename,
            "tags": self.tags,
            "description": self.description,
            "description_action": self.description_action,
            "skipped_tags": self.skipped_tags,
        }


def normalize_photo_folder(value: str) -> str:
    parts = [part for part in value.replace("\\", "/").split("/") if part]
    return "/" + "/".join(parts) if parts else "/"


def infer_personal_photo_root(image_dir: Path) -> str:
    """Convert a personal-space UNC path into the folder name used by Photos API."""
    parts = list(PureWindowsPath(str(image_dir)).parts)
    photos_index = next(
        (index for index, part in enumerate(parts) if part.casefold() == "photos"),
        None,
    )
    if photos_index is None:
        raise SynologyError(
            "无法从图片目录推断群晖个人空间路径；请使用 --synology-photo-root 指定，"
            "例如 /MobileBackup/iPhone/2026/09。"
        )
    return normalize_photo_folder("/".join(parts[photos_index + 1 :]))


def infer_synology_url(image_dir: Path) -> str:
    text = str(image_dir)
    if not text.startswith("\\\\"):
        raise SynologyError(
            "无法从本地或映射盘路径推断群晖地址；请使用 --synology-url 指定，"
            "例如 https://NAS_IP:5001。"
        )
    host = PureWindowsPath(text).parts[0].lstrip("\\")
    if not host:
        raise SynologyError("无法从 UNC 图片目录推断群晖地址。")
    return f"https://{host}:5001"


def normalize_synology_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SynologyError(
            "群晖地址格式错误；请填写 https://主机:端口，或完整的 /webapi/entry.cgi 地址。"
        )
    path = parsed.path.rstrip("/")
    known_suffixes = (
        "/webapi/entry.cgi",
        "/webapi/query.cgi",
        "/webapi/auth.cgi",
    )
    lowered = path.casefold()
    for suffix in known_suffixes:
        if lowered.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if path:
        raise SynologyError(f"群晖地址含有无法识别的路径：{parsed.path}")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


class SynologyPhotosClient:
    REQUIRED_APIS = {
        "SYNO.Foto.Browse.Folder": 1,
        "SYNO.Foto.Browse.Item": 1,
        "SYNO.Foto.Browse.GeneralTag": 1,
    }

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        otp_code: str | None = None,
        *,
        verify_tls: bool = True,
        timeout: float = 30.0,
        requester: Callable[[dict[str, str]], dict[str, Any]] | None = None,
    ) -> None:
        self.base_url = normalize_synology_url(base_url)
        self.endpoint = f"{self.base_url}/webapi/entry.cgi"
        self.username = username
        self.password = password
        self.otp_code = otp_code or None
        self.timeout = timeout
        self._requester = requester
        self._ssl_context = (
            ssl.create_default_context()
            if verify_tls
            else ssl._create_unverified_context()  # noqa: SLF001
        )
        self._sid: str | None = None
        self._syno_token: str | None = None
        self._versions: dict[str, int] = {}
        self._folders: dict[str, int] = {}
        self._items: dict[tuple[int, str], list[dict[str, Any]]] = {}
        self._tags: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "SynologyPhotosClient":
        self.login()
        return self

    def __exit__(self, *_args: object) -> None:
        self.logout()

    def _post(self, params: dict[str, Any]) -> dict[str, Any]:
        encoded: dict[str, str] = {}
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, (list, dict)):
                encoded[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            elif isinstance(value, bool):
                encoded[key] = "true" if value else "false"
            else:
                encoded[key] = str(value)
        if self._requester is not None:
            return self._requester(encoded)
        headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
        if self._sid:
            headers["Cookie"] = f"id={self._sid}"
        if self._syno_token:
            headers["X-SYNO-TOKEN"] = self._syno_token
        request = urllib.request.Request(
            self.endpoint,
            data=urllib.parse.urlencode(encoded).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self._ssl_context
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise SynologyError(f"无法连接群晖：{exc.reason}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SynologyError("群晖返回了无法解析的响应。") from exc
        return payload

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        payload = self._post(params)
        if not payload.get("success"):
            code = payload.get("error", {}).get("code", "unknown")
            if params.get("api") == "SYNO.API.Auth":
                if code == 400:
                    raise SynologyError(
                        "群晖登录失败（错误码 400）：账号不存在或密码错误；"
                        "该错误与两步验证码无关。请使用 DSM 的实际登录账号并重新输入密码。"
                    )
                if code == 403:
                    raise SynologyError(
                        "群晖要求两步验证码（错误码 403），但本次输入为空。"
                        "程序不会发送短信或推送；请打开 Synology Secure SignIn/验证器，"
                        "直接键入当前 6 位动态码并按 Enter（输入过程不会回显）。"
                    )
                if code == 404:
                    raise SynologyError(
                        "群晖两步验证失败（错误码 404）：验证码无效、已使用或已经过期。"
                    )
                raise SynologyError(f"群晖登录失败（错误码 {code}），请检查密码和验证码。")
            raise SynologyError(
                f"群晖接口调用失败：{params.get('api')}.{params.get('method')}"
                f"（错误码 {code}）。",
                code=int(code) if isinstance(code, int) else None,
                api=str(params.get("api") or "") or None,
                method=str(params.get("method") or "") or None,
            )
        data = payload.get("data", {})
        return data if isinstance(data, dict) else {"value": data}

    def _call(self, api: str, method: str, **params: Any) -> dict[str, Any]:
        if not self._sid:
            raise SynologyError("尚未登录群晖。")
        request: dict[str, Any] = {
            "api": api,
            "version": self._versions[api],
            "method": method,
            "_sid": self._sid,
            **params,
        }
        if self._syno_token:
            request["SynoToken"] = self._syno_token
        return self._request(request)

    def login(self) -> None:
        query = self._request(
            {
                "api": "SYNO.API.Info",
                "version": 1,
                "method": "query",
                "query": ",".join(self.REQUIRED_APIS),
            }
        )
        for api, minimum in self.REQUIRED_APIS.items():
            info = query.get(api)
            maximum = int(info.get("maxVersion", 0)) if isinstance(info, dict) else 0
            if maximum < minimum:
                raise SynologyError(f"当前 Synology Photos 不提供所需接口：{api}。")
            self._versions[api] = maximum

        params: dict[str, Any] = {
            "api": "SYNO.API.Auth",
            "version": 7,
            "method": "login",
            "account": self.username,
            "passwd": self.password,
            "format": "sid",
            "enable_syno_token": "yes",
        }
        if self.otp_code:
            params["otp_code"] = self.otp_code
        data = self._request(params)
        self._sid = str(data.get("sid", "")) or None
        self._syno_token = str(data.get("synotoken", "")) or None
        if not self._sid:
            raise SynologyError("群晖登录成功响应中没有会话 ID。")

    def logout(self) -> None:
        if not self._sid:
            return
        try:
            self._request(
                {
                    "api": "SYNO.API.Auth",
                    "version": 7,
                    "method": "logout",
                    "_sid": self._sid,
                }
            )
        finally:
            self._sid = None
            self._syno_token = None

    @staticmethod
    def _list(data: dict[str, Any]) -> list[dict[str, Any]]:
        value = data.get("list", [])
        return [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []

    def _paged(self, api: str, method: str = "list", **params: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        offset = 0
        limit = 1000
        while True:
            data = self._call(api, method, offset=offset, limit=limit, **params)
            page = self._list(data)
            result.extend(page)
            if len(page) < limit:
                return result
            offset += len(page)

    def _folders_for_targets(self, folder_names: set[str] | None) -> list[dict[str, Any]]:
        targets = {
            normalize_photo_folder(name).casefold() for name in (folder_names or set())
        }
        folders = self._paged("SYNO.Foto.Browse.Folder")
        if not targets:
            return folders
        seen_ids = {int(folder["id"]) for folder in folders if "id" in folder}
        queue = [
            folder
            for folder in folders
            if "id" in folder
            and any(
                target == normalize_photo_folder(str(folder.get("name", ""))).casefold()
                or target.startswith(
                    normalize_photo_folder(str(folder.get("name", ""))).casefold() + "/"
                )
                for target in targets
            )
        ]
        while queue:
            known_names = {
                normalize_photo_folder(str(folder.get("name", ""))).casefold()
                for folder in folders
            }
            if targets.issubset(known_names):
                break
            parent = queue.pop(0)
            children = self._paged(
                "SYNO.Foto.Browse.Folder", id=int(parent["id"])
            )
            for child in children:
                if "id" not in child or int(child["id"]) in seen_ids:
                    continue
                seen_ids.add(int(child["id"]))
                folders.append(child)
                child_name = normalize_photo_folder(
                    str(child.get("name", ""))
                ).casefold()
                if any(
                    target == child_name or target.startswith(child_name + "/")
                    for target in targets
                ):
                    queue.append(child)
        return folders

    def load_catalog(self, folder_names: set[str] | None = None) -> None:
        folders = self._folders_for_targets(folder_names)
        self._folders = {
            normalize_photo_folder(str(folder["name"])).casefold(): int(folder["id"])
            for folder in folders
            if "id" in folder and folder.get("name")
        }
        if folder_names:
            wanted_ids: set[int] = set()
            missing_folders: list[str] = []
            for name in folder_names:
                folder_id = self._folders.get(normalize_photo_folder(name).casefold())
                if folder_id is None:
                    missing_folders.append(normalize_photo_folder(name))
                else:
                    wanted_ids.add(folder_id)
            if missing_folders:
                raise SynologyError(
                    "群晖 Photos 中找不到文件夹：" + "、".join(sorted(missing_folders))
                )
            items = []
            for folder_id in sorted(wanted_ids):
                items.extend(
                    self._paged(
                        "SYNO.Foto.Browse.Item",
                        folder_id=folder_id,
                        additional=["tag", "description"],
                    )
                )
        else:
            items = self._paged(
                "SYNO.Foto.Browse.Item", additional=["tag", "description"]
            )
        self._items = {}
        for item in items:
            if "folder_id" not in item or not item.get("filename"):
                continue
            key = (int(item["folder_id"]), str(item["filename"]).casefold())
            self._items.setdefault(key, []).append(item)
        self._reload_tags()

    def _reload_tags(self) -> None:
        tags = self._paged("SYNO.Foto.Browse.GeneralTag")
        self._tags = {
            str(tag["name"]).strip().casefold(): tag
            for tag in tags
            if "id" in tag and str(tag.get("name", "")).strip()
        }

    def _ensure_tags(self, names: list[str]) -> tuple[list[tuple[str, int]], list[str]]:
        resolved: list[tuple[str, int]] = []
        skipped: list[str] = []
        for name in names:
            key = name.strip().casefold()
            if not key:
                continue
            tag = self._tags.get(key)
            if tag is None:
                try:
                    data = self._call(
                        "SYNO.Foto.Browse.GeneralTag", "create", name=name.strip()
                    )
                except SynologyError as exc:
                    if exc.code != 120:
                        raise SynologyError(
                            f"群晖无法创建标签“{name}”：{exc}",
                            code=exc.code,
                            api=exc.api,
                            method=exc.method,
                        ) from exc
                    # Photos may have created this tag asynchronously after the
                    # initial catalog was loaded. Refresh and reuse it instead of
                    # treating a duplicate-name response as a photo failure.
                    self._reload_tags()
                    tag = self._tags.get(key)
                    if tag is None:
                        skipped.append(name)
                        continue
                    resolved.append((name, int(tag["id"])))
                    continue
                tag = data.get("tag")
                if not isinstance(tag, dict) or "id" not in tag:
                    raise SynologyError(f"群晖没有返回新标签“{name}”的 ID。")
                self._tags[key] = tag
            resolved.append((name, int(tag["id"])))
        unique = list(dict.fromkeys(resolved))
        return unique, list(dict.fromkeys(skipped))

    @staticmethod
    def _additional(item: dict[str, Any]) -> dict[str, Any]:
        value = item.get("additional", {})
        return value if isinstance(value, dict) else {}

    @classmethod
    def _tag_names(cls, item: dict[str, Any]) -> list[str]:
        value = cls._additional(item).get("tag", [])
        if not isinstance(value, list):
            return []
        return [
            str(tag["name"])
            for tag in value
            if isinstance(tag, dict) and str(tag.get("name", "")).strip()
        ]

    @classmethod
    def _description(cls, item: dict[str, Any]) -> str:
        value = cls._additional(item).get("description", "")
        return str(value or "").strip()

    def _refresh_item(self, folder_id: int, filename: str, item_id: int) -> dict[str, Any]:
        items = self._paged(
            "SYNO.Foto.Browse.Item",
            folder_id=folder_id,
            additional=["tag", "description"],
        )
        matches = [item for item in items if int(item.get("id", -1)) == item_id]
        if len(matches) != 1:
            raise SynologyError(f"写入后无法重新读取群晖照片 ID {item_id}。")
        return matches[0]

    def sync_asset(
        self,
        asset: AssetCandidate,
        tags: PhotoTags,
        *,
        image_dir: Path,
        photo_root: str,
        previous_ai_description: str | None = None,
    ) -> SynologySyncResult:
        if not self._folders:
            self.load_catalog()
        relative_parent = asset.primary_path.relative_to(image_dir).parent
        suffix = relative_parent.as_posix()
        folder = normalize_photo_folder(
            photo_root if suffix == "." else f"{photo_root}/{suffix}"
        )
        folder_id = self._folders.get(folder.casefold())
        if folder_id is None:
            raise SynologyError(f"群晖 Photos 中找不到文件夹：{folder}")
        matches = self._items.get((folder_id, asset.primary_path.name.casefold()), [])
        if len(matches) != 1:
            raise SynologyError(
                f"群晖 Photos 中匹配到 {len(matches)} 个同名项目："
                f"{folder}/{asset.primary_path.name}；为避免写错，已停止。"
            )
        item = matches[0]
        item_id = int(item["id"])

        existing_tag_names = {name.casefold() for name in self._tag_names(item)}
        missing_tag_names = [
            name for name in tags.keywords if name.casefold() not in existing_tag_names
        ]
        resolved_tags, skipped_tag_names = self._ensure_tags(missing_tag_names)
        tag_ids = list(dict.fromkeys(tag_id for _, tag_id in resolved_tags))
        if tag_ids:
            data = self._call(
                "SYNO.Foto.Browse.Item", "add_tag", id=[item_id], tag=tag_ids
            )
            if data.get("error_list"):
                # A bulk request can fail when Photos has already attached one of
                # the tags concurrently. Re-read first, then retry only genuinely
                # missing tags one by one so one bad tag cannot hide the rest.
                refreshed_after_bulk = self._refresh_item(
                    folder_id, asset.primary_path.name, item_id
                )
                present = {
                    name.casefold() for name in self._tag_names(refreshed_after_bulk)
                }
                # A long run may hold stale tag IDs while Photos updates its tag
                # catalog in the background. Reload before individual retries.
                self._reload_tags()
                for name, _old_tag_id in resolved_tags:
                    if name.casefold() in present:
                        continue
                    retry_tags, newly_skipped = self._ensure_tags([name])
                    skipped_tag_names.extend(newly_skipped)
                    if not retry_tags:
                        continue
                    tag_id = retry_tags[0][1]
                    retry_data = self._call(
                        "SYNO.Foto.Browse.Item", "add_tag", id=[item_id], tag=[tag_id]
                    )
                    if retry_data.get("error_list"):
                        skipped_tag_names.append(name)

        skipped_tag_names = list(dict.fromkeys(skipped_tag_names))

        current_description = self._description(item)
        previous = (previous_ai_description or "").strip()
        if not current_description:
            description_action = "written"
        elif current_description == tags.description:
            description_action = "unchanged"
        elif previous and current_description == previous:
            description_action = "updated"
        else:
            description_action = "preserved_manual"
        if description_action in {"written", "updated"}:
            data = self._call(
                "SYNO.Foto.Browse.Item", "set", id=[item_id], description=tags.description
            )
            if data.get("error_list"):
                raise SynologyError(f"群晖拒绝给照片 ID {item_id} 写入描述。")

        expected_description = (
            current_description if description_action == "preserved_manual" else tags.description
        )
        skipped_keys = {name.casefold() for name in skipped_tag_names}
        expected_tags = {
            name.casefold() for name in tags.keywords if name.casefold() not in skipped_keys
        }
        missing: set[str] = set()
        final_description = ""
        final_tags: list[str] = []
        for delay in (0.0, 0.5, 1.0, 2.0):
            if delay:
                time.sleep(delay)
            refreshed = self._refresh_item(folder_id, asset.primary_path.name, item_id)
            final_tags = self._tag_names(refreshed)
            missing = expected_tags - {name.casefold() for name in final_tags}
            final_description = self._description(refreshed)
            if not missing and final_description == expected_description:
                break
        if missing:
            # Some Photos versions acknowledge a bulk add without applying it.
            # Refresh tag IDs, retry the still-missing values individually, then
            # give the database a longer second window to become consistent.
            self._reload_tags()
            keyword_by_key = {name.casefold(): name for name in tags.keywords}
            for key in sorted(missing):
                name = keyword_by_key[key]
                retry_tags, newly_skipped = self._ensure_tags([name])
                skipped_tag_names.extend(newly_skipped)
                if not retry_tags:
                    continue
                self._call(
                    "SYNO.Foto.Browse.Item",
                    "add_tag",
                    id=[item_id],
                    tag=[retry_tags[0][1]],
                )

            skipped_keys = {name.casefold() for name in skipped_tag_names}
            expected_tags = {
                name.casefold()
                for name in tags.keywords
                if name.casefold() not in skipped_keys
            }
            for delay in (0.5, 1.0, 2.0, 4.0):
                time.sleep(delay)
                refreshed = self._refresh_item(
                    folder_id, asset.primary_path.name, item_id
                )
                final_tags = self._tag_names(refreshed)
                missing = expected_tags - {
                    name.casefold() for name in final_tags
                }
                final_description = self._description(refreshed)
                if not missing and final_description == expected_description:
                    break
            if missing:
                skipped_tag_names.extend(keyword_by_key[key] for key in sorted(missing))
                skipped_tag_names = list(dict.fromkeys(skipped_tag_names))
        if final_description != expected_description:
            raise SynologyError("群晖接口返回成功，但多次复查后照片描述仍不一致。")
        return SynologySyncResult(
            item_id=item_id,
            folder=folder,
            filename=asset.primary_path.name,
            tags=final_tags,
            description=final_description,
            description_action=description_action,
            skipped_tags=skipped_tag_names,
        )
