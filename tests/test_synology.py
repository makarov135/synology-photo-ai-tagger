from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photo_tagger.ai import PhotoTags
from photo_tagger.media import AssetCandidate
from photo_tagger.synology import (
    SynologyError,
    SynologyPhotosClient,
    infer_personal_photo_root,
    normalize_photo_folder,
    normalize_synology_url,
)


def _asset(root: Path) -> AssetCandidate:
    photo = root / "123.jpg"
    return AssetCandidate("123", "123", photo, (photo,), (), "test-signature")


class SynologyTests(unittest.TestCase):
    def test_path_helpers(self) -> None:
        path = Path(r"\\192.0.2.10\home\Photos\MobileBackup\iPhone\2026\09")
        self.assertEqual(infer_personal_photo_root(path), "/MobileBackup/iPhone/2026/09")
        self.assertEqual(
            normalize_photo_folder("\\MobileBackup\\iPhone\\2026\\09\\"),
            "/MobileBackup/iPhone/2026/09",
        )
        self.assertEqual(
            normalize_synology_url("https://192.0.2.10:5001/webapi/entry.cgi"),
            "https://192.0.2.10:5001",
        )
        self.assertEqual(
            normalize_synology_url("https://192.0.2.10:5001"),
            "https://192.0.2.10:5001",
        )

    def test_sync_adds_tags_and_description_then_verifies(self) -> None:
        calls: list[tuple[str, str]] = []
        state: dict[str, object] = {
            "description": "",
            "tags": [],
            "known_tags": [{"id": 8, "name": "洗车场"}],
        }

        def requester(params: dict[str, str]) -> dict[str, object]:
            api, method = params["api"], params["method"]
            calls.append((api, method))
            if api == "SYNO.API.Info":
                versions = {
                    "SYNO.Foto.Browse.Folder": 2,
                    "SYNO.Foto.Browse.Item": 7,
                    "SYNO.Foto.Browse.GeneralTag": 2,
                }
                return {"success": True, "data": {name: {"maxVersion": version} for name, version in versions.items()}}
            if api == "SYNO.API.Auth":
                return {"success": True, "data": {"sid": "sid"}}
            if api == "SYNO.Foto.Browse.Folder":
                return {"success": True, "data": {"list": [{"id": 42, "name": "/MobileBackup/iPhone/2026/09"}]}}
            if api == "SYNO.Foto.Browse.GeneralTag" and method == "list":
                return {"success": True, "data": {"list": list(state["known_tags"])}}
            if api == "SYNO.Foto.Browse.GeneralTag" and method == "create":
                tag = {"id": 9, "name": params["name"]}
                state["known_tags"].append(tag)
                return {"success": True, "data": {"tag": tag}}
            if api == "SYNO.Foto.Browse.Item" and method == "add_tag":
                ids = json.loads(params["tag"])
                state["tags"] = [tag for tag in state["known_tags"] if tag["id"] in ids]
                return {"success": True, "data": {}}
            if api == "SYNO.Foto.Browse.Item" and method == "set":
                state["description"] = params["description"]
                return {"success": True, "data": {}}
            if api == "SYNO.Foto.Browse.Item" and method == "list":
                item = {
                    "id": 7,
                    "folder_id": 42,
                    "filename": "123.jpg",
                    "additional": {"tag": list(state["tags"]), "description": state["description"]},
                }
                return {"success": True, "data": {"list": [item]}}
            raise AssertionError(params)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = SynologyPhotosClient("https://nas:5001", "user", "password", requester=requester)
            client.login()
            client.load_catalog()
            result = client.sync_asset(
                _asset(root),
                PhotoTags(description="室内自动洗车场", keywords=["洗车场", "汽车"]),
                image_dir=root,
                photo_root="/MobileBackup/iPhone/2026/09",
            )

        self.assertEqual(result.item_id, 7)
        self.assertEqual(result.description, "室内自动洗车场")
        self.assertEqual(set(result.tags), {"洗车场", "汽车"})
        self.assertEqual(result.description_action, "written")
        self.assertIn(("SYNO.Foto.Browse.Item", "add_tag"), calls)
        self.assertIn(("SYNO.Foto.Browse.Item", "set"), calls)

    def test_folder_lookup_descends_only_target_branch(self) -> None:
        folder_calls: list[str | None] = []
        hierarchy = {
            None: [{"id": 1, "name": "/MobileBackup"}, {"id": 99, "name": "/Other"}],
            "1": [{"id": 2, "name": "/MobileBackup/iPhone"}],
            "2": [{"id": 3, "name": "/MobileBackup/iPhone/2026"}],
            "3": [{"id": 4, "name": "/MobileBackup/iPhone/2026/09"}],
        }

        def requester(params: dict[str, str]) -> dict[str, object]:
            api, method = params["api"], params["method"]
            if api == "SYNO.API.Info":
                names = SynologyPhotosClient.REQUIRED_APIS
                return {"success": True, "data": {name: {"maxVersion": 2} for name in names}}
            if api == "SYNO.API.Auth":
                return {"success": True, "data": {"sid": "sid"}}
            if api == "SYNO.Foto.Browse.Folder":
                parent = params.get("id")
                folder_calls.append(parent)
                return {"success": True, "data": {"list": hierarchy.get(parent, [])}}
            if api in {"SYNO.Foto.Browse.Item", "SYNO.Foto.Browse.GeneralTag"} and method == "list":
                return {"success": True, "data": {"list": []}}
            raise AssertionError(params)

        client = SynologyPhotosClient("https://nas:5001", "user", "password", requester=requester)
        client.login()
        client.load_catalog({"/MobileBackup/iPhone/2026/09"})
        self.assertEqual(folder_calls, [None, "1", "2", "3"])
        self.assertNotIn("99", folder_calls)
        self.assertEqual(client._folders["/mobilebackup/iphone/2026/09"], 4)

    def test_sync_refuses_ambiguous_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = SynologyPhotosClient("https://nas:5001", "user", "password", requester=lambda _params: {})
            client._sid = "sid"
            client._versions = {
                "SYNO.Foto.Browse.Folder": 2,
                "SYNO.Foto.Browse.Item": 7,
                "SYNO.Foto.Browse.GeneralTag": 2,
            }
            client._folders = {"/mobilebackup/iphone/2026/09": 42}
            duplicate = {"id": 7, "folder_id": 42, "filename": "123.jpg"}
            client._items = {(42, "123.jpg"): [duplicate, {**duplicate, "id": 8}]}
            with self.assertRaisesRegex(SynologyError, "匹配到 2 个"):
                client.sync_asset(
                    _asset(root),
                    PhotoTags(description="描述", keywords=["甲", "乙", "丙"]),
                    image_dir=root,
                    photo_root="/MobileBackup/iPhone/2026/09",
                )

    def test_sync_recovers_from_partial_bulk_tag_failure(self) -> None:
        known_tags = [
            {"id": 8, "name": "洗车场"},
            {"id": 9, "name": "汽车"},
            {"id": 10, "name": "室内"},
        ]
        assigned_ids = [8]
        add_payloads: list[list[int]] = []

        def requester(params: dict[str, str]) -> dict[str, object]:
            api, method = params["api"], params["method"]
            if api == "SYNO.API.Info":
                return {
                    "success": True,
                    "data": {
                        name: {"maxVersion": 2}
                        for name in SynologyPhotosClient.REQUIRED_APIS
                    },
                }
            if api == "SYNO.API.Auth":
                return {"success": True, "data": {"sid": "sid"}}
            if api == "SYNO.Foto.Browse.Folder":
                return {
                    "success": True,
                    "data": {
                        "list": [
                            {"id": 42, "name": "/MobileBackup/iPhone/2026/09"}
                        ]
                    },
                }
            if api == "SYNO.Foto.Browse.GeneralTag" and method == "list":
                return {"success": True, "data": {"list": known_tags}}
            if api == "SYNO.Foto.Browse.Item" and method == "list":
                item = {
                    "id": 7,
                    "folder_id": 42,
                    "filename": "123.jpg",
                    "additional": {
                        "tag": [tag for tag in known_tags if tag["id"] in assigned_ids],
                        "description": "室内自动洗车场",
                    },
                }
                return {"success": True, "data": {"list": [item]}}
            if api == "SYNO.Foto.Browse.Item" and method == "add_tag":
                ids = json.loads(params["tag"])
                add_payloads.append(ids)
                if len(ids) > 1:
                    assigned_ids.append(ids[0])
                    return {"success": True, "data": {"error_list": [{"code": 1}]}}
                assigned_ids.extend(tag_id for tag_id in ids if tag_id not in assigned_ids)
                return {"success": True, "data": {}}
            raise AssertionError(params)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = SynologyPhotosClient(
                "https://nas:5001", "user", "password", requester=requester
            )
            client.login()
            client.load_catalog()
            result = client.sync_asset(
                _asset(root),
                PhotoTags(
                    description="室内自动洗车场",
                    keywords=["洗车场", "汽车", "室内"],
                ),
                image_dir=root,
                photo_root="/MobileBackup/iPhone/2026/09",
            )

        self.assertEqual(set(result.tags), {"洗车场", "汽车", "室内"})
        self.assertEqual(add_payloads, [[9, 10], [10]])

    def test_create_tag_120_refreshes_and_reuses_concurrent_tag(self) -> None:
        list_calls = 0

        def requester(params: dict[str, str]) -> dict[str, object]:
            nonlocal list_calls
            if params["api"] == "SYNO.Foto.Browse.GeneralTag" and params["method"] == "list":
                list_calls += 1
                tags = [] if list_calls == 1 else [{"id": 9, "name": "汽车"}]
                return {"success": True, "data": {"list": tags}}
            if params["api"] == "SYNO.Foto.Browse.GeneralTag" and params["method"] == "create":
                return {"success": False, "error": {"code": 120}}
            raise AssertionError(params)

        client = SynologyPhotosClient(
            "https://nas:5001", "user", "password", requester=requester
        )
        client._sid = "sid"
        client._versions = {"SYNO.Foto.Browse.GeneralTag": 2}
        client._reload_tags()

        self.assertEqual(client._ensure_tags(["汽车"]), ([("汽车", 9)], []))
        self.assertEqual(list_calls, 2)

    def test_unsupported_tag_is_skipped(self) -> None:
        def requester(params: dict[str, str]) -> dict[str, object]:
            if params["api"] == "SYNO.Foto.Browse.GeneralTag" and params["method"] == "list":
                return {"success": True, "data": {"list": []}}
            if params["api"] == "SYNO.Foto.Browse.GeneralTag" and params["method"] == "create":
                return {"success": False, "error": {"code": 120}}
            raise AssertionError(params)

        client = SynologyPhotosClient(
            "https://nas:5001", "user", "password", requester=requester
        )
        client._sid = "sid"
        client._versions = {"SYNO.Foto.Browse.GeneralTag": 2}

        self.assertEqual(client._ensure_tags(["2017"]), ([], ["2017"]))

    def test_permanently_rejected_item_tag_is_skipped(self) -> None:
        known_tags = [
            {"id": 8, "name": "烤网"},
            {"id": 9, "name": "烧烤"},
            {"id": 10, "name": "食物"},
        ]
        existing_tags = known_tags[1:]

        def requester(params: dict[str, str]) -> dict[str, object]:
            api, method = params["api"], params["method"]
            if api == "SYNO.Foto.Browse.GeneralTag" and method == "list":
                return {"success": True, "data": {"list": known_tags}}
            if api == "SYNO.Foto.Browse.Item" and method == "add_tag":
                return {"success": True, "data": {"error_list": [7]}}
            if api == "SYNO.Foto.Browse.Item" and method == "list":
                return {
                    "success": True,
                    "data": {
                        "list": [
                            {
                                "id": 7,
                                "folder_id": 42,
                                "filename": "123.jpg",
                                "additional": {
                                    "tag": existing_tags,
                                    "description": "烤肉照片",
                                },
                            }
                        ]
                    },
                }
            raise AssertionError(params)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = SynologyPhotosClient(
                "https://nas:5001", "user", "password", requester=requester
            )
            client._sid = "sid"
            client._versions = {
                "SYNO.Foto.Browse.Folder": 2,
                "SYNO.Foto.Browse.Item": 7,
                "SYNO.Foto.Browse.GeneralTag": 2,
            }
            client._folders = {"/mobilebackup/iphone/2026/09": 42}
            client._items = {
                (42, "123.jpg"): [
                    {
                        "id": 7,
                        "folder_id": 42,
                        "filename": "123.jpg",
                        "additional": {
                            "tag": existing_tags,
                            "description": "烤肉照片",
                        },
                    }
                ]
            }
            client._tags = {tag["name"].casefold(): tag for tag in known_tags}
            result = client.sync_asset(
                _asset(root),
                PhotoTags("烤肉照片", ("烤网", "烧烤", "食物")),
                image_dir=root,
                photo_root="/MobileBackup/iPhone/2026/09",
            )

        self.assertEqual(result.skipped_tags, ["烤网"])

    def test_sync_retries_eventually_consistent_readback(self) -> None:
        known_tags = [
            {"id": 8, "name": "猫"},
            {"id": 9, "name": "窗户"},
            {"id": 10, "name": "室内"},
        ]
        assigned = False
        stale_reads = 0

        def requester(params: dict[str, str]) -> dict[str, object]:
            nonlocal assigned, stale_reads
            api, method = params["api"], params["method"]
            if api == "SYNO.Foto.Browse.GeneralTag" and method == "list":
                return {"success": True, "data": {"list": known_tags}}
            if api == "SYNO.Foto.Browse.Item" and method == "add_tag":
                assigned = True
                stale_reads = 1
                return {"success": True, "data": {}}
            if api == "SYNO.Foto.Browse.Item" and method == "list":
                visible = assigned and stale_reads == 0
                if assigned and stale_reads:
                    stale_reads -= 1
                return {
                    "success": True,
                    "data": {
                        "list": [
                            {
                                "id": 7,
                                "folder_id": 42,
                                "filename": "123.jpg",
                                "additional": {
                                    "tag": known_tags if visible else [],
                                    "description": "猫坐在窗边",
                                },
                            }
                        ]
                    },
                }
            raise AssertionError(params)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "photo_tagger.synology.time.sleep"
        ) as sleep:
            root = Path(temporary)
            client = SynologyPhotosClient(
                "https://nas:5001", "user", "password", requester=requester
            )
            client._sid = "sid"
            client._versions = {
                "SYNO.Foto.Browse.Folder": 2,
                "SYNO.Foto.Browse.Item": 7,
                "SYNO.Foto.Browse.GeneralTag": 2,
            }
            client._folders = {"/mobilebackup/iphone/2026/09": 42}
            client._items = {
                (42, "123.jpg"): [
                    {
                        "id": 7,
                        "folder_id": 42,
                        "filename": "123.jpg",
                        "additional": {"tag": [], "description": "猫坐在窗边"},
                    }
                ]
            }
            client._tags = {tag["name"].casefold(): tag for tag in known_tags}
            result = client.sync_asset(
                _asset(root),
                PhotoTags("猫坐在窗边", ("猫", "窗户", "室内")),
                image_dir=root,
                photo_root="/MobileBackup/iPhone/2026/09",
            )

        self.assertEqual(set(result.tags), {"猫", "窗户", "室内"})
        sleep.assert_called_once_with(0.5)

    def test_silent_bulk_failure_retries_tags_individually(self) -> None:
        known_tags = [
            {"id": 8, "name": "城市"},
            {"id": 9, "name": "建筑"},
            {"id": 10, "name": "街道"},
        ]
        assigned_ids: list[int] = []
        add_calls: list[list[int]] = []

        def requester(params: dict[str, str]) -> dict[str, object]:
            api, method = params["api"], params["method"]
            if api == "SYNO.Foto.Browse.GeneralTag" and method == "list":
                return {"success": True, "data": {"list": known_tags}}
            if api == "SYNO.Foto.Browse.Item" and method == "add_tag":
                ids = json.loads(params["tag"])
                add_calls.append(ids)
                if len(ids) == 1:
                    assigned_ids.extend(ids)
                return {"success": True, "data": {}}
            if api == "SYNO.Foto.Browse.Item" and method == "list":
                return {
                    "success": True,
                    "data": {
                        "list": [
                            {
                                "id": 7,
                                "folder_id": 42,
                                "filename": "123.jpg",
                                "additional": {
                                    "tag": [
                                        tag
                                        for tag in known_tags
                                        if tag["id"] in assigned_ids
                                    ],
                                    "description": "城市街道",
                                },
                            }
                        ]
                    },
                }
            raise AssertionError(params)

        with tempfile.TemporaryDirectory() as temporary, patch(
            "photo_tagger.synology.time.sleep"
        ):
            root = Path(temporary)
            client = SynologyPhotosClient(
                "https://nas:5001", "user", "password", requester=requester
            )
            client._sid = "sid"
            client._versions = {
                "SYNO.Foto.Browse.Folder": 2,
                "SYNO.Foto.Browse.Item": 7,
                "SYNO.Foto.Browse.GeneralTag": 2,
            }
            client._folders = {"/mobilebackup/iphone/2026/09": 42}
            client._items = {
                (42, "123.jpg"): [
                    {
                        "id": 7,
                        "folder_id": 42,
                        "filename": "123.jpg",
                        "additional": {"tag": [], "description": "城市街道"},
                    }
                ]
            }
            client._tags = {tag["name"].casefold(): tag for tag in known_tags}
            result = client.sync_asset(
                _asset(root),
                PhotoTags("城市街道", ("城市", "建筑", "街道")),
                image_dir=root,
                photo_root="/MobileBackup/iPhone/2026/09",
            )

        self.assertEqual(set(result.tags), {"城市", "建筑", "街道"})
        self.assertEqual(add_calls, [[8, 9, 10], [8], [9], [10]])


if __name__ == "__main__":
    unittest.main()
