from __future__ import annotations

import json
import unittest

from PIL import Image

from photo_tagger.ai import LlamaCppPhotoTagger, parse_model_response


def _completion(description: str, keywords: list[str], *, prompt_ms: float) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"description": description, "keywords": keywords, "ocr_text": []},
                        ensure_ascii=False,
                    )
                }
            }
        ],
        "timings": {"prompt_ms": prompt_ms},
    }


class FakeTagger(LlamaCppPhotoTagger):
    def __init__(self, responses: list[dict], *, verify_tags: bool) -> None:
        self.responses = responses
        self.verify_tags = verify_tags
        self.last_timings = {}
        self.message_counts: list[int] = []

    def _complete(self, messages: list[dict]) -> dict:
        self.message_counts.append(len(messages))
        return self.responses.pop(0)


class ParseModelResponseTests(unittest.TestCase):
    def test_parses_fenced_json_and_deduplicates(self) -> None:
        raw = """```json
        {"description":"一只狗在夜晚的江边散步", "keywords":["狗","夜景","狗","江边"], "ocr_text":["滨江公园"]}
        ```"""
        result = parse_model_response(raw)
        self.assertEqual(result.description, "一只狗在夜晚的江边散步")
        self.assertEqual(result.keywords, ("狗", "夜景", "江边", "滨江公园"))
        self.assertEqual(result.ocr_text, ("滨江公园",))

    def test_rejects_too_few_keywords(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_response('{"description":"海边", "keywords":["海"], "ocr_text":[]}')

    def test_verification_pass_returns_reviewed_tags_and_both_timings(self) -> None:
        tagger = FakeTagger(
            [
                _completion("一辆相机放在展柜内", ["相机", "展柜", "室内"], prompt_ms=1.0),
                _completion("一台哈苏相机放在展柜内", ["哈苏", "相机", "展柜"], prompt_ms=2.0),
            ],
            verify_tags=True,
        )
        result = tagger.analyze([Image.new("RGB", (2, 2))])
        self.assertEqual(result.description, "一台哈苏相机放在展柜内")
        self.assertEqual(tagger.message_counts, [2, 4])
        self.assertEqual(tagger.last_timings["first_pass"]["prompt_ms"], 1.0)
        self.assertEqual(tagger.last_timings["verification"]["prompt_ms"], 2.0)

    def test_invalid_verification_falls_back_to_valid_first_pass(self) -> None:
        first = _completion(
            "手机屏幕显示赛车游戏主页",
            ["手机屏幕", "赛车游戏", "用户主页"],
            prompt_ms=1.0,
        )
        truncated = {
            "choices": [{"message": {"content": '{"description":"截断","keywords":["赛车"'}}],
            "timings": {"prompt_ms": 2.0},
        }
        tagger = FakeTagger([first, truncated], verify_tags=True)

        result = tagger.analyze([Image.new("RGB", (2, 2))])

        self.assertEqual(result.description, "手机屏幕显示赛车游戏主页")
        self.assertEqual(result.keywords, ("手机屏幕", "赛车游戏", "用户主页"))
        self.assertTrue(tagger.last_timings["verification_fallback"])
        self.assertIn("不是合法 JSON", tagger.last_timings["verification_error"])
        self.assertEqual(tagger.last_timings["verification"]["prompt_ms"], 2.0)

    def test_invalid_first_pass_uses_compact_recovery(self) -> None:
        truncated = {
            "choices": [{"message": {"content": '{"description":"截断"'}}],
            "timings": {"prompt_ms": 1.0},
        }
        recovered = _completion(
            "一只猫坐在窗边", ["猫", "窗户", "室内"], prompt_ms=2.0
        )
        tagger = FakeTagger([truncated, recovered], verify_tags=False)

        result = tagger.analyze([Image.new("RGB", (2, 2))])

        self.assertEqual(result.keywords, ("猫", "窗户", "室内"))
        self.assertEqual(tagger.message_counts, [2, 2])
        self.assertTrue(tagger.last_timings["first_pass"]["recovery_used"])
        self.assertIn("不是合法 JSON", tagger.last_timings["first_pass"]["initial_error"])

    def test_truncated_recovery_salvages_complete_description_and_keywords(self) -> None:
        truncated = {
            "choices": [{"message": {"content": '{"description":"截断"'}}],
            "timings": {"prompt_ms": 1.0},
        }
        ocr_truncated = {
            "choices": [
                {
                    "message": {
                        "content": '{"description":"桌上摆着中式点心","keywords":["点心","餐桌","室内"],"ocr_text":["份份份份'
                    }
                }
            ],
            "timings": {"prompt_ms": 2.0},
        }
        tagger = FakeTagger([truncated, ocr_truncated], verify_tags=False)

        result = tagger.analyze([Image.new("RGB", (2, 2))])

        self.assertEqual(result.description, "桌上摆着中式点心")
        self.assertEqual(result.keywords, ("点心", "餐桌", "室内"))
        self.assertEqual(result.ocr_text, ())
        self.assertTrue(tagger.last_timings["first_pass"]["partial_core_recovery"])

    def test_single_pass_skips_verification(self) -> None:
        tagger = FakeTagger(
            [_completion("一只狗在草地上", ["狗", "草地", "户外"], prompt_ms=1.0)],
            verify_tags=False,
        )
        result = tagger.analyze([Image.new("RGB", (2, 2))])
        self.assertEqual(result.keywords, ("狗", "草地", "户外"))
        self.assertEqual(tagger.message_counts, [2])
        self.assertNotIn("verification", tagger.last_timings)


if __name__ == "__main__":
    unittest.main()
