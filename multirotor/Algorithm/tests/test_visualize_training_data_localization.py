import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import visualize_training_data


class VisualizeTrainingDataLocalizationTests(unittest.TestCase):
    def test_localized_text_returns_chinese_when_ui_lang_is_zh(self):
        original_lang = os.environ.get("AIRSIM_UI_LANG")
        os.environ["AIRSIM_UI_LANG"] = "zh"
        try:
            self.assertEqual(
                visualize_training_data._localized_text("中文提示", "English message"),
                "中文提示",
            )
        finally:
            if original_lang is None:
                os.environ.pop("AIRSIM_UI_LANG", None)
            else:
                os.environ["AIRSIM_UI_LANG"] = original_lang

    def test_localized_text_returns_english_by_default(self):
        original_lang = os.environ.get("AIRSIM_UI_LANG")
        os.environ.pop("AIRSIM_UI_LANG", None)
        try:
            self.assertEqual(
                visualize_training_data._localized_text("中文提示", "English message"),
                "English message",
            )
        finally:
            if original_lang is not None:
                os.environ["AIRSIM_UI_LANG"] = original_lang


if __name__ == "__main__":
    unittest.main()
