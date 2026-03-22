"""Unit tests for ToolManager (stdlib unittest; loads tool_manager without importing ai package)."""

import importlib.util
import sys
import unittest
from pathlib import Path

_PROGRAM_ROOT = Path(__file__).resolve().parents[1]
if str(_PROGRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROGRAM_ROOT))

_tm_path = Path(__file__).resolve().parent / "tool_manager.py"
_spec = importlib.util.spec_from_file_location("ai_tool_manager_test", _tm_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
ToolManager = _mod.ToolManager

from utils.setup_loader import get_default_weather_city


class TestToolManagerDetection(unittest.TestCase):
    def setUp(self):
        self.tm = ToolManager(enable_keyword_trigger=True)

    def test_mood_cloudy_no_weather_shortcut(self):
        """Broad weather words without a place must not keyword-short-circuit."""
        self.assertIsNone(self.tm.detect_tool_call("今天心情不好阴天"))

    def test_beijing_weather_shortcuts(self):
        r = self.tm.detect_tool_call("北京今天天气怎么样")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "weather")
        self.assertEqual(r[1].get("city"), "北京")

    def test_marker_weather_without_city_still_triggers(self):
        """Model markers are unchanged: empty city still returns weather tool."""
        r = self.tm.detect_tool_marker("[TOOL:GET_WEATHER]")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "weather")

    def test_english_city_not_from_sentence_start_garbage(self):
        """Removed Title Case sentence-start pattern: Hello should not become a city."""
        self.assertIsNone(self.tm.detect_tool_call("Hello what is the weather like"))

    def test_london_in_sentence_triggers_weather(self):
        r = self.tm.detect_tool_call("What's the weather in London tomorrow")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "weather")
        self.assertEqual(r[1].get("city"), "London")

    def test_time_keyword(self):
        r = self.tm.detect_tool_call("现在几点了")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "time")

    def test_explicit_chinese_weather_uses_default_city(self):
        r = self.tm.detect_tool_call("没啥 帮我看看今天天气怎么样")
        self.assertIsNotNone(r)
        self.assertEqual(r[0], "weather")
        expected = get_default_weather_city() or "上海"
        self.assertEqual(r[1].get("city"), expected)

    def test_remove_markers_keeps_normal_chinese(self):
        """Bracket-only marker patterns must not erase conversational phrases."""
        s = self.tm.remove_tool_markers("需要先说完这句，现在几点了都行")
        self.assertIn("需要", s)
        self.assertIn("现在几点了", s)

    def test_no_default_when_disabled(self):
        tm = ToolManager(enable_keyword_trigger=True, default_weather_city="")
        self.assertIsNone(tm.detect_tool_call("今天天气怎么样"))


if __name__ == "__main__":
    unittest.main()
