"""Unit tests for ToolManager (stdlib unittest; loads tool_manager without importing ai package)."""

import importlib.util
import unittest
from pathlib import Path

_tm_path = Path(__file__).resolve().parent / "tool_manager.py"
_spec = importlib.util.spec_from_file_location("ai_tool_manager_test", _tm_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
ToolManager = _mod.ToolManager


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


if __name__ == "__main__":
    unittest.main()
