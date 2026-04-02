"""
Tool management module
Handles time queries, API calls and other tool functions
"""
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Tuple

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    from utils.setup_loader import get_default_weather_city
except ImportError:  # e.g. tests with minimal sys.path
    def get_default_weather_city():  # type: ignore
        return None

try:
    from utils.time_context import local_time_short_for_tool
except ImportError:

    def local_time_short_for_tool() -> str:  # type: ignore
        return ""

try:
    from utils.open_app_registry import (
        extract_open_app_target,
        launch_open_app,
        resolve_open_app,
    )
except ImportError:

    def extract_open_app_target(text: str):  # type: ignore
        return None

    def resolve_open_app(phrase: str):  # type: ignore
        return None

    def launch_open_app(params):  # type: ignore
        return (False, "open_app unavailable")

logger = logging.getLogger(__name__)

# Chinese phrases that indicate a real weather question (allow default city when no place named)
_EXPLICIT_WEATHER_QUERY_ZH = re.compile(
    r"天气(?:怎么样|如何|好不|好嘛|好吗|[\?？])|"
    r"什么天气|啥天气|天气预报|"
    r"(?:看看|看下|查查|查一下|帮我|劳驾).{0,20}天气|"
    r"(?:气温|温度|多少度)(?:怎样|如何|多少|[\?？])?"
)

# Reject (.+?)…weather capture when it is clearly not a place name (English)
_WEATHER_KEYWORD_CITY_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "what",
        "whats",
        "how",
        "is",
        "it",
        "today",
        "now",
        "current",
        "tell",
        "me",
        "about",
        "like",
        "outside",
        "this",
        "that",
        "there",
    }
)


class ToolManager:
    """Tool manager, handles tool call requests"""

    # Only bracket/tool-style markers here. Natural-language phrases belong in KEYWORD_TRIGGERS
    # so remove_tool_markers() does not strip normal model output (e.g. "需要", "现在几点了").
    TOOL_CALL_PATTERNS = {
        "time": [
            r"\[TOOL:GET_TIME\]",
            r"\[Tool:Get\s+Time\]",
            r"\[tool:get_time\]",
            r"\[工具:获取时间\]",
        ],
        "weather": [
            r"\[Tool:Get\s+Weather\][^\]]*城市[:：]?\s*([^\]]+)",
            r"\[Tool:Get\s+weather\][^\]]*城市[:：]?\s*([^\]]+)",
            r"\[Tool:get\s+weather\][^\]]*城市[:：]?\s*([^\]]+)",
            r"\[TOOL:GET_WEATHER\][^\]]*城市[:：]?\s*([^\]]+)",
            r"\[tool:get_weather\][^\]]*城市[:：]?\s*([^\]]+)",
            r"\[Tool:Get\s+Weather\]",
            r"\[Tool:Get\s+weather\]",
            r"\[Tool:get\s+weather\]",
            r"\[TOOL:GET_WEATHER\]",
            r"\[tool:get_weather\]",
            r"\[工具:获取天气\]\s*城市[:：]?\s*([^\]]+)",
            r"\[工具:获取天气\]",
        ],
        "exchange_rate": [
            r"\[TOOL:GET_EXCHANGE_RATE\]\s*货币[:：]?\s*([^\]]+)",
            r"\[Tool:Get\s+Exchange\s+Rate\]\s*货币[:：]?\s*([^\]]+)",
            r"\[tool:get_exchange_rate\]\s*货币[:：]?\s*([^\]]+)",
            r"\[TOOL:GET_EXCHANGE_RATE\]",
            r"\[Tool:Get\s+Exchange\s+Rate\]",
            r"\[工具:获取汇率\]\s*货币[:：]?\s*([^\]]+)",
            r"\[工具:获取汇率\]",
        ],
        "open_app": [
            r"\[TOOL:OPEN_APP\][^\]]*应用[:：]?\s*([^\]]+)",
            r"\[tool:open_app\][^\]]*app[:：]?\s*([^\]]+)",
            r"\[工具:打开程序\][^\]]*应用[:：]?\s*([^\]]+)",
        ],
    }

    KEYWORD_TRIGGERS = {
        "time": [
            r"现在.*?时间|现在.*?几点|当前时间|现在几点了|现在.*?什么时候|时间.*?多少",
            r"请求获取时间|需要知道现在的时间",
            r"what.*?time|current.*?time|what.*?time.*?is.*?it",
        ],
        "weather": [
            r"天气(?:怎么样|如何|好不|好嘛|好吗|[\?？])",
            r"什么天气|啥天气|天气预报",
            r"(?:看看|看下|查查|查一下|帮我|劳驾).{0,20}天气",
            r"(?:气温|温度|多少度)(?:怎样|如何|多少|[\?？])?",
            r"请求获取天气.*?城市[:：]?\s*([^\s\n]+)",
            r"查询天气.*?城市[:：]?\s*([^\s\n]+)",
            r"天气|气温|温度|下雨|晴天|阴天|多云|下雪|刮风",
            r"weather|temperature|rain|sunny|cloudy|snow",
            r"(.+?)(?:的|地)?(?:天气|气温|温度)",
            r"(.+?)(?:的|地)?(?:weather|temperature)",
        ],
        "exchange_rate": [
            r"汇率|兑换|换汇|货币.*?汇率|(.+?)(?:对|兑|换)(.+?)(?:的|地)?(?:汇率|兑换率)",
            r"exchange.*?rate|currency.*?rate|(.+?)(?:to|against|vs)(.+?)(?:rate|exchange)",
            r"请求获取汇率.*?货币[:：]?\s*([^\s\n]+)",
            r"查询汇率.*?货币[:：]?\s*([^\s\n]+)",
        ],
        # Keyword triggers for open_app are handled only via extract_open_app_target + whitelist
        # (no broad regex here — avoids false positives).
        "open_app": [],
    }

    CITY_PATTERNS = [
        r"(北京|上海|广州|深圳|杭州|南京|成都|武汉|西安|重庆|天津|苏州|长沙|郑州|东莞|青岛|沈阳|大连|厦门|福州|济南|合肥|昆明|哈尔滨|石家庄|太原|南昌|贵阳|南宁|海口|兰州|银川|西宁|乌鲁木齐|拉萨|呼和浩特|香港|澳门|台北)",
        r"\b(Beijing|Shanghai|Guangzhou|Shenzhen|Hangzhou|Nanjing|Chengdu|Wuhan|Xi\'?an|Chongqing|Tianjin|Suzhou|Changsha|Zhengzhou|Dongguan|Qingdao|Shenyang|Dalian|Xiamen|Fuzhou|Jinan|Hefei|Kunming|Harbin|Shijiazhuang|Taiyuan|Nanchang|Guiyang|Nanning|Haikou|Lanzhou|Yinchuan|Xining|Urumqi|Lhasa|Hohhot|Hong\s*Kong|Macau|Taipei|Tokyo|New\s*York|London|Paris|Berlin|Moscow|Sydney|Melbourne|Toronto|Vancouver|Los\s*Angeles|San\s*Francisco|Chicago|Miami|Seattle|Boston|Washington)\b",
    ]

    CURRENCY_PATTERNS = [
        r"([A-Z]{3})[/\-]?([A-Z]{3})",
        r"(人民币|美元|欧元|英镑|日元|港币|澳元|加元|瑞士法郎|新西兰元)",
    ]

    def __init__(
        self,
        enable_keyword_trigger: bool = True,
        default_weather_city: Optional[str] = None,
    ):
        """
        Initialize tool manager

        Args:
            enable_keyword_trigger: Whether to enable keyword auto-trigger (default True)
            default_weather_city: When set, overrides env and setup. If None, reads WEATHER_DEFAULT_CITY;
                if that env var is unset, uses city parsed from setup.txt ROLE_ZH / ROLE_EN (e.g. 「住在上海」);
                empty env string disables default city. If setup has no city, falls back to "上海".
        """
        self.weather_api_key = os.getenv("WEATHER_API_KEY", "")
        self.exchange_rate_api_key = os.getenv("EXCHANGE_RATE_API_KEY", "")
        self.enable_keyword_trigger = enable_keyword_trigger

        if default_weather_city is not None:
            self._default_weather_city = default_weather_city.strip() or None
        else:
            raw = os.environ.get("WEATHER_DEFAULT_CITY")
            if raw is not None:
                self._default_weather_city = raw.strip() or None
            else:
                self._default_weather_city = get_default_weather_city() or "上海"

        self._tool_patterns: Dict[str, List[Pattern[str]]] = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in self.TOOL_CALL_PATTERNS.items()
        }
        self._keyword_patterns: Dict[str, List[Pattern[str]]] = {
            k: [re.compile(p, re.IGNORECASE) for p in v]
            for k, v in self.KEYWORD_TRIGGERS.items()
        }
        self._city_patterns = [re.compile(p) for p in self.CITY_PATTERNS]
        self._currency_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CURRENCY_PATTERNS
        ]
        self._ws_collapse = re.compile(r"\s+")

        if REQUESTS_AVAILABLE and requests is not None:
            self._http = requests.Session()
            self._http.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; LiveRobot/1.0) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                }
            )
        else:
            self._http = None

    def _extract_city_from_text(self, text: str) -> Optional[str]:
        """Extract city name from text"""
        for pattern in self._city_patterns:
            match = pattern.search(text)
            if match:
                city = match.group(1).strip()
                if city:
                    return city
        return None

    def _city_from_weather_keyword_match(
        self, text: str, match: re.Match[str]
    ) -> Optional[str]:
        """Resolve city for keyword-triggered weather: full text first, then capture group."""
        city = self._extract_city_from_text(text)
        if city:
            return city
        if not match.lastindex:
            return None
        try:
            raw = match.group(1).strip()
        except (IndexError, AttributeError):
            return None
        if not raw:
            return None
        # Latin junk from (.+?)…weather
        if raw.isascii() and raw.lower() in _WEATHER_KEYWORD_CITY_STOPWORDS:
            return None
        return self._extract_city_from_text(raw)

    def _zh_text(self, text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def _allow_default_weather_city(self, text: str) -> bool:
        """Default city only for clear Chinese weather questions (avoids English false positives)."""
        if not self._default_weather_city:
            return False
        if not self._zh_text(text):
            return False
        return bool(_EXPLICIT_WEATHER_QUERY_ZH.search(text))

    def _extract_currency_from_text(self, text: str) -> Optional[str]:
        """Extract currency pair from text"""
        for pattern in self._currency_patterns:
            match = pattern.search(text)
            if match:
                if len(match.groups()) >= 2:
                    base = match.group(1).upper()
                    target = match.group(2).upper()
                    return f"{base}/{target}"
                if len(match.groups()) == 1:
                    currency = match.group(1)
                    currency_map = {
                        "人民币": "CNY",
                        "美元": "USD",
                        "欧元": "EUR",
                        "英镑": "GBP",
                        "日元": "JPY",
                        "港币": "HKD",
                        "澳元": "AUD",
                        "加元": "CAD",
                    }
                    if currency in currency_map:
                        return f"{currency_map[currency]}/CNY"
        return None

    def detect_tool_marker(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Detect tool call markers in text"""
        text = text.strip()

        for pattern in self._tool_patterns["time"]:
            if pattern.search(text):
                return ("time", {})

        for pattern in self._tool_patterns["weather"]:
            match = pattern.search(text)
            if match:
                city = ""
                try:
                    if match.lastindex:
                        city_group = match.group(1)
                        if city_group:
                            city = city_group.strip()
                except (IndexError, AttributeError):
                    city = ""
                return ("weather", {"city": city})

        for pattern in self._tool_patterns["exchange_rate"]:
            match = pattern.search(text)
            if match:
                currency = ""
                try:
                    if match.lastindex:
                        currency_group = match.group(1)
                        if currency_group:
                            currency = currency_group.strip()
                except (IndexError, AttributeError):
                    currency = ""
                return ("exchange_rate", {"currency": currency})

        for pattern in self._tool_patterns["open_app"]:
            match = pattern.search(text)
            if match:
                raw = match.group(1).strip() if match.lastindex else ""
                spec = resolve_open_app(raw) if raw else None
                if spec:
                    return ("open_app", spec)

        return None

    def detect_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Detect tool call requests in text (including markers and keywords)"""
        text = text.strip()

        marker_result = self.detect_tool_marker(text)
        if marker_result:
            return marker_result

        if self.enable_keyword_trigger:
            for pattern in self._keyword_patterns["time"]:
                if pattern.search(text):
                    return ("time", {})

            for pattern in self._keyword_patterns["weather"]:
                match = pattern.search(text)
                if not match:
                    continue
                city = self._city_from_weather_keyword_match(text, match)
                if city:
                    return ("weather", {"city": city})
                if self._allow_default_weather_city(text):
                    return ("weather", {"city": self._default_weather_city})
                break

            for pattern in self._keyword_patterns["exchange_rate"]:
                match = pattern.search(text)
                if match:
                    currency = self._extract_currency_from_text(text)
                    if not currency:
                        try:
                            if match.lastindex and match.lastindex >= 2:
                                base = match.group(1).strip()
                                target = match.group(2).strip()
                                currency = f"{base}/{target}"
                        except (IndexError, AttributeError):
                            pass
                    return ("exchange_rate", {"currency": currency or ""})

            phrase = extract_open_app_target(text)
            if phrase is not None:
                spec = resolve_open_app(phrase)
                if spec:
                    return ("open_app", spec)

        return None

    def remove_tool_markers(self, text: str) -> str:
        """Remove all tool call markers from text"""
        if not text:
            return text

        processed_text = text
        for patterns in self._tool_patterns.values():
            for pattern in patterns:
                processed_text = pattern.sub("", processed_text)

        processed_text = self._ws_collapse.sub(" ", processed_text).strip()
        return processed_text

    def execute_tool(self, tool_name: str, params: Dict[str, Any] = None) -> str:
        """Execute tool call"""
        if params is None:
            params = {}

        if tool_name == "time":
            return self._get_time()
        if tool_name == "weather":
            return self._request_weather(params.get("city", ""))
        if tool_name == "exchange_rate":
            return self._request_exchange_rate(params.get("currency", ""))
        if tool_name == "open_app":
            ok, msg = launch_open_app(params)
            return msg
        return f"Unknown tool: {tool_name}"

    def _get_time(self) -> str:
        """Get current time"""
        now = datetime.now()
        time_str = now.strftime("%Y年%m月%d日 %H:%M:%S")
        return f"当前时间是：{time_str}"

    def _request_weather(self, city: str) -> str:
        """Get weather information"""
        if not city:
            return "需要查询天气，但未指定城市。请告诉我你想查询哪个城市的天气？"

        if not REQUESTS_AVAILABLE or self._http is None:
            return f"抱歉，无法查询{city}的天气信息，因为缺少必要的库。"

        try:
            url = f"https://wttr.in/{city}?format=j1&lang=zh"
            response = self._http.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                current = data.get("current_condition", [{}])[0]
                temp_c = current.get("temp_C", "N/A")
                desc = current.get("lang_zh", [{}])[0].get(
                    "value",
                    current.get("weatherDesc", [{}])[0].get("value", "未知"),
                )
                # No district/city name — model will phrase naturally for the user
                tline = local_time_short_for_tool()
                if tline:
                    return f"{desc}，气温{temp_c}°C。（{tline}）"
                return f"{desc}，气温{temp_c}°C"
            return f"抱歉，查询{city}的天气信息时出现错误。"
        except Exception as e:
            logger.warning("Weather request failed for %r: %s: %s", city, type(e).__name__, e)
            return f"抱歉，无法查询{city}的天气信息。"

    def _request_exchange_rate(self, currency: str) -> str:
        """Get exchange rate information"""
        if not currency:
            return "需要查询汇率，但未指定货币。请告诉我你想查询哪个货币的汇率？"

        if not REQUESTS_AVAILABLE or self._http is None:
            return f"抱歉，无法查询{currency}的汇率信息，因为缺少必要的库。"

        base_for_log = ""
        try:
            currency = currency.replace("-", "").replace(" ", "").replace("/", "").upper()

            if len(currency) < 6:
                return "货币格式不正确，请使用格式如 'USD/CNY' 或 'USDCNY'。"

            base_currency = currency[:3]
            base_for_log = base_currency
            target_currency = currency[3:6] if len(currency) >= 6 else "CNY"

            if self.exchange_rate_api_key:
                url = "https://data.fixer.io/api/latest"
                params = {
                    "access_key": self.exchange_rate_api_key,
                    "base": base_currency,
                    "symbols": target_currency,
                }
            else:
                url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
                params = None

            response = self._http.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if self.exchange_rate_api_key:
                    if data.get("success", False):
                        rate = data.get("rates", {}).get(target_currency)
                        if rate:
                            return (
                                f"{base_currency} 对 {target_currency} 的汇率是 "
                                f"1 {base_currency} = {rate:.4f} {target_currency}"
                            )
                    error_msg = data.get("error", {}).get("info", "未知错误")
                    return f"抱歉，查询汇率时出现错误：{error_msg}"
                rates = data.get("rates", {})
                if target_currency in rates:
                    rate = rates[target_currency]
                    return (
                        f"{base_currency} 对 {target_currency} 的汇率是 "
                        f"1 {base_currency} = {rate:.4f} {target_currency}"
                    )
                return f"抱歉，未找到货币 {target_currency} 的汇率信息。"
            return f"抱歉，查询{currency}的汇率信息时出现错误。"
        except Exception as e:
            logger.warning(
                "Exchange rate request failed (base=%s): %s: %s",
                base_for_log or "?",
                type(e).__name__,
                e,
            )
            return f"抱歉，无法查询{currency}的汇率信息。"

    def process_response(self, response: str) -> Tuple[str, bool, Optional[Tuple[str, str]]]:
        """
        Process AI-generated response, detect and handle tool calls
        Note: Only detects tool call markers, not keywords (to avoid false triggers from keywords in AI responses)

        Args:
            response: AI-generated response text

        Returns:
            (processed response, whether tool call was detected, optional (tool_name, raw_tool_output))
        """
        tool_call = self.detect_tool_marker(response)

        if tool_call:
            tool_name, params = tool_call
            tool_result = self.execute_tool(tool_name, params)
            trace: Optional[Tuple[str, str]] = (tool_name, tool_result)

            processed_response = response
            for pattern in self._tool_patterns.get(tool_name, []):
                processed_response = pattern.sub("", processed_response)

            processed_response = self._ws_collapse.sub(" ", processed_response).strip()

            if not processed_response or len(processed_response) < 3:
                return tool_result, True, trace

            if processed_response[-1] not in "。！？.!?":
                processed_response += "。"
            return f"{processed_response} {tool_result}", True, trace

        return response, False, None
