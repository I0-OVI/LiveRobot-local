"""
Tool management module
Handles time queries, API calls and other tool functions
"""
import re
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


class ToolManager:
    """Tool manager, handles tool call requests"""
    
    TOOL_CALL_PATTERNS = {
        'time': [
            r'\[TOOL:GET_TIME\]',
            r'\[Tool:Get\s+Time\]',
            r'\[tool:get_time\]',
            r'\[工具:获取时间\]',
            r'请求获取时间',
            r'需要知道现在的时间',
            r'现在几点了',
        ],
        'weather': [
            r'\[Tool:Get\s+Weather\][^\]]*城市[:：]?\s*([^\]]+)',
            r'\[Tool:Get\s+weather\][^\]]*城市[:：]?\s*([^\]]+)',
            r'\[Tool:get\s+weather\][^\]]*城市[:：]?\s*([^\]]+)',
            r'\[TOOL:GET_WEATHER\][^\]]*城市[:：]?\s*([^\]]+)',
            r'\[tool:get_weather\][^\]]*城市[:：]?\s*([^\]]+)',
            r'\[Tool:Get\s+Weather\]',
            r'\[Tool:Get\s+weather\]',
            r'\[Tool:get\s+weather\]',
            r'\[TOOL:GET_WEATHER\]',
            r'\[tool:get_weather\]',
            r'\[工具:获取天气\]\s*城市[:：]?\s*([^\]]+)',
            r'\[工具:获取天气\]',
            r'请求获取天气.*?城市[:：]?\s*([^\]]+)',
            r'查询天气.*?城市[:：]?\s*([^\]]+)',
        ],
        'exchange_rate': [
            r'\[TOOL:GET_EXCHANGE_RATE\]\s*货币[:：]?\s*([^\]]+)',
            r'\[Tool:Get\s+Exchange\s+Rate\]\s*货币[:：]?\s*([^\]]+)',
            r'\[tool:get_exchange_rate\]\s*货币[:：]?\s*([^\]]+)',
            r'\[TOOL:GET_EXCHANGE_RATE\]',
            r'\[Tool:Get\s+Exchange\s+Rate\]',
            r'\[工具:获取汇率\]\s*货币[:：]?\s*([^\]]+)',
            r'\[工具:获取汇率\]',
            r'请求获取汇率.*?货币[:：]?\s*([^\]]+)',
            r'查询汇率.*?货币[:：]?\s*([^\]]+)',
        ],
    }
    
    KEYWORD_TRIGGERS = {
        'time': [
            r'现在.*?时间|现在.*?几点|当前时间|现在几点了|现在.*?什么时候|时间.*?多少',
            r'what.*?time|current.*?time|what.*?time.*?is.*?it',
        ],
        'weather': [
            r'天气|气温|温度|下雨|晴天|阴天|多云|下雪|刮风',
            r'weather|temperature|rain|sunny|cloudy|snow',
            r'(.+?)(?:的|地)?(?:天气|气温|温度)',
            r'(.+?)(?:的|地)?(?:weather|temperature)',
        ],
        'exchange_rate': [
            r'汇率|兑换|换汇|货币.*?汇率|(.+?)(?:对|兑|换)(.+?)(?:的|地)?(?:汇率|兑换率)',
            r'exchange.*?rate|currency.*?rate|(.+?)(?:to|against|vs)(.+?)(?:rate|exchange)',
        ],
    }
    
    CITY_PATTERNS = [
        r'(北京|上海|广州|深圳|杭州|南京|成都|武汉|西安|重庆|天津|苏州|长沙|郑州|东莞|青岛|沈阳|大连|厦门|福州|济南|合肥|昆明|哈尔滨|石家庄|太原|南昌|贵阳|南宁|海口|兰州|银川|西宁|乌鲁木齐|拉萨|呼和浩特|香港|澳门|台北)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\b(Beijing|Shanghai|Guangzhou|Shenzhen|Hangzhou|Nanjing|Chengdu|Wuhan|Xi\'?an|Chongqing|Tianjin|Suzhou|Changsha|Zhengzhou|Dongguan|Qingdao|Shenyang|Dalian|Xiamen|Fuzhou|Jinan|Hefei|Kunming|Harbin|Shijiazhuang|Taiyuan|Nanchang|Guiyang|Nanning|Haikou|Lanzhou|Yinchuan|Xining|Urumqi|Lhasa|Hohhot|Hong\s*Kong|Macau|Taipei|Tokyo|New\s*York|London|Paris|Berlin|Moscow|Sydney|Melbourne|Toronto|Vancouver|Los\s*Angeles|San\s*Francisco|Chicago|Miami|Seattle|Boston|Washington)\b',
    ]
    
    CURRENCY_PATTERNS = [
        r'([A-Z]{3})[/\-]?([A-Z]{3})',
        r'(人民币|美元|欧元|英镑|日元|港币|澳元|加元|瑞士法郎|新西兰元)',
    ]
    
    def __init__(self, enable_keyword_trigger: bool = True):
        """
        Initialize tool manager
        
        Args:
            enable_keyword_trigger: Whether to enable keyword auto-trigger (default True)
        """
        self.weather_api_key = os.getenv('WEATHER_API_KEY', '')
        self.exchange_rate_api_key = os.getenv('EXCHANGE_RATE_API_KEY', '')
        self.enable_keyword_trigger = enable_keyword_trigger
    
    def _extract_city_from_text(self, text: str) -> Optional[str]:
        """Extract city name from text"""
        for pattern in self.CITY_PATTERNS:
            match = re.search(pattern, text)
            if match:
                city = match.group(1).strip()
                if city and len(city) > 0:
                    return city
        return None
    
    def _extract_currency_from_text(self, text: str) -> Optional[str]:
        """Extract currency pair from text"""
        for pattern in self.CURRENCY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    base = match.group(1).upper()
                    target = match.group(2).upper()
                    return f"{base}/{target}"
                elif len(match.groups()) == 1:
                    currency = match.group(1)
                    currency_map = {
                        '人民币': 'CNY', '美元': 'USD', '欧元': 'EUR', '英镑': 'GBP',
                        '日元': 'JPY', '港币': 'HKD', '澳元': 'AUD', '加元': 'CAD'
                    }
                    if currency in currency_map:
                        return f"{currency_map[currency]}/CNY"
        return None
    
    def detect_tool_marker(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Detect tool call markers in text"""
        text = text.strip()
        
        for pattern in self.TOOL_CALL_PATTERNS['time']:
            if re.search(pattern, text, re.IGNORECASE):
                return ('time', {})
        
        for pattern in self.TOOL_CALL_PATTERNS['weather']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                city = ""
                try:
                    if match.groups() and len(match.groups()) > 0:
                        city_group = match.group(1)
                        if city_group:
                            city = city_group.strip()
                except (IndexError, AttributeError):
                    city = ""
                return ('weather', {'city': city})
        
        for pattern in self.TOOL_CALL_PATTERNS['exchange_rate']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                currency = ""
                try:
                    if match.groups() and len(match.groups()) > 0:
                        currency_group = match.group(1)
                        if currency_group:
                            currency = currency_group.strip()
                except (IndexError, AttributeError):
                    currency = ""
                return ('exchange_rate', {'currency': currency})
        
        return None
    
    def detect_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Detect tool call requests in text (including markers and keywords)"""
        text = text.strip()
        
        marker_result = self.detect_tool_marker(text)
        if marker_result:
            return marker_result
        
        if self.enable_keyword_trigger:
            for pattern in self.KEYWORD_TRIGGERS['time']:
                if re.search(pattern, text, re.IGNORECASE):
                    return ('time', {})
            
            for pattern in self.KEYWORD_TRIGGERS['weather']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    city = self._extract_city_from_text(text)
                    if not city:
                        try:
                            if match.groups() and len(match.groups()) > 0:
                                city = match.group(1).strip()
                        except (IndexError, AttributeError):
                            pass
                    return ('weather', {'city': city or ''})
            
            for pattern in self.KEYWORD_TRIGGERS['exchange_rate']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    currency = self._extract_currency_from_text(text)
                    if not currency:
                        try:
                            if match.groups() and len(match.groups()) >= 2:
                                base = match.group(1).strip()
                                target = match.group(2).strip()
                                currency = f"{base}/{target}"
                        except (IndexError, AttributeError):
                            pass
                    return ('exchange_rate', {'currency': currency or ''})
        
        return None
    
    def remove_tool_markers(self, text: str) -> str:
        """Remove all tool call markers from text"""
        if not text:
            return text
        
        processed_text = text
        
        for tool_name, patterns in self.TOOL_CALL_PATTERNS.items():
            for pattern in patterns:
                processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
        
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        return processed_text
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any] = None) -> str:
        """Execute tool call"""
        if params is None:
            params = {}
        
        if tool_name == 'time':
            return self._get_time()
        elif tool_name == 'weather':
            return self._request_weather(params.get('city', ''))
        elif tool_name == 'exchange_rate':
            return self._request_exchange_rate(params.get('currency', ''))
        else:
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
        
        if not REQUESTS_AVAILABLE:
            return f"抱歉，无法查询{city}的天气信息，因为缺少必要的库。"
        
        try:
            url = f"https://wttr.in/{city}?format=j1&lang=zh"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get('current_condition', [{}])[0]
                location = data.get('nearest_area', [{}])[0].get('areaName', [{}])[0].get('value', city)
                temp_c = current.get('temp_C', 'N/A')
                desc = current.get('lang_zh', [{}])[0].get('value', current.get('weatherDesc', [{}])[0].get('value', '未知'))
                weather_info = f"{location}天气{desc}，气温{temp_c}°C"
                return weather_info
            else:
                return f"抱歉，查询{city}的天气信息时出现错误。"
        except Exception:
            return f"抱歉，无法查询{city}的天气信息。"
    
    def _request_exchange_rate(self, currency: str) -> str:
        """Get exchange rate information"""
        if not currency:
            return "需要查询汇率，但未指定货币。请告诉我你想查询哪个货币的汇率？"
        
        if not REQUESTS_AVAILABLE:
            return f"抱歉，无法查询{currency}的汇率信息，因为缺少必要的库。"
        
        try:
            currency = currency.replace('-', '').replace(' ', '').replace('/', '').upper()
            
            if len(currency) < 6:
                return f"货币格式不正确，请使用格式如 'USD/CNY' 或 'USDCNY'。"
            
            base_currency = currency[:3]
            target_currency = currency[3:6] if len(currency) >= 6 else 'CNY'
            
            if self.exchange_rate_api_key:
                url = f"http://data.fixer.io/api/latest"
                params = {
                    'access_key': self.exchange_rate_api_key,
                    'base': base_currency,
                    'symbols': target_currency
                }
            else:
                url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
                params = None
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if self.exchange_rate_api_key:
                    if data.get('success', False):
                        rate = data.get('rates', {}).get(target_currency)
                        if rate:
                            return f"{base_currency} 对 {target_currency} 的汇率是 1 {base_currency} = {rate:.4f} {target_currency}"
                    else:
                        error_msg = data.get('error', {}).get('info', '未知错误')
                        return f"抱歉，查询汇率时出现错误：{error_msg}"
                else:
                    rates = data.get('rates', {})
                    if target_currency in rates:
                        rate = rates[target_currency]
                        return f"{base_currency} 对 {target_currency} 的汇率是 1 {base_currency} = {rate:.4f} {target_currency}"
                    else:
                        return f"抱歉，未找到货币 {target_currency} 的汇率信息。"
            else:
                return f"抱歉，查询{currency}的汇率信息时出现错误。"
        except Exception:
            return f"抱歉，无法查询{currency}的汇率信息。"
    
    def process_response(self, response: str) -> Tuple[str, bool]:
        """
        Process AI-generated response, detect and handle tool calls
        Note: Only detects tool call markers, not keywords (to avoid false triggers from keywords in AI responses)
        
        Args:
            response: AI-generated response text
            
        Returns:
            (processed response, whether tool call was detected)
        """
        # Only detect tool call markers, not keywords (to avoid false triggers from keywords in AI responses)
        tool_call = self.detect_tool_marker(response)
        
        if tool_call:
            tool_name, params = tool_call
            tool_result = self.execute_tool(tool_name, params)
            
            # Remove tool call markers, keep other text
            processed_response = response
            
            # Remove tool call markers (using more precise pattern matching, supports various formats)
            for pattern in self.TOOL_CALL_PATTERNS.get(tool_name, []):
                # Remove matched tool call markers
                processed_response = re.sub(pattern, '', processed_response, flags=re.IGNORECASE)
            
            # Clean up extra spaces and newlines
            processed_response = re.sub(r'\s+', ' ', processed_response).strip()
            
            # If response is empty or only contains tool call markers, return tool result directly
            if not processed_response or len(processed_response) < 3:
                return tool_result, True
            
            # Otherwise, add tool result to response (separated by newline, more natural)
            # If response ends with period, question mark, exclamation mark, add directly; otherwise add period
            if processed_response and processed_response[-1] not in '。！？.!?':
                processed_response += '。'
            final_response = f"{processed_response} {tool_result}"
            return final_response, True
        
        return response, False
