"""
唐笑/唐哭 音频播放模块
当用户说"唐笑"或"唐哭"时，从 wav 文件夹播放对应音频，并返回任务完成标识
"""
import os
import time
import threading
from typing import Optional, Tuple

# 关键词与文件名包含的匹配规则（用户输入关键词 -> 文件名需包含）
TANG_KEYWORDS = {
    "唐笑": "唐笑",
    "唐哭": "唐哭",
}

# 仅支持 wav 格式
AUDIO_EXTENSIONS = (".wav",)


def get_wav_dir() -> str:
    """获取 wav 文件夹路径（项目根目录下的 wav）"""
    from utils.path_config import get_project_root
    return os.path.join(get_project_root(), "wav")


# 唐笑/唐哭 对应 picture 文件夹中的贴图（覆盖在 Live2D 头部）
TANG_OVERLAY_IMAGES = {
    "唐笑": "爱音大笑.png",
    "唐哭": "爱音唐哭.png",  # 若无则需用户添加
}


def get_tang_overlay_image_path(keyword: str) -> Optional[str]:
    """获取唐笑/唐哭对应的头部贴图路径"""
    fname = TANG_OVERLAY_IMAGES.get(keyword)
    if not fname:
        return None
    from utils.path_config import get_project_root
    path = os.path.join(get_project_root(), "picture", fname)
    return path if os.path.isfile(path) else None


def _find_audio_for_keyword(wav_dir: str, keyword: str) -> Optional[str]:
    """
    在 wav 目录中查找包含关键词的音频文件
    例如：唐笑 -> 爱音唐笑.wav，唐哭 -> 爱音唐哭.wav
    """
    if not os.path.isdir(wav_dir):
        return None
    pattern = TANG_KEYWORDS.get(keyword)
    if not pattern:
        return None
    for name in os.listdir(wav_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() in AUDIO_EXTENSIONS and pattern in base:
            return os.path.join(wav_dir, name)
    return None


def detect_and_play_tang_audio(user_input: str, wav_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[threading.Event]]:
    """
    检测用户输入是否包含 唐笑/唐哭，若包含则启动 wav 播放（不等待），立即返回
    文字生成可与 wav 并行；TTS 应在 wav 播放完毕后再播
    
    Args:
        user_input: 用户输入文本
        wav_dir: wav 目录路径，None 则自动获取
        
    Returns:
        (keyword, done_event): 若播放了音频，返回 ("唐笑"/"唐哭", Event)；否则 (None, None)
        done_event 在 wav 播放完毕时 set，调用方需 wait 后再播 TTS
    """
    if not user_input or not user_input.strip():
        return None, None
    
    text = user_input.strip()
    wav_dir = wav_dir or get_wav_dir()
    
    for keyword in TANG_KEYWORDS:
        if keyword in text:
            audio_path = _find_audio_for_keyword(wav_dir, keyword)
            if audio_path:
                done_event = threading.Event()
                def _play_in_thread():
                    try:
                        if _play_audio_file(audio_path):
                            print(f"[Tang Audio] 已播放: {keyword} -> {os.path.basename(audio_path)}")
                        else:
                            print(f"[Tang Audio] 播放失败: {audio_path}")
                    except Exception as e:
                        print(f"[Tang Audio] 播放异常: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        done_event.set()
                threading.Thread(target=_play_in_thread, daemon=True).start()
                return keyword, done_event
            else:
                print(f"[Tang Audio] 未找到音频文件（关键词: {keyword}，目录: {wav_dir}）")
            break
    
    return None, None


def _play_audio_file(file_path: str) -> bool:
    """使用 pygame 播放 wav 音频文件"""
    if not os.path.isfile(file_path):
        print(f"[Tang Audio] 文件不存在: {file_path}")
        return False
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.01)
        return True
    except Exception as e:
        print(f"[Tang Audio] pygame 播放失败: {e}")
        return False


def get_tang_task_completed_message(played_keyword: str) -> str:
    """
    获取任务已完成的上下文消息，用于注入到生成提示中
    
    Args:
        played_keyword: "唐笑" 或 "唐哭"
        
    Returns:
        要附加到用户输入前的系统说明
    """
    return f"【系统提示】用户说了「{played_keyword}」，已播放对应音频，该任务已完成。请根据用户原话进行回复。\n\n"
