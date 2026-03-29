"""
Text normalization for TTS: remove emoji / pictographs so backends do not read them aloud.
"""
import re

# Broad Unicode ranges covering emoji, dingbats, flags, skin tones, ZWJ / VS16 used in sequences
_EMOJI_AND_PRESENTATION = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F9FF"  # alchemical / supplemental symbols & pictographs
    "\U0001FA00-\U0001FAFF"  # chess / extended-A
    "\U0001F100-\U0001F1FF"  # enclosed alphanumerics / regional indicators (flags)
    "\U0001F200-\U0001F2FF"  # enclosed ideographic supplement
    "\U00002600-\U000026FF"  # misc symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U00002300-\U000023FF"  # misc technical
    "\U0001F3FB-\U0001F3FF"  # skin tone modifiers
    "\U0000200D"  # ZWJ (emoji sequences)
    "\U0000FE0F"  # variation selector-16 (emoji presentation)
    "]+",
    flags=re.UNICODE,
)


def strip_for_tts(text: str) -> str:
    """
    Remove emoji and related symbols from text before sending to TTS.
    Collapses extra spaces left behind.
    """
    if not text:
        return text
    s = _EMOJI_AND_PRESENTATION.sub("", text)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()
