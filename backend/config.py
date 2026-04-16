from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
OPENAI_TRANSCRIBE_MODEL = "whisper-1"

BILIBILI_API_VIEW = "https://api.bilibili.com/x/web-interface/view"
BILIBILI_API_REPLY = "https://api.bilibili.com/x/v2/reply/main"
