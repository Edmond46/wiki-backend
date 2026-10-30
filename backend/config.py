import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载，避免额外依赖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------- MySQL ----------------
MYSQL_HOST = _env("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = _env_int("MYSQL_PORT", 3306)
MYSQL_USER = _env("MYSQL_USER", "root")
MYSQL_PASSWORD = _env("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = _env("MYSQL_DATABASE", "iwiki")

# ---------------- Qdrant ----------------
# QDRANT_MODE=local 时使用本地文件模式（无需服务端，便于演示/测试）
QDRANT_MODE = _env("QDRANT_MODE", "local")          # local | server
QDRANT_URL = _env("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = _env("QDRANT_API_KEY", "")
QDRANT_PATH = _env("QDRANT_PATH", str(BASE_DIR / "data" / "qdrant"))
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "iwiki_chunks")

# ---------------- Embedding ----------------
# embedding = dashscope(阿里云百炼 OpenAI 兼容接口) | hash(离线哈希向量，降级用)
EMBEDDING_PROVIDER = _env("EMBEDDING_PROVIDER", "hash")
DASHSCOPE_API_KEY = _env("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = _env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL = _env("DASHSCOPE_MODEL", "text-embedding-v3")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024)

# ---------------- OCR ----------------
# TESSERACT_CMD 为空时使用 PATH 中的 tesseract；Windows 一般形如
# C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = _env("TESSERACT_CMD", "")
OCR_LANG = _env("OCR_LANG", "chi_sim+eng")
# PDF 页面文字少于该阈值时判定为扫描页，改走 OCR
OCR_MIN_TEXT_CHARS = _env_int("OCR_MIN_TEXT_CHARS", 20)
OCR_DPI = _env_int("OCR_DPI", 200)
# auto | tesseract | pymupdf（未安装 tesseract 时 OCR 会记录失败原因，不会中断入库）
OCR_ENGINE = _env("OCR_ENGINE", "auto")
# PDF 处理后端：auto（优先 PyMuPDF，回退 pypdf）| pypdf | fitz
PDF_BACKEND = _env("PDF_BACKEND", "auto")

# ---------------- 检索参数 ----------------
CHUNK_SIZE = _env_int("CHUNK_SIZE", 320)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 80)
RECALL_LIMIT = _env_int("RECALL_LIMIT", 30)        # 单路召回条数
RRF_K = _env_int("RRF_K", 60)                      # RRF 平滑常数
WEIGHT_KEYWORD = _env_float("WEIGHT_KEYWORD", 1.0)
WEIGHT_SEMANTIC = _env_float("WEIGHT_SEMANTIC", 1.0)
QUERY_WEIGHT_SENSITIVITY = _env_float("QUERY_WEIGHT_SENSITIVITY", 8.0)  # 动态权重强度

# ---------------- 存储 ----------------
UPLOAD_DIR = Path(_env("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
MAX_CONTENT_MB = _env_int("MAX_CONTENT_MB", 50)
HOT_CACHE_TTL = _env_int("HOT_CACHE_TTL", 60)
