import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _load_file_config() -> dict:
    # Default config path: src/code-vector-memory.jsonc (repo source root).
    # Can be overridden by CODE_MEMORY_CONFIG_PATH (useful for tests or custom layouts).
    override = os.getenv("CODE_MEMORY_CONFIG_PATH", "").strip()
    if override:
        cfg_path = Path(override)
    else:
        # config.py -> src/code_memory/config.py, so src root is parents[1].
        cfg_path = Path(__file__).resolve().parents[1] / "code-vector-memory.jsonc"
    try:
        if not cfg_path.exists():
            return {}
        raw = cfg_path.read_text(encoding="utf-8")

        def _strip_jsonc(text: str) -> str:
            out = []
            in_string = False
            escape = False
            i = 0
            while i < len(text):
                ch = text[i]
                nxt = text[i + 1] if i + 1 < len(text) else ""

                if in_string:
                    out.append(ch)
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    i += 1
                    continue

                if ch == '"':
                    in_string = True
                    out.append(ch)
                    i += 1
                    continue

                if ch == "/" and nxt == "/":
                    i += 2
                    while i < len(text) and text[i] not in ("\n", "\r"):
                        i += 1
                    continue

                out.append(ch)
                i += 1
            return "".join(out)

        data = json.loads(_strip_jsonc(raw))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_CONFIG_PATH_USED = os.getenv("CODE_MEMORY_CONFIG_PATH", "").strip()
_FILE_CONFIG = _load_file_config()


def get_setting(name: str, default=None):
    # Precedence: environment > config file > code defaults.
    global _CONFIG_PATH_USED, _FILE_CONFIG
    current_path = os.getenv("CODE_MEMORY_CONFIG_PATH", "").strip()
    if current_path != _CONFIG_PATH_USED:
        _CONFIG_PATH_USED = current_path
        _FILE_CONFIG = _load_file_config()
    if name in os.environ:
        return os.environ.get(name)
    if name in _FILE_CONFIG:
        value = _FILE_CONFIG.get(name)
        return default if value is None else value
    return default


def get_bool(name: str, default: bool) -> bool:
    raw = get_setting(name, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    return str(raw).lower() not in ("0", "false", "no")


def get_int(name: str, default: int) -> int:
    raw = get_setting(name, default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def get_float(name: str, default: float) -> float:
    raw = get_setting(name, default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _norm_path(value, *, default: Path) -> Path:
    raw = get_setting(value, None) if isinstance(value, str) else value
    if raw is None:
        path = default
    else:
        path = Path(raw)
    try:
        return path.expanduser().resolve()
    except Exception:
        return path


ROOT = _norm_path("CODE_MEMORY_WORKSPACE", default=Path.cwd())

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_NAME = str(get_setting("CODE_MEMORY_EMBED_MODEL", DEFAULT_EMBED_MODEL))
EMBED_DIM_RAW = get_setting("CODE_MEMORY_EMBED_DIM")
EMBED_DIM_CONFIGURED = EMBED_DIM_RAW is not None
try:
    EMBED_DIM = int(EMBED_DIM_RAW) if EMBED_DIM_RAW else 384
except Exception:
    EMBED_DIM = 384
    EMBED_DIM_CONFIGURED = False
MODEL_CACHE_DIR = _norm_path("CODE_MEMORY_MODEL_DIR", default=(Path.home() / ".cache" / "code-memory"))

DEFAULT_TOP_K = get_int("CODE_MEMORY_TOP_K", 12)
DEFAULT_TOP_P = get_float("CODE_MEMORY_TOP_P", 0.6)
RECENCY_WEIGHT = get_float("CODE_MEMORY_RECENCY_WEIGHT", 0.2)
PRIORITY_WEIGHT = get_float("CODE_MEMORY_PRIORITY_WEIGHT", 0.15)
FTS_BONUS = get_float("CODE_MEMORY_FTS_BONUS", 0.1)
OVERSAMPLE_K = get_int("CODE_MEMORY_OVERSAMPLE_K", 4)

ENABLE_SESSION_SCOPE = get_bool("CODE_MEMORY_ENABLE_SESSION_SCOPE", False)

# libSQL (optional alternative DB backend)
# - If CODE_MEMORY_DB_URL/LIBSQL_URL is set, use libsql-client instead of sqlite3.
# - Works with local sqld ("libsql://...") or remote Turso; user can keep it local by running sqld on localhost.
DB_URL = str(get_setting("CODE_MEMORY_DB_URL", get_setting("LIBSQL_URL", "")) or "").strip()
DB_AUTH_TOKEN = str(get_setting("CODE_MEMORY_DB_AUTH_TOKEN", get_setting("LIBSQL_AUTH_TOKEN", "")) or "").strip()
if not DB_URL:
    raise RuntimeError("CODE_MEMORY_DB_URL (or LIBSQL_URL) is required; this project only supports libSQL backends.")
DB_BACKEND = "libsql"

SESSION_BONUS = get_float("CODE_MEMORY_SESSION_BONUS", 0.2)
CROSS_SESSION_PENALTY = get_float("CODE_MEMORY_CROSS_SESSION_PENALTY", 0.0)

# Local GGUF model settings (used for NER/entity extraction).
NER_CTX = get_int("CODE_MEMORY_NER_CTX", 2048)
NER_THREADS = get_int("CODE_MEMORY_NER_THREADS", 4)
NER_MAX_TOKENS = get_int("CODE_MEMORY_NER_MAX_TOKENS", 800)
NER_TEMPERATURE = get_float("CODE_MEMORY_NER_TEMPERATURE", 0.1)
NER_TOP_P = get_float("CODE_MEMORY_NER_TOP_P", 0.9)
NER_REPEAT_PENALTY = get_float("CODE_MEMORY_NER_REPEAT_PENALTY", 1.05)
NER_N_GPU_LAYERS = get_int("CODE_MEMORY_NER_GPU_LAYERS", 0)
NER_MAX_INPUT_CHARS = get_int("CODE_MEMORY_NER_MAX_INPUT_CHARS", 25000)
NER_PROMPT = str(get_setting("CODE_MEMORY_NER_PROMPT", "") or "")
NER_AUTO_INSTALL = get_bool("CODE_MEMORY_AUTO_INSTALL", True)
PIP_ARGS = str(get_setting("CODE_MEMORY_PIP_ARGS", "") or "").strip()

LOG_LEVEL = str(get_setting("CODE_MEMORY_LOG_LEVEL", "INFO")).upper()
LOG_BASE = get_setting("CODE_MEMORY_LOG_DIR") or get_setting("CODE_MEMORY_LOG_FILE")


def _resolve_logfile(base: Optional[str]) -> Optional[Path]:
    if not base:
        return None
    base_path = Path(base).expanduser()
    if base_path.suffix:
        log_dir = base_path.parent if base_path.parent != Path() else Path.cwd()
    else:
        log_dir = base_path
    log_dir = log_dir.expanduser()
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return (log_dir / f"code-memory-{stamp}.log").expanduser()


LOG_FILE = _resolve_logfile(LOG_BASE)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=str(LOG_FILE) if LOG_FILE else None,
    filemode="a",
)
logger = logging.getLogger("code-memory")

def _is_debug_logging() -> bool:
    try:
        return str(LOG_LEVEL).upper() == "DEBUG"
    except Exception:
        return False


def _in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def _tail(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    if max_chars > 0 and len(text) > max_chars:
        return "...(truncated)...\n" + text[-max_chars:]
    return text


def install_llama_cpp() -> None:
    cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
    if PIP_ARGS:
        cmd.extend(PIP_ARGS.split())
    else:
        if os.name == "nt":
            cmd.append("--prefer-binary")
    if not PIP_ARGS and not _in_venv():
        cmd.append("--user")
    logger.info("ner.install_llama_cpp: cmd=%s", " ".join(cmd))
    logger.info(
        "ner.install_llama_cpp: python=%s venv=%s platform=%s",
        sys.executable,
        _in_venv(),
        sys.platform,
    )
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("ner.install_llama_cpp: pip exit_code=%s", proc.returncode)
        logger.error("ner.install_llama_cpp: pip stdout:\n%s", _tail(proc.stdout))
        logger.error("ner.install_llama_cpp: pip stderr:\n%s", _tail(proc.stderr))
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)
    if _is_debug_logging():
        logger.debug("ner.install_llama_cpp: pip stdout:\n%s", _tail(proc.stdout))
        logger.debug("ner.install_llama_cpp: pip stderr:\n%s", _tail(proc.stderr))
