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


ROOT = Path(get_setting("CODE_MEMORY_WORKSPACE", Path.cwd())).resolve()
DB_DIR = Path(get_setting("CODE_MEMORY_DB_DIR", Path.cwd()))
_db_path_override = get_setting("CODE_MEMORY_DB_PATH", None)
DB_PATH = Path(_db_path_override if _db_path_override is not None else (DB_DIR / "code_memory.db")).resolve()

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_NAME = str(get_setting("CODE_MEMORY_EMBED_MODEL", DEFAULT_EMBED_MODEL))
EMBED_DIM_RAW = get_setting("CODE_MEMORY_EMBED_DIM")
EMBED_DIM_CONFIGURED = EMBED_DIM_RAW is not None
try:
    EMBED_DIM = int(EMBED_DIM_RAW) if EMBED_DIM_RAW else 384
except Exception:
    EMBED_DIM = 384
    EMBED_DIM_CONFIGURED = False
MODEL_CACHE_DIR = Path(get_setting("CODE_MEMORY_MODEL_DIR", Path.home() / ".cache" / "code-memory"))

DEFAULT_TOP_K = get_int("CODE_MEMORY_TOP_K", 12)
DEFAULT_TOP_P = get_float("CODE_MEMORY_TOP_P", 0.6)
RECENCY_WEIGHT = get_float("CODE_MEMORY_RECENCY_WEIGHT", 0.2)
PRIORITY_WEIGHT = get_float("CODE_MEMORY_PRIORITY_WEIGHT", 0.15)
FTS_BONUS = get_float("CODE_MEMORY_FTS_BONUS", 0.1)
OVERSAMPLE_K = get_int("CODE_MEMORY_OVERSAMPLE_K", 4)

ENABLE_VEC = get_bool("CODE_MEMORY_ENABLE_VEC", True)
ENABLE_FTS = get_bool("CODE_MEMORY_ENABLE_FTS", True)
ENABLE_GRAPH = get_bool("CODE_MEMORY_ENABLE_GRAPH", False)

SUMMARY_CTX = get_int("CODE_MEMORY_SUMMARY_CTX", 2048)
SUMMARY_THREADS = get_int("CODE_MEMORY_SUMMARY_THREADS", 4)
SUMMARY_MAX_TOKENS = get_int("CODE_MEMORY_SUMMARY_MAX_TOKENS", 200)
SUMMARY_TEMPERATURE = get_float("CODE_MEMORY_SUMMARY_TEMPERATURE", 0.2)
SUMMARY_TOP_P = get_float("CODE_MEMORY_SUMMARY_TOP_P", 0.9)
SUMMARY_REPEAT_PENALTY = get_float("CODE_MEMORY_SUMMARY_REPEAT_PENALTY", 1.05)
SUMMARY_N_GPU_LAYERS = get_int("CODE_MEMORY_SUMMARY_GPU_LAYERS", 0)
SUMMARY_MAX_CHARS = get_int("CODE_MEMORY_SUMMARY_MAX_CHARS", 300)
SUMMARY_PROMPT = str(get_setting("CODE_MEMORY_SUMMARY_PROMPT", "") or "")
SUMMARY_AUTO_INSTALL = get_bool("CODE_MEMORY_AUTO_INSTALL", True)
SUMMARY_PIP_ARGS = str(get_setting("CODE_MEMORY_PIP_ARGS", "") or "").strip()

LOG_LEVEL = str(get_setting("CODE_MEMORY_LOG_LEVEL", "INFO")).upper()
LOG_BASE = get_setting("CODE_MEMORY_LOG_DIR") or get_setting("CODE_MEMORY_LOG_FILE")


def _resolve_logfile(base: Optional[str]) -> Optional[Path]:
    if not base:
        return None
    base_path = Path(base)
    if base_path.suffix:
        log_dir = base_path.parent if base_path.parent != Path() else Path.cwd()
    else:
        log_dir = base_path
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return log_dir / f"code-memory-{stamp}.log"


LOG_FILE = _resolve_logfile(LOG_BASE)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=str(LOG_FILE) if LOG_FILE else None,
    filemode="a",
)
logger = logging.getLogger("code-memory")


def _in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def install_llama_cpp() -> None:
    cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
    if SUMMARY_PIP_ARGS:
        cmd.extend(SUMMARY_PIP_ARGS.split())
    else:
        if os.name == "nt":
            cmd.append("--prefer-binary")
    if not SUMMARY_PIP_ARGS and not _in_venv():
        cmd.append("--user")
    logger.info("Attempting to install dependency: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
