from __future__ import annotations

import atexit
import os
from pathlib import Path

from .config import (
    MODEL_CACHE_DIR,
    SUMMARY_AUTO_INSTALL,
    SUMMARY_CTX,
    SUMMARY_MAX_CHARS,
    SUMMARY_MAX_TOKENS,
    SUMMARY_N_GPU_LAYERS,
    SUMMARY_PROMPT,
    SUMMARY_REPEAT_PENALTY,
    SUMMARY_TEMPERATURE,
    SUMMARY_THREADS,
    SUMMARY_TOP_P,
    get_float,
    get_int,
    get_setting,
    install_llama_cpp,
    logger,
)


LLAMA_CPP_AVAILABLE = False
Llama = None
_SUMMARY_LLM = None


def _summary_auto_download_enabled() -> bool:
    return os.getenv("CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD", "1").lower() not in ("0", "false", "no")


DEFAULT_SUMMARY_REPO_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"


def looks_like_hf_repo_id(value: str) -> bool:
    v = (value or "").strip()
    if not v or v.endswith(".gguf"):
        return False
    if v.startswith(("http://", "https://")):
        return "huggingface.co/" in v
    return "/" in v and not any(sep in v for sep in ("\\", ":", " "))


def normalize_repo_id(value: str) -> str:
    v = (value or "").strip()
    if v.startswith(("http://", "https://")) and "huggingface.co/" in v:
        v = v.split("huggingface.co/", 1)[1]
        v = v.split("?", 1)[0].strip("/")
    return v


def download_gguf_from_repo(repo_id: str) -> str:
    if not _summary_auto_download_enabled():
        return ""
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
    except Exception as exc:  # pragma: no cover
        logger.warning("huggingface_hub not available for GGUF download: %s", exc)
        return ""

    repo_id = normalize_repo_id(repo_id)
    if not repo_id:
        return ""

    preferred_files = [
        "Qwen2.5-Coder-0.5B-Instruct-Q4_K_M.gguf",
        "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf",
        "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        "qwen2.5-coder-0.5b-instruct-q8_0.gguf",
    ]

    cache_dir = MODEL_CACHE_DIR / "gguf" / repo_id.replace("/", "__")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    try:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception as exc:
        logger.warning("Could not list HF repo files for %s: %s", repo_id, exc)
        return ""

    filename = None
    for cand in preferred_files:
        if cand in files:
            filename = cand
            break

    if not filename:
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        for f in ggufs:
            lf = f.lower()
            if "instruct" in lf and ("q4" in lf or "iq" in lf):
                filename = f
                break
        if not filename and ggufs:
            filename = sorted(ggufs)[0]

    if not filename:
        logger.warning("No .gguf file found in HF repo %s", repo_id)
        return ""

    try:
        logger.info("Downloading GGUF model from HF: %s/%s", repo_id, filename)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(cache_dir),
        )
        return str(Path(path).resolve())
    except Exception as exc:  # pragma: no cover
        logger.warning("GGUF download failed for %s/%s: %s", repo_id, filename, exc)
        return ""


SUMMARY_MODEL = os.getenv("CODE_MEMORY_SUMMARY_MODEL", "").strip()


def resolve_summary_model_path() -> str:
    configured = str(get_setting("CODE_MEMORY_SUMMARY_MODEL", "") or "").strip()
    if configured:
        if Path(configured).exists():
            return configured
        if looks_like_hf_repo_id(configured):
            if not _summary_auto_download_enabled():
                return ""
            return download_gguf_from_repo(configured)
        return configured
    return ""


def ensure_summary_model_path() -> str:
    global SUMMARY_MODEL
    if SUMMARY_MODEL:
        try:
            if Path(SUMMARY_MODEL).exists():
                return SUMMARY_MODEL
        except Exception:
            pass
        if looks_like_hf_repo_id(SUMMARY_MODEL):
            resolved = download_gguf_from_repo(SUMMARY_MODEL)
            if resolved:
                SUMMARY_MODEL = resolved
            return SUMMARY_MODEL
        return SUMMARY_MODEL

    resolved = resolve_summary_model_path()
    if resolved:
        SUMMARY_MODEL = resolved
        return SUMMARY_MODEL

    return ""


def ensure_llama_cpp() -> bool:
    global LLAMA_CPP_AVAILABLE, Llama
    if LLAMA_CPP_AVAILABLE and Llama is not None:
        return True
    try:
        from llama_cpp import Llama as _Llama

        Llama = _Llama
        LLAMA_CPP_AVAILABLE = True
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("Import llama_cpp failed: %s", exc)
        if not SUMMARY_AUTO_INSTALL:
            return False
    try:
        install_llama_cpp()
        from llama_cpp import Llama as _Llama

        Llama = _Llama
        LLAMA_CPP_AVAILABLE = True
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("llama_cpp installation failed: %s", exc)
        return False


def gpu_offload_supported() -> bool:
    try:
        import llama_cpp.llama_cpp as lc

        return bool(lc.llama_supports_gpu_offload())
    except Exception:
        return False


def llama_system_info() -> str:
    try:
        import llama_cpp.llama_cpp as lc

        return str(lc.llama_print_system_info() or "").strip()
    except Exception:
        return ""


def summary_enabled() -> bool:
    if not ensure_summary_model_path():
        return False
    try:
        if not Path(SUMMARY_MODEL).exists():
            return False
    except Exception:
        return False
    return ensure_llama_cpp()


def summary_prompt() -> str:
    if SUMMARY_PROMPT:
        return SUMMARY_PROMPT
    return (
        "You are a concise summarizer for developer activity. "
        "Summarize the content into 1-3 sentences. "
        "Return plain text only."
    )


def get_summary_llm():
    global _SUMMARY_LLM
    if _SUMMARY_LLM is not None:
        return _SUMMARY_LLM
    if not summary_enabled():
        return None
    try:
        if Llama is None:
            return None
        logger.info("Loading local GGUF: %s", SUMMARY_MODEL)
        _SUMMARY_LLM = Llama(
            model_path=SUMMARY_MODEL,
            n_ctx=SUMMARY_CTX,
            n_threads=SUMMARY_THREADS,
            n_gpu_layers=SUMMARY_N_GPU_LAYERS,
            verbose=False,
        )
        try:
            close = getattr(_SUMMARY_LLM, "close", None)
            if callable(close):
                atexit.register(close)
        except Exception:
            pass
        return _SUMMARY_LLM
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load local GGUF: %s", exc)
        return None


def generate_summary(content: str) -> str:
    if not summary_enabled():
        return ""
    try:
        llm = get_summary_llm()
        if not llm:
            return ""
        prompt = summary_prompt()
        text = content
        if SUMMARY_MAX_CHARS and len(text) > SUMMARY_MAX_CHARS:
            text = text[:SUMMARY_MAX_CHARS]
        res = llm(
            f"{prompt}\n\nCONTENT:\n{text}\n\nSUMMARY:",
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=SUMMARY_TEMPERATURE,
            top_p=SUMMARY_TOP_P,
            repeat_penalty=SUMMARY_REPEAT_PENALTY,
        )
        out = (res.get("choices") or [{}])[0].get("text") or ""
        return out.strip()
    except Exception as exc:  # pragma: no cover
        logger.warning("Summary generation failed: %s", exc)
        return ""


def auto_summary(content: str, max_len: int = 240) -> str:
    summary = generate_summary(content)
    if summary:
        return summary[:max_len]
    first = content.strip().splitlines()[0] if content.strip() else ""
    if not first:
        return content[:max_len]
    if len(first) > max_len:
        return first[: max_len - 3] + "..."
    return first
