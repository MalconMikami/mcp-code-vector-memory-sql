from __future__ import annotations

import atexit
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import (
    MODEL_CACHE_DIR,
    NER_AUTO_INSTALL,
    NER_CTX,
    NER_MAX_INPUT_CHARS,
    NER_MAX_TOKENS,
    NER_N_GPU_LAYERS,
    NER_PROMPT,
    NER_REPEAT_PENALTY,
    NER_TEMPERATURE,
    NER_THREADS,
    NER_TOP_P,
    get_setting,
    install_llama_cpp,
    logger,
)


LLAMA_CPP_AVAILABLE = False
Llama = None
_NER_LLM = None


def _is_debug_logging() -> bool:
    try:
        return str(get_setting("CODE_MEMORY_LOG_LEVEL", "INFO")).upper() == "DEBUG"
    except Exception:
        return False


def _tail(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    if max_chars > 0 and len(text) > max_chars:
        return "...(truncated)...\n" + text[-max_chars:]
    return text


def _strip_code_heavy_sections(text: str, *, max_chars: int) -> str:
    """
    Reduce code/diff payloads before sending to the local GGUF model.

    Goal: extract "world facts" (intentions, constraints, decisions, errors, commands, configs), not raw code.
    """
    s = text or ""
    if not s:
        return ""

    # Remove fenced blocks (```...```), including language-tagged fences.
    s = re.sub(r"```[\s\S]*?```", "\n[CODE_BLOCK_OMITTED]\n", s)

    # Remove large unified diffs (common in tool outputs).
    s = re.sub(r"(?ms)^(diff --git[\s\S]*?)(?=^diff --git|\\Z)", "\n[DIFF_OMITTED]\n", s)

    s = s.strip()
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars] + "\n...[TRUNCATED]..."
    return s


def _extract_first_json_object(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Best-effort JSON extraction from LLM output.
    Returns (obj, error_message). error_message == "" means success.
    """
    raw = (text or "").strip()
    if not raw:
        return {}, "empty output"

    m = re.search(r"```(?:json)?\\s*([\\s\\S]*?)\\s*```", raw, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {}, "json is not an object"
        except Exception as exc:
            return {}, f"failed to parse fenced json: {exc}"

    dec = json.JSONDecoder()
    idx = 0
    while True:
        start = raw.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = dec.raw_decode(raw[start:])
            if isinstance(obj, dict):
                return obj, ""
            idx = start + max(1, end)
        except Exception:
            idx = start + 1

    return {}, "no json object found"


def _ner_auto_download_enabled() -> bool:
    return os.getenv("CODE_MEMORY_NER_AUTO_DOWNLOAD", "1").lower() not in ("0", "false", "no")


DEFAULT_NER_REPO_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"


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
    if not _ner_auto_download_enabled():
        logger.info("ner.gguf_download: disabled (CODE_MEMORY_NER_AUTO_DOWNLOAD=0/false/no) repo=%s", repo_id)
        return ""
    logger.info("ner.gguf_download: start repo=%s cache_base=%s", repo_id, MODEL_CACHE_DIR)
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
    except Exception as exc:  # pragma: no cover
        logger.warning("ner.gguf_download: huggingface_hub import failed: %s", exc)
        return ""

    repo_id = normalize_repo_id(repo_id)
    if not repo_id:
        logger.warning("ner.gguf_download: invalid repo_id after normalize")
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
        logger.info("ner.gguf_download: repo_files=%s repo=%s", len(files), repo_id)
    except Exception as exc:
        logger.warning("ner.gguf_download: list_repo_files failed repo=%s err=%s", repo_id, exc)
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
        logger.warning("ner.gguf_download: no .gguf file found repo=%s", repo_id)
        return ""

    try:
        logger.info("ner.gguf_download: downloading repo=%s file=%s -> %s", repo_id, filename, cache_dir)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=str(cache_dir),
        )
        resolved = str(Path(path).resolve())
        logger.info("ner.gguf_download: done path=%s", resolved)
        return resolved
    except Exception:  # pragma: no cover
        logger.exception("ner.gguf_download: failed repo=%s file=%s", repo_id, filename)
        return ""


NER_MODEL = os.getenv("CODE_MEMORY_NER_MODEL", "").strip()


def resolve_ner_model_path() -> str:
    configured = str(get_setting("CODE_MEMORY_NER_MODEL", "") or "").strip()
    if _is_debug_logging():
        logger.debug(
            "ner.model_resolve: configured=%r auto_download=%s cache_base=%s",
            configured,
            _ner_auto_download_enabled(),
            MODEL_CACHE_DIR,
        )
    if configured:
        if Path(configured).exists():
            logger.info("ner.model_resolve: using local path=%s", configured)
            return configured
        if looks_like_hf_repo_id(configured):
            if not _ner_auto_download_enabled():
                logger.info("ner.model_resolve: auto_download disabled; repo_id=%s", configured)
                return ""
            return download_gguf_from_repo(configured)
        logger.warning("ner.model_resolve: value is not a file and not a HF repo id: %s", configured)
        return ""
    logger.info("ner.model_resolve: no model configured (CODE_MEMORY_NER_MODEL empty)")
    return ""


def ensure_ner_model_path() -> str:
    global NER_MODEL
    if NER_MODEL:
        try:
            if Path(NER_MODEL).exists():
                logger.info("ner.model_ensure: cached NER_MODEL exists path=%s", NER_MODEL)
                return NER_MODEL
        except Exception:
            pass
        if looks_like_hf_repo_id(NER_MODEL):
            resolved = download_gguf_from_repo(NER_MODEL)
            if resolved:
                NER_MODEL = resolved
                logger.info("ner.model_ensure: downloaded -> %s", NER_MODEL)
            return NER_MODEL
        logger.warning("ner.model_ensure: NER_MODEL set but not found on disk: %s", NER_MODEL)
        return NER_MODEL

    resolved = resolve_ner_model_path()
    if resolved:
        NER_MODEL = resolved
        logger.info("ner.model_ensure: resolved NER_MODEL=%s", NER_MODEL)
        return NER_MODEL
    return ""


def ner_enabled() -> bool:
    model = ensure_ner_model_path()
    if not model:
        return False
    try:
        if not Path(model).exists():
            logger.warning("ner.enabled: model path does not exist: %s", model)
            return False
    except Exception:
        return False
    return True


def _load_llama_cpp() -> None:
    global LLAMA_CPP_AVAILABLE, Llama
    if LLAMA_CPP_AVAILABLE:
        return
    try:
        from llama_cpp import Llama as _Llama

        Llama = _Llama
        LLAMA_CPP_AVAILABLE = True
        logger.info("ner.llama_cpp: import ok")
        return
    except Exception as exc:
        logger.warning("ner.llama_cpp: import failed: %s", exc)
        if not NER_AUTO_INSTALL:
            logger.info("ner.llama_cpp: auto_install disabled (CODE_MEMORY_AUTO_INSTALL=false)")
            return
        try:
            logger.info("ner.llama_cpp: attempting auto-install llama-cpp-python")
            install_llama_cpp()
            from llama_cpp import Llama as _Llama

            Llama = _Llama
            LLAMA_CPP_AVAILABLE = True
            logger.info("ner.llama_cpp: import ok after install")
        except Exception:
            logger.exception("ner.llama_cpp: installation/import failed")


def get_ner_llm():
    global _NER_LLM
    if _NER_LLM is not None:
        return _NER_LLM
    if not ner_enabled():
        if _is_debug_logging():
            logger.debug("ner.llm: disabled (ner_enabled=false)")
        _NER_LLM = None
        return None

    _load_llama_cpp()
    if not LLAMA_CPP_AVAILABLE or Llama is None:
        logger.warning("ner.llm: llama-cpp-python unavailable")
        _NER_LLM = None
        return None

    model = ensure_ner_model_path()
    if not model:
        _NER_LLM = None
        return None

    try:
        logger.info(
            "ner.llm: loading gguf=%s ctx=%s threads=%s gpu_layers=%s",
            model,
            NER_CTX,
            NER_THREADS,
            NER_N_GPU_LAYERS,
        )
        _NER_LLM = Llama(
            model_path=model,
            n_ctx=NER_CTX,
            n_threads=NER_THREADS,
            n_gpu_layers=NER_N_GPU_LAYERS,
            verbose=_is_debug_logging(),
        )

        def _cleanup():
            global _NER_LLM
            try:
                close = getattr(_NER_LLM, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            _NER_LLM = None

        atexit.register(_cleanup)
        return _NER_LLM
    except Exception:
        logger.exception("ner.llm: failed to load gguf=%s", model)
        _NER_LLM = None
        return None


def extract_world_entities(content: str) -> dict:
    """
    Use a local GGUF (if available) to extract "real-world" entities/relations from text.

    Returns:
    {
      "entities": [{"type": "...", "name": "...", "evidence": "...", "confidence": 0.0..1.0}],
      "relations": [{"source": "...", "target": "...", "type": "...", "evidence": "..."}]
    }
    """
    llm = get_ner_llm()
    if not llm:
        return {"entities": [], "relations": []}

    text = _strip_code_heavy_sections(content, max_chars=int(NER_MAX_INPUT_CHARS) if NER_MAX_INPUT_CHARS else 25000)
    if not text:
        return {"entities": [], "relations": []}

    prompt = NER_PROMPT.strip() if NER_PROMPT else ""
    if not prompt:
        prompt = (
            "You are an entity and relation extractor for software-engineering memory.\n"
            "Extract high-signal real-world/project facts: intentions, decisions, constraints, technologies, services, "
            "modules/components, environments, configs, commands, errors, URLs, tickets, file paths.\n"
            "\n"
            "The input may contain code blocks mixed with natural language.\n"
            "Only include code identifiers when they are clearly discussed as real concepts (module/component/public API/error).\n"
            "Do NOT emit low-value symbols (local variables, every identifier you see).\n"
            "\n"
            "Output requirements:\n"
            "- Return ONLY a single valid JSON object. No markdown. No explanations.\n"
            "- Evidence must be a short quote from the original text.\n"
            "- Remove duplicates and normalize names.\n"
            "- Limit: max 30 entities, max 20 relations.\n"
            "\n"
            "Allowed entity types (entities[].type):\n"
            "- technology, service, component, environment, file_path, url, command, env_var, config_key, ticket, error, data_store, api, code_symbol, other\n"
            "\n"
            "Allowed relation types (relations[].type):\n"
            "- uses, depends_on, configured_with, caused_by, located_in, related_to, fixed_by, triggered_by, other\n"
            "\n"
            "Schema:\n"
            "{\n"
            '  "entities": [{"type":"technology","name":"PostgreSQL","evidence":"...","confidence":0.0}],\n'
            '  "relations": [{"source":"SQLAlchemy","target":"PostgreSQL","type":"uses","evidence":"..."}]\n'
            "}\n"
            "\n"
            f"Input: {text}\n"
            "Output:"
        )

    try:
        res = llm(
            prompt,
            max_tokens=int(NER_MAX_TOKENS) if NER_MAX_TOKENS else 800,
            temperature=float(NER_TEMPERATURE) if NER_TEMPERATURE is not None else 0.1,
            top_p=float(NER_TOP_P) if NER_TOP_P is not None else 0.9,
            repeat_penalty=float(NER_REPEAT_PENALTY) if NER_REPEAT_PENALTY is not None else 1.05,
        )
        out = (res.get("choices") or [{}])[0].get("text") or ""
        obj, err = _extract_first_json_object(out)
        if err:
            logger.warning("ner.extract: invalid json: %s", err)
            if _is_debug_logging():
                logger.debug("ner.extract: raw output tail:\n%s", _tail(out))
            return {"entities": [], "relations": []}

        entities = obj.get("entities") if isinstance(obj.get("entities"), list) else []
        relations = obj.get("relations") if isinstance(obj.get("relations"), list) else []

        cleaned_entities: List[dict] = []
        for e in entities[:30]:
            if not isinstance(e, dict):
                continue
            t = str(e.get("type") or "").strip()
            name = str(e.get("name") or "").strip()
            if not t or not name:
                continue
            ev = str(e.get("evidence") or "").strip()
            try:
                conf = float(e.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            cleaned_entities.append(
                {"type": t, "name": name, "evidence": ev[:240], "confidence": max(0.0, min(1.0, conf)), "source": "llm"}
            )

        cleaned_relations: List[dict] = []
        for r in relations[:20]:
            if not isinstance(r, dict):
                continue
            s = str(r.get("source") or "").strip()
            t = str(r.get("target") or "").strip()
            rt = str(r.get("type") or "").strip()
            if not s or not t or not rt:
                continue
            ev = str(r.get("evidence") or "").strip()
            cleaned_relations.append({"source": s, "target": t, "type": rt, "evidence": ev[:240], "origin": "llm"})

        return {"entities": cleaned_entities, "relations": cleaned_relations}
    except Exception:
        logger.exception("ner.extract: failed")
        return {"entities": [], "relations": []}

