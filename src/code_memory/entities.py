from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from .config import logger
from .ner import extract_world_entities


TREE_SITTER_AVAILABLE = False  # deprecated: we no longer parse code for memory NER
TREE_SITTER_PARSER = None


def _dedupe(entities: Iterable[dict]) -> List[dict]:
    seen: set[Tuple[str, str]] = set()
    out: List[dict] = []
    for e in entities:
        key = (str(e.get("type") or ""), str(e.get("name") or ""))
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def extract_entities(
    content: str, *, path: Optional[str] = None, max_entities: int = 50, use_llm: bool = True
) -> List[dict]:
    """
    NER focused on "world facts" (intentions, decisions, technologies, services, errors, commands, files).

    We intentionally do not parse code (no tree-sitter) because code can be read directly from the workspace.
    """
    entities: List[dict] = []

    if use_llm:
        # Always attempt LLM-based NER first (local GGUF). Falls back gracefully on failure.
        try:
            ner = extract_world_entities(content)
            for e in ner.get("entities", []) or []:
                if not isinstance(e, dict):
                    continue
                entities.append(
                    {
                        "type": e.get("type"),
                        "name": e.get("name"),
                        "source": e.get("source", "llm"),
                        "path": path,
                        "observation": e.get("evidence") or "",
                        "confidence": e.get("confidence", 0.0),
                    }
                )
        except Exception as exc:
            logger.debug("llm NER extraction failed: %s", exc)

    # Lightweight regex fallback for key "world" tokens.
    entities.extend(_extract_world_signals_regex(content, path=path))

    entities = _dedupe(entities)
    if max_entities and len(entities) > max_entities:
        entities = entities[:max_entities]
    return entities


def extract_graph(
    content: str, *, path: Optional[str] = None, max_entities: int = 50, max_relations: int = 50, use_llm: bool = True
) -> dict:
    """
    Extract a lightweight world-facts graph from text.

    Returns:
      {"entities": [...], "relations": [...]}
    """
    entities = extract_entities(content, path=path, max_entities=max_entities, use_llm=use_llm)
    relations: List[dict] = []

    if use_llm:
        try:
            ner = extract_world_entities(content)
            for r in (ner.get("relations") or [])[:max_relations]:
                if not isinstance(r, dict):
                    continue
                src = str(r.get("source") or "").strip()
                tgt = str(r.get("target") or "").strip()
                rel_type = str(r.get("type") or "").strip()
                if not src or not tgt or not rel_type:
                    continue
                relations.append(
                    {
                        "source": src,
                        "target": tgt,
                        "type": rel_type,
                        "evidence": str(r.get("evidence") or "").strip()[:240],
                        "origin": str(r.get("origin") or r.get("source") or "llm"),
                    }
                )
        except Exception as exc:
            logger.debug("llm relation extraction failed: %s", exc)

    # Dedupe relations
    seen_rel: set[Tuple[str, str, str]] = set()
    out_rel: List[dict] = []
    for r in relations:
        key = (r.get("source") or "", r.get("target") or "", r.get("type") or "")
        if not key[0] or not key[1] or not key[2]:
            continue
        if key in seen_rel:
            continue
        seen_rel.add(key)
        out_rel.append(r)

    return {"entities": entities, "relations": out_rel[:max_relations]}


def _snippet(line: str, *, max_len: int = 240) -> str:
    s = (line or "").strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _extract_world_signals_regex(content: str, *, path: Optional[str]) -> List[dict]:
    out: List[dict] = []
    text = content or ""
    lines = text.splitlines()

    # URLs
    for m in re.finditer(r"https?://[^\s)\"']+", text):
        url = m.group(0)
        out.append({"type": "url", "name": url, "source": "regex", "path": path, "observation": url})

    # Windows paths + Unix paths (best-effort)
    for m in re.finditer(r"\b[A-Za-z]:\\[^\s\"']+", text):
        p = m.group(0)
        out.append({"type": "file_path", "name": p, "source": "regex", "path": path, "observation": p})
    for m in re.finditer(r"(?:(?<=\s)|^)/(?:[^\s\"']+)", text):
        p = m.group(0)
        out.append({"type": "file_path", "name": p, "source": "regex", "path": path, "observation": p})

    # Tickets/IDs (GitHub issues, Jira-like)
    for m in re.finditer(r"\b[A-Z][A-Z0-9]+-\d+\b", text):
        out.append({"type": "ticket", "name": m.group(0), "source": "regex", "path": path, "observation": m.group(0)})
    for m in re.finditer(r"\b#\d{2,}\b", text):
        out.append({"type": "ticket", "name": m.group(0), "source": "regex", "path": path, "observation": m.group(0)})

    # Environment variables (prefer obvious patterns)
    for m in re.finditer(r"\b[A-Z][A-Z0-9_]{2,}\b", text):
        name = m.group(0)
        if name.startswith(("CODE_MEMORY_", "OPENCODE_", "OPENAI_", "GITHUB_")):
            out.append({"type": "env_var", "name": name, "source": "regex", "path": path, "observation": name})

    # Commands (lines that look like CLI)
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if l.startswith(("$ ", "> ", "PS> ")):
            out.append({"type": "command", "name": l[:120], "source": "regex", "path": path, "observation": _snippet(l)})
            continue
        if re.match(r"^(pip|python|npm|npx|pnpm|yarn|git|docker|kubectl)\b", l):
            out.append({"type": "command", "name": l[:120], "source": "regex", "path": path, "observation": _snippet(l)})
            continue

    # Errors (very light)
    for line in lines:
        if "Traceback (most recent call last)" in line:
            out.append({"type": "error", "name": "Traceback", "source": "regex", "path": path, "observation": _snippet(line)})
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*(Error|Exception))\b", line)
        if m:
            out.append({"type": "error", "name": m.group(1), "source": "regex", "path": path, "observation": _snippet(line)})

    return out


def walk_tree_for_entities(tree, source: str, *, path: Optional[str] = None) -> List[dict]:
    # Deprecated no-op retained for compatibility with older imports.
    return []
