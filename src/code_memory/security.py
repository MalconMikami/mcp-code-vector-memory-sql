import hashlib
import re


SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*[\\\"']?[A-Za-z0-9\\-_/]{16,})"),
    re.compile(r"(?i)(secret\s*[:=]\s*[\\\"']?[A-Za-z0-9\\-_/]{12,})"),
    re.compile(r"(?i)(password\s*[:=])"),
]


def looks_sensitive(text: str) -> bool:
    return any(p.search(text) for p in SENSITIVE_PATTERNS)


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

