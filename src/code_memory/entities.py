from __future__ import annotations

import re
from typing import List

from .config import logger


try:
    from tree_sitter_languages import get_parser

    TREE_SITTER_AVAILABLE = True
    TREE_SITTER_PARSER = get_parser("python")
except Exception:  # pragma: no cover
    TREE_SITTER_AVAILABLE = False
    TREE_SITTER_PARSER = None


def extract_entities(content: str) -> List[dict]:
    entities: List[dict] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(", content):
        name = match.group(1)
        entities.append({"type": "function_like", "name": name, "source": "regex", "path": None})
    if TREE_SITTER_AVAILABLE and TREE_SITTER_PARSER is not None:
        try:
            tree = TREE_SITTER_PARSER.parse(bytes(content, "utf-8"))
            entities.extend(walk_tree_for_entities(tree, content))
        except Exception as exc:
            logger.debug("tree-sitter entity extraction failed: %s", exc)
    return entities


def walk_tree_for_entities(tree, source: str) -> List[dict]:
    names = []
    cursor = tree.walk()
    visited = set()
    while True:
        node = cursor.node
        if node.id not in visited:
            visited.add(node.id)
            if node.type in ("function_definition", "class_definition"):
                for child in node.children:
                    if child.type == "identifier":
                        name = source[child.start_byte : child.end_byte]
                        names.append(
                            {
                                "type": "function" if node.type == "function_definition" else "class",
                                "name": name,
                                "source": "tree-sitter",
                                "path": None,
                            }
                        )
                        break
        if cursor.goto_first_child():
            continue
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                return names

