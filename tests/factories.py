from dataclasses import dataclass, field
from typing import Dict, List
import random


@dataclass
class MemoryInput:
    content: str
    session_id: str
    kind: str
    tags: str
    priority: int
    metadata: Dict = field(default_factory=dict)


def make_memory(
    index: int,
    session_id: str = "S1",
    keyword: str | None = None,
    priority: int = 3,
) -> MemoryInput:
    token = keyword or f"token_{index}"
    content = f"Memory {index} about {token} in module_{index}.py"
    tags = "test"
    return MemoryInput(
        content=content,
        session_id=session_id,
        kind="note",
        tags=tags,
        priority=priority,
        metadata={"index": index},
    )


def build_memories(count: int = 10, session_id: str = "S1", seed: int = 123) -> List[MemoryInput]:
    rng = random.Random(seed)
    memories: List[MemoryInput] = []
    for i in range(count):
        memories.append(make_memory(i, session_id=session_id, priority=rng.choice([1, 2, 3, 4, 5])))
    return memories


def build_multi_session() -> List[MemoryInput]:
    return build_memories(5, session_id="S1", seed=10) + build_memories(5, session_id="S2", seed=20)
