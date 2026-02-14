"""
RLM (Recursive Language Model) Memory Package

Implements the double-memory architecture from MIT CSAIL's RLM research:
- External Memory Store (EMS): SQLite-backed compressed storage
- REPL Context Buffer (RCB): Working memory with loadable handles
"""

from lollmsbot.agent.rlm.database import RLMDatabase
from lollmsbot.agent.rlm.manager import RLMMemoryManager
from lollmsbot.agent.rlm.models import (
    MemoryChunk,
    MemoryChunkType,
    RCBEntry,
    CompressionStats,
)
from lollmsbot.agent.rlm.sanitizer import PromptInjectionSanitizer
from lollmsbot.agent.rlm.memory_map import MemoryMap, MemoryCategorySummary, MemoryAnchor, KnowledgeGap

__all__ = [
    "RLMDatabase",
    "RLMMemoryManager",
    "MemoryChunk",
    "MemoryChunkType",
    "RCBEntry",
    "CompressionStats",
    "PromptInjectionSanitizer",
    "MemoryMap",
    "MemoryCategorySummary",
    "MemoryAnchor",
    "KnowledgeGap",
]
