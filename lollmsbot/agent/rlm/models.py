"""
RLM Data Models - Type definitions for the Recursive Language Model memory system.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class MemoryChunkType(Enum):
    """Types of memory chunks."""
    SELF_KNOWLEDGE = auto()      # Facts about the agent itself
    CONVERSATION = auto()        # Conversation history
    FACT = auto()                # General factual knowledge
    PROCEDURAL = auto()          # How-to knowledge, skills
    EPISODIC = auto()            # Specific events/experiences
    SKILL_EXECUTION = auto()     # Record of skill usage
    USER_PREFERENCE = auto()     # User-specific preferences
    WORKING_MEMORY = auto()      # Temporary working state


@dataclass
class MemoryChunk:
    """
    A compressed, indexed unit of memory in the EMS (External Memory Store).
    
    Chunks are the atomic units of long-term storage - they can be loaded
    into the RCB on demand via [[MEMORY:chunk_id|metadata]] handles.
    """
    chunk_id: str
    chunk_type: MemoryChunkType
    content_compressed: bytes
    content_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    memory_importance: float = 1.0  # 1.0 = normal, higher = more important
    compression_ratio: float = 1.0  # Original/compressed size ratio
    tags: List[str] = field(default_factory=list)
    summary: str = ""  # Human-readable summary for LLM to see in REPL
    load_hints: List[str] = field(default_factory=list)  # Keywords for retrieval
    source: Optional[str] = None  # Where this memory came from
    archived: bool = False  # True if forgotten/archived
    
    @classmethod
    def create(
        cls,
        chunk_id: str,
        chunk_type: MemoryChunkType,
        content: str,
        memory_importance: float = 1.0,
        tags: Optional[List[str]] = None,
        summary: Optional[str] = None,
        load_hints: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> "MemoryChunk":
        """Factory method to create a chunk with automatic compression."""
        # Ensure content is never None or empty
        if content is None:
            content = ""
        
        # Compress content - use UTF-8 encoding which handles all characters
        try:
            original_bytes = content.encode('utf-8')
        except UnicodeEncodeError:
            # Fallback if UTF-8 fails (shouldn't happen but be safe)
            original_bytes = content.encode('utf-8', errors='replace')
        
        # Ensure we have something to compress
        if len(original_bytes) == 0:
            # Empty content - store a minimal valid compressed blob
            original_bytes = b"[empty]"
        
        # Compress with zlib
        try:
            compressed = zlib.compress(original_bytes, level=6)
        except Exception:
            # If compression fails, use uncompressed but still wrap it
            compressed = original_bytes
        
        # Ensure compressed is never empty
        if not compressed or len(compressed) == 0:
            compressed = b"[compression_failed]"
        
        # Calculate hash and compression ratio
        import hashlib
        content_hash = hashlib.sha256(original_bytes).hexdigest()[:16]
        compression_ratio = len(original_bytes) / len(compressed) if compressed and len(compressed) > 0 else 1.0
        
        return cls(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            content_compressed=compressed,
            content_hash=content_hash,
            memory_importance=memory_importance,
            compression_ratio=compression_ratio,
            tags=tags or [],
            summary=summary or "",
            load_hints=load_hints or [],
            source=source,
        )
    
    def decompress_content(self) -> str:
        """Decompress and return original content."""
        if not self.content_compressed:
            return "[empty content]"
        try:
            decompressed = zlib.decompress(self.content_compressed)
            return decompressed.decode('utf-8')
        except Exception:
            # If decompression fails, try to decode directly (might be uncompressed)
            try:
                return self.content_compressed.decode('utf-8')
            except Exception:
                return "[decompression failed]"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding compressed content)."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.name,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "memory_importance": self.memory_importance,
            "compression_ratio": self.compression_ratio,
            "tags": self.tags,
            "summary": self.summary,
            "load_hints": self.load_hints,
            "source": self.source,
            "archived": self.archived,
        }


@dataclass
class RCBEntry:
    """
    An entry in the REPL Context Buffer (RCB).
    
    The RCB is the working memory visible to the LLM. It contains:
    - Loaded memory chunks (with [[MEMORY:...]] handles)
    - Working state
    - System prompts and tool documentation
    """
    entry_id: int = 0  # Database ID
    entry_type: str = ""  # 'memory_handle', 'system_prompt', 'working_state', 'tool_doc'
    content: str = ""  # The actual text content
    chunk_id: Optional[str] = None  # Reference to EMS chunk if applicable
    display_order: int = 0  # Order in the REPL display
    loaded_at: datetime = field(default_factory=datetime.now)
    
    def format_for_prompt(self, max_length: int = 500) -> str:
        """Format this entry for inclusion in LLM prompt."""
        if self.entry_type == "memory_handle" and self.chunk_id:
            # Format as loadable memory handle
            # Extract summary from content or use placeholder
            summary = self.content[:100].replace('\n', ' ')
            if len(self.content) > 100:
                summary += "..."
            
            return f"[[MEMORY:{self.chunk_id}|{{\"type\": \"{self.entry_type}\", \"summary\": \"{summary}\"}}]]"
        
        return self.content


@dataclass
class CompressionStats:
    """Statistics about memory compression efficiency."""
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0
    chunk_count: int = 0
    
    @property
    def space_saved_bytes(self) -> int:
        return self.original_size_bytes - self.compressed_size_bytes
    
    @property
    def space_saved_percent(self) -> float:
        if self.original_size_bytes == 0:
            return 0.0
        return (self.space_saved_bytes / self.original_size_bytes) * 100
