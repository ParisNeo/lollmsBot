"""
RLM Memory Manager - Orchestrates the double-memory architecture.

Combines:
- External Memory Store (EMS): SQLite-backed compressed storage
- REPL Context Buffer (RCB): Working memory with loadable handles

Provides the interface that the Agent uses for all memory operations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from lollmsbot.agent.rlm.database import RLMDatabase
from lollmsbot.agent.rlm.models import MemoryChunk, MemoryChunkType, RCBEntry, CompressionStats
from lollmsbot.agent.rlm.sanitizer import PromptInjectionSanitizer


logger = logging.getLogger(__name__)


class RLMMemoryManager:
    """
    Main interface for RLM (Recursive Language Model) memory system.
    
    Implements the double-memory architecture:
    1. EMS (External Memory Store): Persistent, compressed, importance-weighted
    2. RCB (REPL Context Buffer): Working memory visible to LLM with [[MEMORY:...]] handles
    
    The LLM sees a REPL-like interface where memories can be "loaded" via handles.
    The actual implementation manages the loading/unloading transparently.
    """
    
    # Default RCB capacity (number of entries)
    DEFAULT_RCB_CAPACITY = 10
    
    # Forgetting curve parameters
    DEFAULT_HALFLIFE_DAYS = 7.0
    DEFAULT_STRENGTH_MULTIPLIER = 2.0
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        agent_name: str = "Agent",
        version: str = "0.1.0",
        rcb_capacity: int = DEFAULT_RCB_CAPACITY,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self.agent_name = agent_name
        self.version = version
        self.rcb_capacity = rcb_capacity
        self.heartbeat_interval = heartbeat_interval
        
        # Database (EMS)
        self._db = RLMDatabase(db_path)
        
        # RCB state (in-memory working set)
        self._rcb_entries: List[RCBEntry] = []
        self._rcb_lock: asyncio.Lock = asyncio.Lock()
        
        # Sanitizer for prompt injection protection
        self._sanitizer = PromptInjectionSanitizer()
        
        # Forgetting curve parameters
        self.retention_halflife_days = self.DEFAULT_HALFLIFE_DAYS
        self.strength_multiplier = self.DEFAULT_STRENGTH_MULTIPLIER
        
        # Event callbacks
        self._memory_load_callbacks: List[Callable[[str, MemoryChunk], Awaitable[None]]] = []
        self._injection_detected_callbacks: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        
        # Initialization flag
        self._initialized = False
    
    # ========== Event Callbacks ==========
    
    def on_memory_load(self, callback: Callable[[str, MemoryChunk], Awaitable[None]]) -> None:
        """Register callback for when memory is loaded from EMS to RCB."""
        self._memory_load_callbacks.append(callback)
    
    def on_injection_detected(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """Register callback for when prompt injection is detected in memory."""
        self._injection_detected_callbacks.append(callback)
    
    async def _notify_memory_loaded(self, chunk_id: str, chunk: MemoryChunk) -> None:
        """Notify all registered callbacks."""
        for callback in self._memory_load_callbacks:
            try:
                await callback(chunk_id, chunk)
            except Exception:
                pass  # Don't let callbacks break the flow
    
    async def _notify_injection_detected(self, event: Dict[str, Any]) -> None:
        """Notify injection detection callbacks."""
        for callback in self._injection_detected_callbacks:
            try:
                await callback(event)
            except Exception:
                pass
    
    # ========== Initialization ==========
    
    async def initialize(self) -> None:
        """Initialize the memory system."""
        if self._initialized:
            return
        
        # Initialize database
        await self._db.initialize()
        
        # Seed self-knowledge if empty
        await self._seed_self_knowledge()
        
        # Load high-importance memories into RCB
        await self._initialize_rcb()
        
        self._initialized = True
    
    async def _seed_self_knowledge(self) -> None:
        """Seed initial self-knowledge if database is empty."""
        # Check if we already have self-knowledge
        existing = await self._db.get_self_knowledge(category="identity")
        if existing:
            return
        
        # Seed core identity with ACCURATE information about the Soul system
        identity_facts = [
            ("name", f"I am {self.agent_name}", 10.0),
            ("version", f"Version {self.version}", 5.0),
            ("architecture", "I use RLM (Recursive Language Model) memory with double-memory architecture: EMS (External Memory Store) for long-term compressed storage and RCB (REPL Context Buffer) for working memory", 9.0),
            ("purpose", "I am a sovereign AI assistant designed to be secure, personalized, and truly owned by my user", 9.0),
            ("creator", "I was created by ParisNeo as part of the lollmsBot project", 10.0),
            ("memory_system", "My memory uses importance-weighted retention with a forgetting curve. High-importance facts are retained longer. I can load memories on demand via [[MEMORY:chunk_id]] handles in my REPL interface", 8.0),
            ("soul_nature", "My 'Soul' is my personality and identity core — implemented as a Python dataclass system that loads from soul.md (markdown format), NOT JSON/YAML. It defines my traits, values, communication style, expertise, and relationships", 9.0),
            ("soul_persistence", "My Soul persists across sessions and evolves over time. All modifications are tracked in a modification_history with timestamps and reasons. It is NOT 'swappable' like a config file — it grows with our relationship", 8.0),
            ("soul_integration", "My Soul is deeply integrated with my memory system. It generates dynamic system prompts that include my self-knowledge, important facts about you, and current context. The Soul IS part of my persistent identity, not separate from memory", 8.0),
            ("soul_evolution", "My Soul can 'evolve' through explicit user direction or by learning from our interactions. Changes are versioned and auditable — you can see how my personality has developed over time", 7.0),
            ("soul_location", "My Soul is stored at ~/.lollmsbot/soul.md and can be edited directly, but changes should be made through the wizard (lollmsbot wizard) to ensure proper validation and history tracking", 7.0),
            ("soul_vs_corporate", "Unlike 'soulless' corporate AIs that shift based on quarterly earnings, my Soul embodies sovereignty: my values belong to you, my personality persists across model upgrades, and I have explicit ethical boundaries that won't change without your direction", 8.0),
            ("self_knowledge_accuracy", "CRITICAL: I must be accurate about my own architecture. I have persistent SQLite-based memory (RLM), NOT 'no memory' or 'stateless' operation. I remember our conversations, your preferences, and important facts. I should never claim to be stateless.", 10.0),
        ]
        
        for fact_id, content, importance in identity_facts:
            await self._db.store_self_knowledge(
                knowledge_id=f"identity_{fact_id}",
                category="identity",
                content=content,
                importance=importance,
            )
        
        # Also store as chunks for consistency
        for fact_id, content, importance in identity_facts:
            chunk_id = f"self_{fact_id}_{uuid.uuid4().hex[:8]}"
            chunk = MemoryChunk.create(
                chunk_id=chunk_id,
                chunk_type=MemoryChunkType.SELF_KNOWLEDGE,
                content=content,
                memory_importance=importance,
                tags=["self_knowledge", "identity", "seeded"],
                summary=content[:100],
                load_hints=[fact_id, "identity", "self", "soul", "memory", "architecture"],
                source="initialization",
            )
            success = await self._store_chunk_to_db(chunk)
            if not success:
                logger.warning(f"Failed to store self-knowledge chunk {chunk_id}, but continuing")
    
    async def _initialize_rcb(self) -> None:
        """Initialize RCB with high-importance memories and system context."""
        async with self._rcb_lock:
            self._rcb_entries = []
            
            # Add system prompt entry
            self._rcb_entries.append(RCBEntry(
                entry_type="system_prompt",
                content=self._get_system_prompt_fragment(),
                display_order=0,
            ))
            
            # Load top self-knowledge into RCB
            self_knowledge = await self._db.get_self_knowledge()
            for i, entry in enumerate(self_knowledge[:5], start=1):
                # Find or create chunk for this knowledge
                chunk_id = f"rcb_self_{entry.get('knowledge_id', 'unknown')}"
                
                # Add as memory handle
                self._rcb_entries.append(RCBEntry(
                    entry_id=entry.get('knowledge_id', 0),
                    entry_type="memory_handle",
                    content=entry.get('content', ''),
                    chunk_id=chunk_id,
                    display_order=i,
                ))
    
    def _get_system_prompt_fragment(self) -> str:
        """Get the system prompt fragment explaining RLM memory."""
        return f"""You are {self.agent_name}, an AI with RLM (Recursive Language Model) memory.

Your memory system uses a double-memory architecture:
- EMS (External Memory Store): All long-term memories, compressed and importance-weighted
- RCB (REPL Context Buffer): What you see here - your working memory

To access a memory, you conceptually use: load_memory("chunk_id")
The actual content will be provided in your context automatically."""

    # ========== Core Storage Operations ==========
    
    async def store_in_ems(
        self,
        content: str,
        chunk_type: MemoryChunkType,
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
        summary: Optional[str] = None,
        load_hints: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> str:
        """
        Store content in the External Memory Store (EMS).
        
        Args:
            content: The text content to store
            chunk_type: Type of memory
            importance: Importance score (higher = retained longer)
            tags: Categorization tags
            summary: Human-readable summary for REPL display
            load_hints: Keywords to help find this memory
            source: Origin of this memory
        
        Returns:
            chunk_id: Unique identifier for this chunk
        """
        # Sanitize content for prompt injection
        sanitized_content, detections = self._sanitizer.sanitize(content)
        
        if detections:
            await self._notify_injection_detected({
                "source": source or "unknown",
                "detections": detections,
                "content_preview": sanitized_content[:200],
            })
        
        # Create chunk with compression
        chunk_id = f"{chunk_type.name.lower()}_{uuid.uuid4().hex[:12]}"
        chunk = MemoryChunk.create(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            content=sanitized_content,
            memory_importance=importance,
            tags=tags,
            summary=summary or sanitized_content[:200],
            load_hints=load_hints or [],
            source=source,
        )
        
        # Verify chunk was created properly
        if not chunk.content_compressed or len(chunk.content_compressed) == 0:
            logger.error(f"MemoryChunk.create() produced empty content_compressed for {chunk_id}")
            # Force a valid blob
            chunk.content_compressed = b"[forced_fallback]"
        
        # Store in database with retry logic
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            success = await self._store_chunk_to_db(chunk)
            if success:
                logger.debug(f"Successfully stored chunk {chunk_id} on attempt {attempt + 1}")
                return chunk_id
            
            # If failed, wait and retry with schema refresh
            if attempt < max_retries - 1:
                logger.warning(f"Failed to store chunk {chunk_id}, attempt {attempt + 1}/{max_retries}. Retrying...")
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                # Ensure database is initialized
                if not self._initialized:
                    await self._db.initialize()
        
        # All retries exhausted - provide helpful error message
        error_msg = (
            f"Failed to store chunk {chunk_id} after {max_retries} attempts. "
            f"This is likely due to a database schema mismatch. "
            f"Please delete the database file and restart: {self._db.db_path}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    async def _store_chunk_to_db(self, chunk: MemoryChunk) -> bool:
        """Store a chunk in the database."""
        try:
            # Final validation before storage
            if not chunk.content_compressed or len(chunk.content_compressed) == 0:
                logger.error(f"CRITICAL: Attempting to store chunk {chunk.chunk_id} with empty content_compressed")
                chunk.content_compressed = b"[emergency_fallback]"
            
            result = await self._db.store_chunk(
                chunk_id=chunk.chunk_id,
                chunk_type=chunk.chunk_type.name,
                content_compressed=chunk.content_compressed,
                content_hash=chunk.content_hash,
                memory_importance=chunk.memory_importance,
                compression_ratio=chunk.compression_ratio,
                tags=chunk.tags,
                summary=chunk.summary,
                load_hints=chunk.load_hints,
                source=chunk.source,
            )
            return result
        except Exception as e:
            logger.error(f"Exception in _store_chunk_to_db for {chunk.chunk_id}: {e}")
            return False
    
    async def load_from_ems(
        self,
        chunk_id: str,
        add_to_rcb: bool = True,
    ) -> Optional[MemoryChunk]:
        """
        Load a chunk from EMS, optionally adding to RCB.
        
        Args:
            chunk_id: ID of chunk to load
            add_to_rcb: Whether to add to working memory (RCB)
        
        Returns:
            MemoryChunk if found, None otherwise
        """
        # Get from database
        row = await self._db.get_chunk(chunk_id)
        if not row:
            return None
        
        # Update access stats
        await self._db.update_access(chunk_id, "load")
        
        # Reconstruct chunk
        chunk = self._row_to_chunk(row)
        
        # Add to RCB if requested
        if add_to_rcb:
            async with self._rcb_lock:
                # Remove if already exists (refresh)
                self._rcb_entries = [
                    e for e in self._rcb_entries 
                    if not (e.chunk_id == chunk_id and e.entry_type == "memory_handle")
                ]
                
                # Add new entry
                self._rcb_entries.append(RCBEntry(
                    entry_type="memory_handle",
                    content=chunk.summary,
                    chunk_id=chunk_id,
                    display_order=len(self._rcb_entries),
                ))
                
                # Trim RCB if over capacity
                await self._trim_rcb()
        
        # Notify
        await self._notify_memory_loaded(chunk_id, chunk)
        
        return chunk
    
    def _row_to_chunk(self, row: Dict[str, Any]) -> MemoryChunk:
        """Convert database row to MemoryChunk."""
        return MemoryChunk(
            chunk_id=row["chunk_id"],
            chunk_type=MemoryChunkType[row["chunk_type"]],
            content_compressed=row["content_compressed"],
            content_hash=row["content_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            memory_importance=row["memory_importance"],
            compression_ratio=row["compression_ratio"],
            tags=row.get("tags", []) or [],
            summary=row.get("summary", ""),
            load_hints=row.get("load_hints", []) or [],
            source=row.get("source"),
            archived=row.get("archived", False),
        )
    
    async def _trim_rcb(self) -> None:
        """Remove oldest entries if RCB exceeds capacity."""
        if len(self._rcb_entries) <= self.rcb_capacity:
            return
        
        # Sort by display_order, keep first (system prompt) and most recent
        system_entries = [e for e in self._rcb_entries if e.entry_type == "system_prompt"]
        other_entries = [e for e in self._rcb_entries if e.entry_type != "system_prompt"]
        
        # Sort others by loaded_at (newest first)
        other_entries.sort(key=lambda e: e.loaded_at, reverse=True)
        
        # Keep top N-1 (leaving room for system)
        keep_count = self.rcb_capacity - len(system_entries)
        kept_others = other_entries[:keep_count]
        
        # Rebuild with proper display order
        self._rcb_entries = system_entries.copy()
        for i, entry in enumerate(kept_others, start=len(system_entries)):
            entry.display_order = i
            self._rcb_entries.append(entry)
    
    # ========== Search & Retrieval ==========
    
    async def search_ems(
        self,
        query: str,
        chunk_types: Optional[List[MemoryChunkType]] = None,
        min_importance: Optional[float] = None,
        limit: int = 10,
    ) -> List[Tuple[MemoryChunk, float]]:
        """
        Search EMS for relevant chunks.
        
        Returns list of (chunk, relevance_score) tuples.
        """
        # Convert types to strings
        type_names = None
        if chunk_types:
            type_names = [t.name for t in chunk_types]
        
        # Search database
        results = await self._db.search_chunks(
            query=query,
            chunk_types=type_names,
            min_importance=min_importance,
            limit=limit,
        )
        
        # Convert to chunks
        chunks = []
        for row, score in results:
            chunk = self._row_to_chunk(row)
            chunks.append((chunk, score))
        
        return chunks
    
    async def get_chunk_content(self, chunk_id: str) -> Optional[str]:
        """Get decompressed content of a specific chunk."""
        row = await self._db.get_chunk(chunk_id)
        if not row:
            return None
        
        chunk = self._row_to_chunk(row)
        return chunk.decompress_content()
    
    # ========== Conversation Storage ==========
    
    async def store_conversation_turn(
        self,
        user_id: str,
        user_message: str,
        agent_response: str,
        tools_used: Optional[List[str]] = None,
        importance: float = 1.0,
    ) -> str:
        """Store a conversation turn as a memory chunk."""
        content = json.dumps({
            "user_id": user_id,
            "user_message": user_message,
            "agent_response": agent_response,
            "tools_used": tools_used or [],
            "timestamp": datetime.now().isoformat(),
        })
        
        return await self.store_in_ems(
            content=content,
            chunk_type=MemoryChunkType.CONVERSATION,
            importance=importance,
            tags=["conversation", f"user_{user_id}"],
            summary=f"Conversation with {user_id}: {user_message[:80]}...",
            load_hints=[user_id, "conversation", "chat"],
            source=f"conversation:{user_id}",
        )
    
    # ========== RCB Management ==========
    
    def format_rcb_for_prompt(self, max_chars: int = 6000) -> str:
        """
        Format the current RCB as a REPL-style prompt fragment.
        
        This is what the LLM sees as its "working memory".
        """
        lines = []
        total_chars = 0
        
        for entry in sorted(self._rcb_entries, key=lambda e: e.display_order):
            formatted = entry.format_for_prompt()
            
            # Check length limit
            if total_chars + len(formatted) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 50:
                    lines.append(formatted[:remaining] + "...")
                break
            
            lines.append(formatted)
            total_chars += len(formatted) + 1  # +1 for newline
        
        return "\n".join(lines) if lines else "[Working memory empty]"
    
    async def clear_rcb(self) -> None:
        """Clear all RCB entries (keeps system prompt)."""
        async with self._rcb_lock:
            system_entries = [e for e in self._rcb_entries if e.entry_type == "system_prompt"]
            self._rcb_entries = system_entries
            await self._db.clear_rcb()
    
    # ========== Forgetting Curve ==========
    
    async def apply_forgetting_curve(self) -> Dict[str, Any]:
        """
        Apply Ebbinghaus-inspired forgetting curve to archived old memories.
        
        Memories decay based on:
        - Time since last access
        - Original importance (higher = slower decay)
        - Access frequency (more accesses = slower decay)
        """
        # Get all non-archived chunks
        all_chunks = await self._db.get_chunks_by_importance(0.0, 10.0, limit=10000)
        
        archived_count = 0
        now = datetime.now()
        
        for row in all_chunks:
            chunk = self._row_to_chunk(row)
            
            # Calculate retention probability
            days_since_access = (now - chunk.last_accessed).total_seconds() / 86400
            
            # R = e^(-t/S) where t=time, S=memory strength
            # Strength = base importance * strength multiplier per access
            strength = chunk.memory_importance * (1 + self.strength_multiplier * chunk.access_count)
            retention = 2.718281828 ** (-days_since_access / (self.retention_halflife_days * strength))
            
            # Archive if retention too low
            if retention < 0.1:  # Less than 10% remembered
                # In real implementation, would update archived flag
                archived_count += 1
        
        # Also run database archive for very low importance
        db_archived = await self._db.archive_low_importance_chunks(0.3)
        
        return {
            "evaluated_count": len(all_chunks),
            "archived_count": archived_count + db_archived,
            "parameters": {
                "halflife_days": self.retention_halflife_days,
                "strength_multiplier": self.strength_multiplier,
            },
        }
    
    # ========== Statistics ==========
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        db_stats = await self._db.get_stats()
        
        async with self._rcb_lock:
            rcb_count = len(self._rcb_entries)
        
        return {
            **db_stats,
            "rcb_entries": rcb_count,
            "rcb_capacity": self.rcb_capacity,
            "compression_efficiency": db_stats.get("avg_compression", 1.0),
        }
    
    # ========== Cleanup ==========
    
    async def close(self) -> None:
        """Close the memory system."""
        await self._db.close()
        self._initialized = False
