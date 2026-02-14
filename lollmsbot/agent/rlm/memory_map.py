"""
Memory Map - Hierarchical visualization of RLM memory for LLM introspection.

Provides structured overviews of what the AI knows, organized by category,
importance, and relationships. Enables efficient memory navigation and
self-awareness of knowledge boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from lollmsbot.agent.rlm.models import MemoryChunk, MemoryChunkType


@dataclass
class MemoryCategorySummary:
    """Summary of a memory category."""
    category: str  # conversation, web_content, fact, skill, self_knowledge, etc.
    count: int = 0
    total_importance: float = 0.0
    avg_importance: float = 0.0
    newest_timestamp: Optional[datetime] = None
    oldest_timestamp: Optional[datetime] = None
    key_topics: List[str] = field(default_factory=list)
    sample_summaries: List[str] = field(default_factory=list)
    total_content_length: int = 0


@dataclass
class MemoryAnchor:
    """A navigable reference point in memory."""
    chunk_id: str
    anchor_type: str  # "key_fact", "important_conversation", "reference_material", "user_preference"
    title: str
    summary: str
    importance: float
    age_days: float
    access_count: int
    related_anchors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    load_hint: str = ""  # How to retrieve: "ask about {topic}" or "use [[MEMORY:{id}]]"


@dataclass
class KnowledgeGap:
    """Identified gap in memory that might need filling."""
    gap_type: str  # "missing_context", "outdated_info", "unverified_fact", "conflicting_data"
    description: str
    suggested_action: str
    priority: float  # 0-10


class MemoryMap:
    """
    Generates hierarchical views of RLM memory for LLM introspection.
    
    The MemoryMap provides:
    - Category summaries (what types of knowledge exist)
    - Anchor points (key memories to start from)
    - Relationship hints (how memories connect)
    - Knowledge gaps (what might be missing)
    
    This allows the LLM to "see" its memory structure and make intelligent
    decisions about retrieval without exhaustive search.
    """
    
    def __init__(self, memory_manager: Any) -> None:
        self._memory = memory_manager
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds: float = 60.0  # Refresh every minute
        
        # Statistics
        self._generation_count: int = 0
        self._last_generation_time: float = 0.0
    
    async def generate(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Generate complete memory map.
        
        Returns hierarchical structure with categories, anchors, and gaps.
        """
        # Check cache
        if not force_refresh and self._cache and self._cache_timestamp:
            age = (datetime.now() - self._cache_timestamp).total_seconds()
            if age < self._cache_ttl_seconds:
                return self._cache
        
        import time
        start_time = time.time()
        
        # Gather all chunks
        all_chunks = await self._get_all_chunks()
        
        # Build structure
        structure = {
            "generated_at": datetime.now().isoformat(),
            "total_memories": len(all_chunks),
            "categories": await self._categorize_memories(all_chunks),
            "anchors": await self._identify_anchors(all_chunks),
            "relationships": await self._map_relationships(all_chunks),
            "knowledge_gaps": await self._identify_gaps(all_chunks),
            "retrieval_hints": self._generate_retrieval_hints(all_chunks),
            "statistics": {
                "avg_importance": sum(c.memory_importance for c in all_chunks) / len(all_chunks) if all_chunks else 0,
                "avg_age_days": self._calculate_avg_age(all_chunks),
                "total_content_bytes": sum(len(c.decompress_content() or "") for c in all_chunks),
            },
        }
        
        # Cache result
        self._cache = structure
        self._cache_timestamp = datetime.now()
        self._generation_count += 1
        self._last_generation_time = time.time() - start_time
        
        return structure
    
    async def _get_all_chunks(self) -> List[MemoryChunk]:
        """Retrieve all non-archived memory chunks."""
        chunks = []
        
        try:
            # Use database to get all active chunks
            rows = await self._memory._db.get_chunks_by_importance(0.0, 10.0, limit=10000)
            for row in rows:
                chunk = self._memory._row_to_chunk(row)
                if not chunk.archived:
                    chunks.append(chunk)
        except Exception as e:
            # Fallback: use RCB entries as proxy
            rcb_entries = await self._memory._db.get_rcb_entries(limit=100)
            for entry in rcb_entries:
                if entry.get("chunk_id"):
                    chunk = await self._memory.load_from_ems(entry["chunk_id"], add_to_rcb=False)
                    if chunk:
                        chunks.append(chunk)
        
        return chunks
    
    async def _categorize_memories(self, chunks: List[MemoryChunk]) -> Dict[str, MemoryCategorySummary]:
        """Group memories by type and generate summaries."""
        by_type: Dict[MemoryChunkType, List[MemoryChunk]] = defaultdict(list)
        
        for chunk in chunks:
            by_type[chunk.chunk_type].append(chunk)
        
        categories: Dict[str, MemoryCategorySummary] = {}
        
        for chunk_type, type_chunks in by_type.items():
            category_name = chunk_type.name.lower()
            
            # Calculate statistics
            importances = [c.memory_importance for c in type_chunks]
            timestamps = [c.last_accessed for c in type_chunks if c.last_accessed]
            
            # Extract topics from tags and summaries
            all_tags: Set[str] = set()
            all_keywords: Set[str] = set()
            for c in type_chunks:
                all_tags.update(c.tags or [])
                # Extract keywords from summary
                if c.summary:
                    words = c.summary.lower().split()[:20]
                    all_keywords.update(w for w in words if len(w) > 3)
            
            # Get sample summaries (highest importance)
            top_chunks = sorted(type_chunks, key=lambda c: -c.memory_importance)[:5]
            sample_summaries = [
                c.summary[:100] + "..." if c.summary and len(c.summary) > 100 else (c.summary or "No summary")
                for c in top_chunks
            ]
            
            # Calculate total content length
            total_length = sum(len(c.decompress_content() or "") for c in type_chunks)
            
            summary = MemoryCategorySummary(
                category=category_name,
                count=len(type_chunks),
                total_importance=sum(importances),
                avg_importance=sum(importances) / len(importances) if importances else 0,
                newest_timestamp=max(timestamps) if timestamps else None,
                oldest_timestamp=min(timestamps) if timestamps else None,
                key_topics=sorted(all_keywords)[:10],  # Top 10 keywords
                sample_summaries=sample_summaries,
                total_content_length=total_length,
            )
            
            categories[category_name] = summary
        
        return categories
    
    async def _identify_anchors(self, chunks: List[MemoryChunk]) -> List[MemoryAnchor]:
        """
        Identify key anchor points - memories that serve as good entry points.
        
        Criteria:
        - High importance (>7)
        - Frequently accessed
        - Recent or timeless
        - Well-connected (many tags)
        """
        anchors = []
        
        for chunk in chunks:
            # Skip low-importance memories
            if chunk.memory_importance < 5.0:
                continue
            
            # Calculate anchor score
            age_days = (datetime.now() - chunk.created_at).days if chunk.created_at else 365
            recency_bonus = max(0, 10 - age_days / 30)  # Bonus for recent memories
            
            access_score = min(chunk.access_count / 10, 5)  # Cap at 5
            
            connectivity = len(chunk.tags or 0) + len(chunk.load_hints or 0)
            connectivity_score = min(connectivity / 5, 3)
            
            anchor_score = (
                chunk.memory_importance * 2 +
                recency_bonus +
                access_score +
                connectivity_score
            )
            
            # Only keep top anchors
            if anchor_score < 15:
                continue
            
            # Determine anchor type
            if chunk.chunk_type == MemoryChunkType.SELF_KNOWLEDGE:
                anchor_type = "key_fact"
            elif chunk.chunk_type == MemoryChunkType.CONVERSATION and chunk.memory_importance >= 8:
                anchor_type = "important_conversation"
            elif chunk.chunk_type == MemoryChunkType.WEB_CONTENT:
                anchor_type = "reference_material"
            elif chunk.chunk_type == MemoryChunkType.USER_PREFERENCE:
                anchor_type = "user_preference"
            else:
                anchor_type = "general_knowledge"
            
            # Generate load hint
            if chunk.load_hints:
                load_hint = f"Use [[MEMORY:{chunk.chunk_id}]] or ask about {chunk.load_hints[0]}"
            else:
                load_hint = f"Use [[MEMORY:{chunk.chunk_id}]]"
            
            anchor = MemoryAnchor(
                chunk_id=chunk.chunk_id,
                anchor_type=anchor_type,
                title=chunk.summary[:60] if chunk.summary else f"Memory {chunk.chunk_id[:12]}",
                summary=chunk.summary[:200] if chunk.summary else "No summary available",
                importance=chunk.memory_importance,
                age_days=age_days,
                access_count=chunk.access_count,
                keywords=(chunk.tags or [])[:5],
                load_hint=load_hint,
            )
            
            anchors.append(anchor)
        
        # Sort by importance and return top 20
        anchors.sort(key=lambda a: -a.importance)
        return anchors[:20]
    
    async def _map_relationships(self, chunks: List[MemoryChunk]) -> Dict[str, List[str]]:
        """Map relationships between memory chunks based on shared tags/hints."""
        relationships: Dict[str, List[str]] = defaultdict(list)
        
        # Build tag index
        tag_to_chunks: Dict[str, List[str]] = defaultdict(list)
        for chunk in chunks:
            for tag in chunk.tags or []:
                tag_to_chunks[tag].append(chunk.chunk_id)
            for hint in chunk.load_hints or []:
                tag_to_chunks[hint].append(chunk.chunk_id)
        
        # Find related chunks (share 2+ tags or are in same tag group)
        for chunk in chunks:
            related: Set[str] = set()
            
            # Direct tag overlap
            my_tags = set(chunk.tags or []) | set(chunk.load_hints or [])
            for tag in my_tags:
                for other_id in tag_to_chunks[tag]:
                    if other_id != chunk.chunk_id:
                        related.add(other_id)
            
            # Semantic similarity via summary keywords
            if chunk.summary:
                my_keywords = set(chunk.summary.lower().split())
                for other in chunks:
                    if other.chunk_id == chunk.chunk_id:
                        continue
                    if other.summary:
                        other_keywords = set(other.summary.lower().split())
                        overlap = len(my_keywords & other_keywords)
                        if overlap >= 3:  # Significant keyword overlap
                            related.add(other.chunk_id)
            
            relationships[chunk.chunk_id] = list(related)[:5]  # Top 5 related
        
        return dict(relationships)
    
    async def _identify_gaps(self, chunks: List[MemoryChunk]) -> List[KnowledgeGap]:
        """Identify potential knowledge gaps."""
        gaps = []
        
        # Check for outdated information
        now = datetime.now()
        for chunk in chunks:
            if chunk.chunk_type == MemoryChunkType.WEB_CONTENT:
                age_days = (now - chunk.created_at).days if chunk.created_at else 0
                if age_days > 30 and chunk.memory_importance > 7:
                    gaps.append(KnowledgeGap(
                        gap_type="outdated_info",
                        description=f"Web content '{chunk.summary[:40]}...' is {age_days} days old",
                        suggested_action="Consider re-fetching if time-sensitive",
                        priority=min(10, chunk.memory_importance * age_days / 30),
                    ))
        
        # Check for unverified facts (high importance, low access)
        for chunk in chunks:
            if chunk.memory_importance > 8 and chunk.access_count < 2:
                gaps.append(KnowledgeGap(
                    gap_type="unverified_fact",
                    description=f"High-importance memory '{chunk.summary[:40]}...' rarely accessed",
                    suggested_action="Verify accuracy with user when relevant",
                    priority=chunk.memory_importance,
                ))
        
        # Check for missing context in conversations
        conv_chunks = [c for c in chunks if c.chunk_type == MemoryChunkType.CONVERSATION]
        if len(conv_chunks) > 10:
            # Check for conversation threads with gaps
            gaps.append(KnowledgeGap(
                gap_type="missing_context",
                description=f"{len(conv_chunks)} conversation memories - some threads may be fragmented",
                suggested_action="Use [[MEMORY:...]] handles to reconstruct full context when needed",
                priority=5.0,
            ))
        
        # Sort by priority
        gaps.sort(key=lambda g: -g.priority)
        return gaps[:10]
    
    def _generate_retrieval_hints(self, chunks: List[MemoryChunk]) -> Dict[str, List[str]]:
        """Generate natural language hints for memory retrieval."""
        hints: Dict[str, List[str]] = {
            "by_topic": [],
            "by_time": [],
            "by_importance": [],
            "by_relationship": [],
        }
        
        # Topic-based hints
        topics: Dict[str, int] = defaultdict(int)
        for chunk in chunks:
            for tag in chunk.tags or []:
                topics[tag] += 1
        
        top_topics = sorted(topics.items(), key=lambda x: -x[1])[:5]
        for topic, count in top_topics:
            hints["by_topic"].append(f"Ask about '{topic}' to access {count} related memories")
        
        # Time-based hints
        recent = [c for c in chunks if c.last_accessed and (datetime.now() - c.last_accessed).days < 7]
        if recent:
            hints["by_time"].append(f"Recent activity: {len(recent)} memories accessed in last 7 days")
        
        # Importance-based hints
        high_imp = [c for c in chunks if c.memory_importance >= 8]
        if high_imp:
            hints["by_importance"].append(f"Critical knowledge: {len(high_imp)} high-importance memories")
        
        # Relationship hints
        hints["by_relationship"].append("Use [[MEMORY:chunk_id]] to load specific memories into context")
        hints["by_relationship"].append("Memories with shared tags are automatically cross-referenced")
        
        return hints
    
    def _calculate_avg_age(self, chunks: List[MemoryChunk]) -> float:
        """Calculate average age in days."""
        if not chunks:
            return 0.0
        
        ages = []
        now = datetime.now()
        for chunk in chunks:
            if chunk.created_at:
                age = (now - chunk.created_at).days
                ages.append(age)
        
        return sum(ages) / len(ages) if ages else 0.0
    
    def format_for_prompt(self, memory_map: Dict[str, Any], max_length: int = 4000) -> str:
        """
        Format memory map as natural language for LLM system prompt.
        
        Structured to help the LLM understand its knowledge landscape
        and make intelligent retrieval decisions.
        """
        lines = [
            "",
            "=" * 60,
            "🧠 YOUR MEMORY STRUCTURE - WHAT YOU KNOW",
            "=" * 60,
            "",
            f"Total memories: {memory_map['total_memories']}",
            f"Generated: {memory_map['generated_at']}",
            f"Avg importance: {memory_map['statistics']['avg_importance']:.1f}/10",
            f"Avg age: {memory_map['statistics']['avg_age_days']:.0f} days",
            "",
            "-" * 60,
            "📂 KNOWLEDGE CATEGORIES (what types of information you have)",
            "-" * 60,
        ]
        
        # Categories
        for cat_name, cat_summary in sorted(
            memory_map['categories'].items(),
            key=lambda x: -x[1].avg_importance
        ):
            lines.extend([
                f"",
                f"  ▶ {cat_name.upper()} ({cat_summary.count} items)",
                f"    Importance: {cat_summary.avg_importance:.1f}/10 avg",
            ])
            
            if cat_summary.newest_timestamp:
                lines.append(f"    Last updated: {cat_summary.newest_timestamp.strftime('%Y-%m-%d')}")
            
            if cat_summary.key_topics:
                topics_str = ", ".join(cat_summary.key_topics[:5])
                lines.append(f"    Key topics: {topics_str}")
            
            if cat_summary.sample_summaries:
                lines.append(f"    Examples: {cat_summary.sample_summaries[0][:80]}...")
        
        lines.extend([
            "",
            "-" * 60,
            "⚓ MEMORY ANCHORS (key entry points to your knowledge)",
            "-" * 60,
            "",
        ])
        
        # Anchors
        for anchor in memory_map['anchors'][:10]:  # Top 10 anchors
            age_str = f"{anchor.age_days:.0f}d ago" if anchor.age_days < 365 else "long-term"
            lines.extend([
                f"  🔹 [{anchor.anchor_type}] {anchor.title[:50]}",
                f"     Importance: {anchor.importance:.1f} | Accessed: {anchor.access_count}x | {age_str}",
                f"     How to retrieve: {anchor.load_hint}",
            ])
            if anchor.keywords:
                lines.append(f"     Keywords: {', '.join(anchor.keywords[:4])}")
            lines.append("")
        
        lines.extend([
            "-" * 60,
            "🔍 RETRIEVAL HINTS (how to find what you need)",
            "-" * 60,
            "",
        ])
        
        hints = memory_map['retrieval_hints']
        for hint_category, hint_list in hints.items():
            lines.append(f"  {hint_category.replace('_', ' ').title()}:")
            for hint in hint_list[:3]:
                lines.append(f"    • {hint}")
        
        # Knowledge gaps
        if memory_map['knowledge_gaps']:
            lines.extend([
                "",
                "-" * 60,
                "⚠️ KNOWLEDGE GAPS (areas that may need attention)",
                "-" * 60,
                "",
            ])
            for gap in memory_map['knowledge_gaps'][:5]:
                lines.append(f"  • [{gap['gap_type']}] {gap['description'][:60]}")
                lines.append(f"    Action: {gap['suggested_action'][:60]}")
        
        lines.extend([
            "",
            "-" * 60,
            "📝 MEMORY ACCESS PROTOCOL",
            "-" * 60,
            "",
            "To access specific memories:",
            "1. Use [[MEMORY:chunk_id]] to load a memory handle",
            "2. Reference anchor keywords naturally in your response",
            "3. For broad queries, I will search relevant categories",
            "4. Recent high-importance memories are loaded automatically",
            "",
            "=" * 60,
        ])
        
        result = "\n".join(lines)
        
        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length - 100] + "\n\n... [Memory structure truncated for length]"
        
        return result
    
    def get_quick_summary(self, memory_map: Dict[str, Any]) -> str:
        """Get brief summary for status checks."""
        cats = memory_map.get('categories', {})
        total = memory_map.get('total_memories', 0)
        anchors = len(memory_map.get('anchors', []))
        
        cat_list = ", ".join(f"{k}({v['count']})" for k, v in list(cats.items())[:3])
        
        return f"{total} memories in {len(cats)} categories ({cat_list}), {anchors} anchors"
    
    def invalidate_cache(self) -> None:
        """Force regeneration on next call."""
        self._cache = None
        self._cache_timestamp = None
