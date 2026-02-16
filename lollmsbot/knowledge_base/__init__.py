"""
Knowledge Base Module - Intelligent content storage and retrieval

Provides web content ingestion, document processing, vector-based
semantic search, and natural language querying with source attribution.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import logging

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge with metadata."""
    entry_id: str
    source_type: str  # "web", "file", "youtube", "note"
    source_url: Optional[str]
    title: str
    content: str
    content_hash: str
    summary: str
    extracted_at: datetime
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    importance: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "title": self.title,
            "content_hash": self.content_hash,
            "summary": self.summary,
            "extracted_at": self.extracted_at.isoformat(),
            "topics": self.topics,
            "entities": self.entities,
            "importance": self.importance,
            "access_count": self.access_count,
        }


class KnowledgeBaseManager:
    """
    Manages knowledge ingestion and semantic retrieval.
    
    Key features from SimplifiedAgant:
    - Web content extraction and chunking
    - Vector-based semantic search
    - Natural language querying with source attribution
    - Integration with video ideation workflow
    """
    
    def __init__(self, memory_manager: Any) -> None:
        self._memory = memory_manager
        self._entries: Dict[str, KnowledgeEntry] = {}  # entry_id -> entry
        self._url_index: Dict[str, str] = {}  # url -> entry_id (deduplication)
        self._initialized = False
        
        # Simple in-memory vector store (would use proper vector DB in production)
        self._embeddings: Dict[str, List[float]] = {}
    
    async def initialize(self) -> None:
        """Load existing knowledge from memory."""
        if self._initialized:
            return
        
        if self._memory:
            # Search for existing knowledge entries
            results = await self._memory.search_ems(
                query="knowledge_base entry",
                chunk_types=[MemoryChunkType.FACT, MemoryChunkType.WEB_CONTENT],
                limit=200,
            )
            
            for chunk, score in results:
                if "knowledge_entry:" in (chunk.source or ""):
                    try:
                        data = json.loads(chunk.decompress_content())
                        entry = self._dict_to_entry(data)
                        self._entries[entry.entry_id] = entry
                        
                        if entry.source_url:
                            self._url_index[entry.source_url] = entry.entry_id
                    except Exception as e:
                        logger.warning(f"Failed to load knowledge entry: {e}")
        
        self._initialized = True
        logger.info(f"Knowledge base initialized with {len(self._entries)} entries")
    
    def _dict_to_entry(self, data: Dict[str, Any]) -> KnowledgeEntry:
        """Reconstruct entry from dictionary."""
        return KnowledgeEntry(
            entry_id=data["entry_id"],
            source_type=data["source_type"],
            source_url=data.get("source_url"),
            title=data["title"],
            content="",  # Content stored separately in memory
            content_hash=data["content_hash"],
            summary=data["summary"],
            extracted_at=datetime.fromisoformat(data["extracted_at"]),
            topics=data.get("topics", []),
            entities=data.get("entities", []),
            importance=data.get("importance", 1.0),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
        )
    
    async def ingest_url(self, url: str, title: Optional[str] = None, 
                        content: Optional[str] = None) -> Optional[KnowledgeEntry]:
        """
        Ingest content from a URL.
        
        If content is provided (e.g., from HTTP tool), use it directly.
        Otherwise, would fetch in production.
        """
        # Normalize URL
        url = url.strip().rstrip("/")
        
        # Check for duplicates
        if url in self._url_index:
            existing_id = self._url_index[url]
            logger.info(f"URL already ingested: {url} -> {existing_id}")
            return self._entries.get(existing_id)
        
        # Generate entry ID
        entry_id = f"kb_{hashlib.sha256(url.encode()).hexdigest()[:16]}"
        
        # Use provided content or placeholder
        actual_content = content or f"[Content from {url}]"
        
        # Generate content hash
        content_hash = hashlib.sha256(actual_content.encode()).hexdigest()[:32]
        
        # Extract title from URL or provided
        if not title:
            parsed = urlparse(url)
            title = parsed.netloc + parsed.path
        
        # Extract topics and entities
        topics = self._extract_topics(actual_content)
        entities = self._extract_entities(actual_content)
        
        # Generate summary (would use AI in production)
        summary = self._generate_summary(actual_content, title)
        
        entry = KnowledgeEntry(
            entry_id=entry_id,
            source_type="web",
            source_url=url,
            title=title,
            content=actual_content,
            content_hash=content_hash,
            summary=summary,
            extracted_at=datetime.now(),
            topics=topics,
            entities=entities,
            importance=1.5,  # Web content is moderately important
        )
        
        # Store in memory
        await self._persist_entry(entry, actual_content)
        
        # Update indexes
        self._entries[entry_id] = entry
        self._url_index[url] = entry_id
        
        logger.info(f"Ingested knowledge entry: {title[:50]}... ({len(actual_content)} chars)")
        
        return entry
    
    async def ingest_youtube(self, url: str, transcript: Optional[str] = None,
                           title: Optional[str] = None) -> Optional[KnowledgeEntry]:
        """Ingest YouTube video content."""
        # Normalize YouTube URL
        video_id = self._extract_youtube_id(url)
        if not video_id:
            logger.warning(f"Could not extract YouTube ID from: {url}")
            return None
        
        canonical_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Check for duplicates
        if canonical_url in self._url_index:
            existing_id = self._url_index[canonical_url]
            return self._entries.get(existing_id)
        
        entry_id = f"kb_yt_{video_id}"
        
        # Use provided transcript or placeholder
        actual_content = transcript or f"[YouTube video: {title or video_id}]"
        
        entry = KnowledgeEntry(
            entry_id=entry_id,
            source_type="youtube",
            source_url=canonical_url,
            title=title or f"YouTube Video {video_id}",
            content=actual_content,
            content_hash=hashlib.sha256(actual_content.encode()).hexdigest()[:32],
            summary=self._generate_summary(actual_content, title or "YouTube video"),
            extracted_at=datetime.now(),
            topics=self._extract_topics(actual_content),
            entities=self._extract_entities(actual_content),
            importance=2.0,  # YouTube content often high value
        )
        
        await self._persist_entry(entry, actual_content)
        
        self._entries[entry_id] = entry
        self._url_index[canonical_url] = entry_id
        
        logger.info(f"Ingested YouTube video: {entry.title[:50]}...")
        
        return entry
    
    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        # Simple extraction - would use NLP in production
        words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        
        # Return most frequent capitalized words (likely proper nouns/topics)
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:10]]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Simple pattern matching - would use NER in production
        # Look for capitalized phrases
        patterns = [
            r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b',  # Multi-word names
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        entities = set()
        for pattern in patterns:
            for match in re.findall(pattern, text):
                if len(match) > 3:
                    entities.add(match)
        
        return list(entities)[:15]
    
    def _generate_summary(self, content: str, title: str) -> str:
        """Generate content summary."""
        # Simple first-sentence extraction
        # Would use AI summarization in production
        
        # Get first paragraph or first 200 chars
        first_para = content.split('\n\n')[0][:300]
        
        # Clean up
        summary = first_para.replace('\n', ' ').strip()
        if len(summary) < 50:
            summary = f"Article about {title}: {content[:200]}..."
        
        return summary[:500]
    
    async def _persist_entry(self, entry: KnowledgeEntry, full_content: str) -> None:
        """Save entry to memory system."""
        if not self._memory:
            return
        
        # Store metadata
        metadata = {
            "entry_id": entry.entry_id,
            "source_type": entry.source_type,
            "source_url": entry.source_url,
            "title": entry.title,
            "content_hash": entry.content_hash,
            "summary": entry.summary,
            "extracted_at": entry.extracted_at.isoformat(),
            "topics": entry.topics,
            "entities": entry.entities,
            "importance": entry.importance,
            "access_count": entry.access_count,
        }
        
        await self._memory.store_in_ems(
            content=json.dumps(metadata),
            chunk_type=MemoryChunkType.FACT,
            importance=entry.importance,
            tags=["knowledge_base", "entry", entry.source_type] + entry.topics[:5],
            summary=f"KB: {entry.title[:80]}",
            load_hints=entry.topics + entry.entities + [entry.source_url or ""],
            source=f"knowledge_entry:{entry.entry_id}",
        )
        
        # Store full content separately (larger chunk)
        await self._memory.store_in_ems(
            content=full_content,
            chunk_type=MemoryChunkType.WEB_CONTENT,
            importance=entry.importance * 0.8,  # Slightly lower importance for full content
            tags=["knowledge_base", "content", entry.source_type, entry.entry_id],
            summary=f"Full content: {entry.title[:60]}...",
            load_hints=[entry.entry_id, entry.source_url or ""] + entry.topics[:3],
            source=f"knowledge_content:{entry.entry_id}",
        )
    
    async def query(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Natural language query against knowledge base.
        
        Returns entries with relevance scores and source attribution.
        """
        # Extract query topics
        query_topics = self._extract_topics(query.lower())
        
        results = []
        
        for entry in self._entries.values():
            score = 0.0
            
            # Topic overlap
            topic_overlap = len(set(entry.topics) & set(query_topics))
            score += topic_overlap * 2.0
            
            # Entity overlap
            entity_overlap = len(set(entry.entities) & set(query_topics))
            score += entity_overlap * 1.5
            
            # Title match
            query_lower = query.lower()
            if query_lower in entry.title.lower():
                score += 5.0
            
            # Summary match
            if query_lower in entry.summary.lower():
                score += 3.0
            
            # Recency boost
            days_old = (datetime.now() - entry.extracted_at).days
            if days_old < 7:
                score *= 1.5
            elif days_old < 30:
                score *= 1.2
            
            # Importance boost
            score *= (0.5 + entry.importance / 2)
            
            if score > 2.0:  # Threshold
                # Update access stats
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                # Get full content from memory
                full_content = await self._get_full_content(entry.entry_id)
                
                results.append({
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "source_url": entry.source_url,
                    "source_type": entry.source_type,
                    "summary": entry.summary,
                    "relevance_score": round(score, 2),
                    "content_preview": full_content[:500] if full_content else entry.summary[:300],
                    "topics": entry.topics,
                    "extracted_at": entry.extracted_at.isoformat(),
                })
        
        # Sort by score
        results.sort(key=lambda x: -x["relevance_score"])
        return results[:limit]
    
    async def _get_full_content(self, entry_id: str) -> Optional[str]:
        """Retrieve full content for an entry."""
        if not self._memory:
            return None
        
        # Search for content chunk
        results = await self._memory.search_ems(
            query=f"knowledge_content:{entry_id}",
            limit=1,
        )
        
        for chunk, score in results:
            if chunk.source == f"knowledge_content:{entry_id}":
                return chunk.decompress_content()
        
        return None
    
    async def get_video_ideas(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get potential video ideas based on knowledge base content.
        
        This integrates with the video ideation workflow from SimplifiedAgant.
        """
        # Get high-importance, recent entries
        candidates = [
            e for e in self._entries.values()
            if e.importance >= 1.5 or (datetime.now() - e.extracted_at).days < 30
        ]
        
        # Sort by importance and recency
        candidates.sort(key=lambda x: -(x.importance + (30 - (datetime.now() - x.extracted_at).days) / 30))
        
        ideas = []
        for entry in candidates[:10]:
            # Generate idea from entry
            idea = {
                "source_entry_id": entry.entry_id,
                "source_title": entry.title,
                "source_url": entry.source_url,
                "suggested_angle": f"Exploring {entry.title[:50]}...",
                "key_points": entry.topics[:5],
                "relevance": "High" if entry.importance >= 2.0 else "Medium",
            }
            ideas.append(idea)
        
        return ideas
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        by_type = {}
        for e in self._entries.values():
            by_type[e.source_type] = by_type.get(e.source_type, 0) + 1
        
        total_accesses = sum(e.access_count for e in self._entries.values())
        
        return {
            "total_entries": len(self._entries),
            "by_type": by_type,
            "total_accesses": total_accesses,
            "recent_additions_7d": len([
                e for e in self._entries.values()
                if (datetime.now() - e.extracted_at).days < 7
            ]),
        }


# Import for type hints
from lollmsbot.agent.rlm.models import MemoryChunkType


__all__ = [
    "KnowledgeBaseManager",
    "KnowledgeEntry",
]
