"""
RLM Memory Maintenance - Deduplication, Consolidation, and Optimization

Provides heartbeat-driven memory organization to prevent bloat and redundancy.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict

from lollmsbot.agent.rlm.models import MemoryChunk, MemoryChunkType
from lollmsbot.agent.rlm.manager import RLMMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of memory deduplication pass."""
    duplicates_found: int = 0
    duplicates_removed: int = 0
    bytes_saved: int = 0
    merged_chunks: List[Tuple[str, str, str]] = field(default_factory=list)  # (removed_id, kept_id, reason)


@dataclass
class ConsolidationResult:
    """Result of memory consolidation pass."""
    clusters_found: int = 0
    narratives_created: int = 0
    chunks_consolidated: int = 0
    new_narrative_ids: List[str] = field(default_factory=list)


@dataclass
class MaintenanceReport:
    """Complete report from a memory maintenance run."""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    deduplication: Optional[DeduplicationResult] = None
    consolidation: Optional[ConsolidationResult] = None
    compression_ratio: float = 1.0
    total_chunks_before: int = 0
    total_chunks_after: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "deduplication": {
                "duplicates_found": self.deduplication.duplicates_found if self.deduplication else 0,
                "duplicates_removed": self.deduplication.duplicates_removed if self.deduplication else 0,
                "bytes_saved": self.deduplication.bytes_saved if self.deduplication else 0,
            } if self.deduplication else None,
            "consolidation": {
                "clusters_found": self.consolidation.clusters_found if self.consolidation else 0,
                "narratives_created": self.consolidation.narratives_created if self.consolidation else 0,
                "chunks_consolidated": self.chunks_consolidated if self.consolidation else 0,
            } if self.consolidation else None,
            "compression_ratio": self.compression_ratio,
            "total_chunks_before": self.total_chunks_before,
            "total_chunks_after": self.total_chunks_after,
            "errors": self.errors,
        }


class ContentHasher:
    """Generate content fingerprints for deduplication."""
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Generate hash of content for exact duplicate detection."""
        # Normalize: lowercase, strip whitespace, normalize unicode
        normalized = content.lower().strip()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:32]
    
    @staticmethod
    def hash_url(url: str) -> str:
        """Generate hash for URL-based content (web pages)."""
        # Normalize URL: remove trailing slashes, fragments, normalize query order
        normalized = url.lower().strip().rstrip('/')
        # Remove common tracking parameters
        for param in ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid', 'gclid']:
            # Simple removal - production would parse properly
            if f'{param}=' in normalized:
                # This is a simplified version
                pass
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:24]
    
    @staticmethod
    def similarity_hash(content: str, bands: int = 4) -> List[str]:
        """Generate locality-sensitive hashes for near-duplicate detection."""
        # SimHash-style approach: split content into bands
        words = content.lower().split()
        hashes = []
        band_size = max(1, len(words) // bands)
        
        for i in range(bands):
            start = i * band_size
            end = start + band_size if i < bands - 1 else len(words)
            band_content = ' '.join(words[start:end])
            band_hash = hashlib.sha256(band_content.encode('utf-8')).hexdigest()[:16]
            hashes.append(band_hash)
        
        return hashes


class MemoryMaintenance:
    """
    Maintenance operations for RLM memory system.
    
    Provides deduplication, consolidation, and optimization services
    that run periodically via Heartbeat or can be triggered manually.
    """
    
    # Minimum time between automatic deduplication runs (prevent thrashing)
    MIN_DEDUP_INTERVAL_MINUTES = 30
    
    # Thresholds for consolidation
    CONSOLIDATION_SIMILARITY_THRESHOLD = 0.7  # 70% similar chunks get consolidated
    
    def __init__(self, memory_manager: RLMMemoryManager):
        self.memory = memory_manager
        self.content_hasher = ContentHasher()
        
        # Track when we last ran operations
        self._last_dedup: Optional[datetime] = None
        self._last_consolidation: Optional[datetime] = None
        
        # Callbacks for progress reporting
        self._progress_callbacks: List[Callable[[str, int, int], None]] = []
    
    def on_progress(self, callback: Callable[[str, int, int], None]) -> None:
        """Register callback for progress updates: (operation, current, total)."""
        self._progress_callbacks.append(callback)
    
    def _report_progress(self, operation: str, current: int, total: int) -> None:
        """Notify all progress callbacks."""
        for cb in self._progress_callbacks:
            try:
                cb(operation, current, total)
            except Exception:
                pass  # Don't let callbacks break maintenance
    
    async def run_full_maintenance(
        self,
        enable_deduplication: bool = True,
        enable_consolidation: bool = True,
        enable_url_dedup: bool = True,
    ) -> MaintenanceReport:
        """
        Run complete memory maintenance cycle.
        
        This is the main entry point for heartbeat-triggered maintenance.
        """
        report = MaintenanceReport()
        report.total_chunks_before = await self._count_active_chunks()
        
        try:
            # Phase 1: Deduplication (fast, always run if enough time passed)
            if enable_deduplication and self._should_run_dedup():
                logger.info("🔍 Starting memory deduplication...")
                report.deduplication = await self.deduplicate_by_content(
                    include_url_based=enable_url_dedup
                )
                self._last_dedup = datetime.now()
            
            # Phase 2: Consolidation (slower, run less frequently)
            if enable_consolidation and self._should_run_consolidation():
                logger.info("🔄 Starting memory consolidation...")
                report.consolidation = await self.consolidate_related_memories()
                self._last_consolidation = datetime.now()
            
            # Update stats
            report.total_chunks_after = await self._count_active_chunks()
            if report.total_chunks_before > 0:
                report.compression_ratio = report.total_chunks_after / report.total_chunks_before
            
        except Exception as e:
            logger.error(f"Memory maintenance failed: {e}")
            report.errors.append(str(e))
        
        report.completed_at = datetime.now()
        return report
    
    def _should_run_dedup(self) -> bool:
        """Check if enough time has passed since last deduplication."""
        if self._last_dedup is None:
            return True
        elapsed = datetime.now() - self._last_dedup
        return elapsed > timedelta(minutes=self.MIN_DEDUP_INTERVAL_MINUTES)
    
    def _should_run_consolidation(self) -> bool:
        """Check if consolidation should run (less frequent than dedup)."""
        if self._last_consolidation is None:
            return True
        elapsed = datetime.now() - self._last_consolidation
        # Run consolidation every 6 hours or so
        return elapsed > timedelta(hours=6)
    
    async def _count_active_chunks(self) -> int:
        """Count non-archived chunks in memory."""
        try:
            stats = await self.memory.get_stats()
            return stats.get('active_chunks', 0)
        except Exception:
            return 0
    
    async def deduplicate_by_content(
        self,
        include_url_based: bool = True,
        similarity_threshold: float = 0.95,
    ) -> DeduplicationResult:
        """
        Find and merge duplicate content chunks.
        
        For web content: uses URL hash to find exact duplicates
        For other content: uses content hash for exact duplicates
        Near-duplicates: uses similarity hashing
        """
        result = DeduplicationResult()
        
        try:
            # Get all web content chunks (the most common source of dups)
            from lollmsbot.agent.rlm.database import RLMDatabase
            
            # Query all chunks directly from DB for efficiency
            all_chunks: List[Dict[str, Any]] = []
            async with self.memory._db._connection.execute(
                """
                SELECT chunk_id, chunk_type, content_compressed, source, summary, 
                       last_accessed, access_count, memory_importance
                FROM memory_chunks 
                WHERE (archived IS NULL OR archived = 0)
                """
            ) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    all_chunks.append({
                        'chunk_id': row[0],
                        'chunk_type': row[1],
                        'content_compressed': row[2],
                        'source': row[3],
                        'summary': row[4],
                        'last_accessed': row[5],
                        'access_count': row[6],
                        'importance': row[7],
                    })
            
            logger.info(f"🔍 Analyzing {len(all_chunks)} chunks for duplicates...")
            
            # Separate by type for different dedup strategies
            web_chunks = [c for c in all_chunks if c['chunk_type'] == 'WEB_CONTENT']
            other_chunks = [c for c in all_chunks if c['chunk_type'] != 'WEB_CONTENT']
            
            # Phase 1: URL-based deduplication for web content
            if include_url_based and web_chunks:
                url_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                
                for chunk in web_chunks:
                    source = chunk.get('source', '')
                    # Extract URL from source field (format: "http_tool:https://example.com")
                    if source and source.startswith('http_tool:'):
                        url = source[10:]  # Remove prefix
                        url_hash = self.content_hasher.hash_url(url)
                        url_groups[url_hash].append(chunk)
                
                # Merge groups with multiple entries
                for url_hash, group in url_groups.items():
                    if len(group) > 1:
                        # Keep the most accessed/important one, merge others into it
                        group.sort(key=lambda c: (c['access_count'], c['importance']), reverse=True)
                        keeper = group[0]
                        
                        for dup in group[1:]:
                            await self._merge_duplicate_chunks(dup['chunk_id'], keeper['chunk_id'])
                            result.duplicates_found += 1
                            result.duplicates_removed += 1
                            result.merged_chunks.append((
                                dup['chunk_id'],
                                keeper['chunk_id'],
                                f"URL duplicate: {dup.get('source', 'unknown')[:50]}..."
                            ))
                            # Estimate bytes saved (compressed size)
                            result.bytes_saved += len(dup.get('content_compressed', b''))
                            
                            self._report_progress(
                                "dedup_url", 
                                result.duplicates_removed, 
                                len(web_chunks)
                            )
            
            # Phase 2: Content hash deduplication for exact matches
            content_hashes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            
            for chunk in other_chunks:
                # Decompress and hash content
                try:
                    chunk_obj = await self.memory._db.get_chunk(chunk['chunk_id'])
                    if chunk_obj:
                        content = self._decompress_content(chunk_obj['content_compressed'])
                        content_hash = self.content_hasher.hash_content(content[:10000])  # First 10k chars
                        content_hashes[content_hash].append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to hash chunk {chunk['chunk_id']}: {e}")
            
            # Merge content duplicates
            for content_hash, group in content_hashes.items():
                if len(group) > 1:
                    group.sort(key=lambda c: (c['access_count'], c['importance'], c['last_accessed']), reverse=True)
                    keeper = group[0]
                    
                    for dup in group[1:]:
                        await self._merge_duplicate_chunks(dup['chunk_id'], keeper['chunk_id'])
                        result.duplicates_found += 1
                        result.duplicates_removed += 1
                        result.merged_chunks.append((
                            dup['chunk_id'],
                            keeper['chunk_id'],
                            "Content hash duplicate"
                        ))
                        result.bytes_saved += len(dup.get('content_compressed', b''))
                        
                        self._report_progress(
                            "dedup_content",
                            result.duplicates_removed,
                            len(other_chunks)
                        )
            
            logger.info(f"✅ Deduplication complete: removed {result.duplicates_removed} duplicates, "
                       f"saved ~{result.bytes_saved / 1024:.1f} KB")
            
        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            raise
        
        return result
    
    async def _merge_duplicate_chunks(self, remove_id: str, keep_id: str) -> None:
        """
        Merge a duplicate chunk into the keeper and archive the duplicate.
        
        Preserves access history and importance from both.
        """
        try:
            # Get both chunks
            remove_chunk = await self.memory._db.get_chunk(remove_id)
            keep_chunk = await self.memory._db.get_chunk(keep_id)
            
            if not remove_chunk or not keep_chunk:
                return
            
            # Merge metadata: add access counts, take max importance
            merged_access_count = (keep_chunk.get('access_count', 0) + 
                                  remove_chunk.get('access_count', 0))
            merged_importance = max(
                keep_chunk.get('memory_importance', 1.0),
                remove_chunk.get('memory_importance', 1.0)
            )
            
            # Update keeper with merged stats
            await self.memory._db._connection.execute(
                """
                UPDATE memory_chunks 
                SET access_count = ?, memory_importance = ?, 
                    last_accessed = ?
                WHERE chunk_id = ?
                """,
                (
                    merged_access_count,
                    merged_importance,
                    datetime.now().isoformat(),
                    keep_id
                )
            )
            
            # Log the merge in access log
            await self.memory._db._connection.execute(
                """
                INSERT INTO access_log (chunk_id, access_type, context_info)
                VALUES (?, ?, ?)
                """,
                (keep_id, 'merged_duplicate', f"Merged from {remove_id}"))
            
            # Archive the duplicate (don't delete, preserve audit trail)
            await self.memory._db._connection.execute(
                """
                UPDATE memory_chunks 
                SET archived = 1, 
                    summary = ?,
                    tags = ?
                WHERE chunk_id = ?
                """,
                (
                    f"[MERGED into {keep_id}] {remove_chunk.get('summary', '')}"[:200],
                    json.dumps(["merged_duplicate", "archived", f"merged_into_{keep_id}"]),
                    remove_id
                )
            )
            
            await self.memory._db._connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to merge chunks {remove_id} -> {keep_id}: {e}")
            raise
    
    async def consolidate_related_memories(self) -> ConsolidationResult:
        """
        Find related memories and merge into coherent narratives.
        
        Example: Multiple mentions of "Python project" become a consolidated
        project memory with timeline, learnings, and current status.
        """
        result = ConsolidationResult()
        
        try:
            # Get recent conversation chunks
            from lollmsbot.agent.rlm.database import RLMDatabase
            
            conv_chunks: List[Dict[str, Any]] = []
            async with self.memory._db._connection.execute(
                """
                SELECT chunk_id, content_compressed, summary, tags, load_hints, created_at
                FROM memory_chunks 
                WHERE chunk_type = 'CONVERSATION' 
                AND (archived IS NULL OR archived = 0)
                AND created_at > ?
                ORDER BY created_at DESC
                LIMIT 200
                """,
                ((datetime.now() - timedelta(days=7)).isoformat(),)
            ) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    conv_chunks.append({
                        'chunk_id': row[0],
                        'content_compressed': row[1],
                        'summary': row[2],
                        'tags': json.loads(row[3]) if row[3] else [],
                        'load_hints': json.loads(row[4]) if row[4] else [],
                        'created_at': row[5],
                    })
            
            # Simple clustering by shared tags/load_hints
            clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            
            for chunk in conv_chunks:
                # Use first tag or load_hint as cluster key
                key_tags = [t for t in (chunk.get('tags') or []) 
                           if t not in ['conversation', 'user']]
                if key_tags:
                    cluster_key = key_tags[0]
                    clusters[cluster_key].append(chunk)
            
            # Create narratives for clusters with 3+ related chunks
            for cluster_key, cluster_chunks in clusters.items():
                if len(cluster_chunks) >= 3:
                    narrative = await self._create_narrative(cluster_key, cluster_chunks)
                    if narrative:
                        result.narratives_created += 1
                        result.chunks_consolidated += len(cluster_chunks)
                        result.new_narrative_ids.append(narrative['id'])
                        
                        # Mark constituents as consolidated
                        for chunk in cluster_chunks:
                            await self.memory._db._connection.execute(
                                """
                                UPDATE memory_chunks 
                                SET archived = 1,
                                    summary = ?,
                                    tags = ?
                                WHERE chunk_id = ?
                                """,
                                (
                                    f"[CONSOLIDATED into {narrative['id']}] {chunk.get('summary', '')}"[:200],
                                    json.dumps(["consolidated", "archived", f"narrative_{narrative['id']}"]),
                                    chunk['chunk_id']
                                )
                            )
                        
                        self._report_progress(
                            "consolidation",
                            result.chunks_consolidated,
                            len(conv_chunks)
                        )
            
            result.clusters_found = len([c for c in clusters.values() if len(c) >= 3])
            
            await self.memory._db._connection.commit()
            
            logger.info(f"✅ Consolidation complete: {result.narratives_created} narratives from "
                       f"{result.chunks_consolidated} chunks")
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            raise
        
        return result
    
    async def _create_narrative(
        self,
        theme: str,
        chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a narrative memory from related chunks.
        
        In production, this would use the LLM to synthesize a coherent story.
        For now, we create a structured summary.
        """
        try:
            # Decompress all chunk contents
            contents = []
            for chunk in chunks:
                try:
                    content = self._decompress_content(chunk['content_compressed'])
                    contents.append(content)
                except Exception:
                    continue
            
            if not contents:
                return None
            
            # Create timeline from chunks
            timeline = []
            for i, chunk in enumerate(sorted(chunks, key=lambda c: c.get('created_at', ''))):
                timeline.append({
                    'sequence': i + 1,
                    'summary': chunk.get('summary', '')[:100],
                    'original_chunk': chunk['chunk_id'],
                })
            
            narrative_content = {
                'theme': theme,
                'type': 'narrative_memory',
                'consolidated_from': [c['chunk_id'] for c in chunks],
                'chunk_count': len(chunks),
                'timeline': timeline,
                'key_points': self._extract_key_points(contents),
                'created_at': datetime.now().isoformat(),
                'raw_contents_summary': f"Consolidated {len(contents)} conversation turns about {theme}",
            }
            
            # Store as new chunk
            narrative_id = await self.memory.store_in_ems(
                content=json.dumps(narrative_content, indent=2),
                chunk_type=MemoryChunkType.EPISODIC,  # Narrative/episodic memory
                importance=7.0,  # High importance for consolidated knowledge
                tags=['narrative', 'consolidated', theme, 'auto_generated'],
                summary=f"Narrative: {theme} ({len(chunks)} events)",
                load_hints=[theme, 'narrative', 'consolidated', 'story'],
                source=f"consolidation:{theme}",
            )
            
            return {
                'id': narrative_id,
                'theme': theme,
                'chunks_consolidated': len(chunks),
            }
            
        except Exception as e:
            logger.error(f"Failed to create narrative for {theme}: {e}")
            return None
    
    def _extract_key_points(self, contents: List[str]) -> List[str]:
        """Extract key points from multiple content strings."""
        # Simple extraction: find repeated phrases, important sentences
        all_text = ' '.join(contents)
        sentences = all_text.split('.')
        
        # Score by length and content markers
        scored = []
        for sent in sentences:
            score = 0
            if len(sent) > 20 and len(sent) < 200:
                score += 1
            if any(w in sent.lower() for w in ['important', 'key', 'main', 'decided', 'concluded']):
                score += 2
            if '?' in sent:
                score -= 1  # Questions less likely to be key points
            scored.append((sent.strip(), score))
        
        # Return top 5
        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:5] if len(s) > 10]
    
    def _decompress_content(self, compressed: bytes) -> str:
        """Decompress content bytes to string."""
        import zlib
        try:
            decompressed = zlib.decompress(compressed)
            return decompressed.decode('utf-8')
        except Exception:
            # Fallback: might be uncompressed
            try:
                return compressed.decode('utf-8')
            except:
                return "[decompression failed]"
    
    async def get_maintenance_summary(self) -> Dict[str, Any]:
        """Get summary of memory state and maintenance history."""
        stats = await self.memory.get_stats()
        
        return {
            'total_chunks': stats.get('active_chunks', 0),
            'archived_chunks': stats.get('archived_chunks', 0),
            'average_importance': stats.get('avg_importance', 0),
            'last_deduplication': self._last_dedup.isoformat() if self._last_dedup else None,
            'last_consolidation': self._last_consolidation.isoformat() if self._last_consolidation else None,
            'rcb_entries': stats.get('rcb_entries', 0),
        }
