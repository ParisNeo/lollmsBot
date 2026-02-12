"""
Document Manager - Hierarchical Document Analysis and Writing Support

Provides structured document handling for large-scale writing projects.
Integrates with RLM memory to enable REPL-style navigation of documents
that exceed context windows through hierarchical decomposition.

Architecture:
- Documents are decomposed into a tree: Document → Chapters → Sections → Blocks
- Each node is stored in RLM with [[MEMORY:...]] handles for navigation
- Semantic indexing enables cross-referencing and retrieval
- Writing operations maintain global awareness through structure injection
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Iterator, Callable
from collections import defaultdict

import logging

from lollmsbot.agent.rlm import RLMMemoryManager, MemoryChunk, MemoryChunkType, RCBEntry
from lollmsbot.agent.rlm.models import CompressionStats


logger = logging.getLogger(__name__)


class BlockType(Enum):
    """Types of document blocks in hierarchy."""
    DOCUMENT = auto()      # Root: book, report, etc.
    PART = auto()          # Major division (Part I, etc.)
    CHAPTER = auto()       # Chapter level
    SECTION = auto()       # Section/heading level
    SUBSECTION = auto()    # Subsection
    PARAGRAPH = auto()     # Paragraph (leaf node)
    TABLE = auto()         # Data table
    FIGURE = auto()        # Figure with caption
    CODE = auto()          # Code block
    QUOTE = auto()         # Block quote
    LIST = auto()          # List/enum


@dataclass
class DocumentBlock:
    """
    A node in the hierarchical document tree.
    
    Each block knows its position in the hierarchy and can generate
    contextual prompts showing its relationship to the whole.
    """
    # Identity
    block_id: str
    block_type: BlockType
    title: str = ""
    content: str = ""  # Raw text content
    
    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    sequence: int = 0  # Order among siblings
    
    # Metadata
    word_count: int = 0
    char_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    # Semantic
    summary: str = ""  # AI-generated summary
    key_concepts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    
    # RLM integration
    memory_chunk_id: Optional[str] = None
    
    # Writing state
    status: str = "draft"  # draft, review, final, locked
    notes: List[str] = field(default_factory=list)  # Editor notes
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "block_id": self.block_id,
            "block_type": self.block_type.name,
            "title": self.title,
            "content_preview": self.content[:200] if len(self.content) > 200 else self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "sequence": self.sequence,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "summary": self.summary,
            "key_concepts": self.key_concepts,
            "memory_chunk_id": self.memory_chunk_id,
            "status": self.status,
        }
    
    def get_path_fragment(self) -> str:
        """Get breadcrumb-style path for this block."""
        return f"{self.block_type.name.lower()}:{self.title or self.block_id[:8]}"
    
    def estimate_tokens(self) -> int:
        """Rough token estimate for this block's content."""
        # ~4 chars per token average
        return len(self.content) // 4


class DocumentIndex:
    """
    Semantic and structural index for a document.
    
    Enables fast lookup by:
    - Concept/entity mentions
    - Structural position
    - Content similarity
    """
    
    def __init__(self, document_id: str):
        self.document_id = document_id
        
        # Structural index: block_id -> block metadata
        self._blocks: Dict[str, DocumentBlock] = {}
        
        # Parent-child relationships
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._parents: Dict[str, str] = {}
        
        # Semantic indices
        self._concept_index: Dict[str, Set[str]] = defaultdict(set)  # concept -> block_ids
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)   # entity -> block_ids
        self._title_words: Dict[str, Set[str]] = defaultdict(set)    # word in title -> block_ids
        
        # Sequence order: level -> ordered block_ids
        self._sequence: Dict[BlockType, List[str]] = defaultdict(list)
        
        # Full-text search (simplified)
        self._word_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_block(self, block: DocumentBlock) -> None:
        """Index a new block."""
        self._blocks[block.block_id] = block
        
        # Hierarchy
        if block.parent_id:
            self._children[block.parent_id].append(block.block_id)
            self._parents[block.block_id] = block.parent_id
        
        # Sequence
        self._sequence[block.block_type].append(block.block_id)
        
        # Semantic indexing
        for concept in block.key_concepts:
            self._concept_index[concept.lower()].add(block.block_id)
        
        for entity in block.entities:
            self._entity_index[entity.lower()].add(block.block_id)
        
        # Title words
        for word in re.findall(r'\b\w+\b', block.title.lower()):
            if len(word) > 3:
                self._title_words[word].add(block.block_id)
        
        # Content words (first 100 words only for index)
        content_words = re.findall(r'\b\w+\b', block.content.lower())[:100]
        for word in set(content_words):
            if len(word) > 4:
                self._word_index[word].add(block.block_id)
    
    def get_block(self, block_id: str) -> Optional[DocumentBlock]:
        """Retrieve block by ID."""
        return self._blocks.get(block_id)
    
    def get_children(self, block_id: str) -> List[DocumentBlock]:
        """Get all children of a block."""
        child_ids = self._children.get(block_id, [])
        return [self._blocks[cid] for cid in child_ids if cid in self._blocks]
    
    def get_siblings(self, block_id: str) -> List[DocumentBlock]:
        """Get siblings (same parent) of a block."""
        parent_id = self._parents.get(block_id)
        if not parent_id:
            return []
        all_children = self._children.get(parent_id, [])
        sibling_ids = [cid for cid in all_children if cid != block_id]
        return [self._blocks[sid] for sid in sibling_ids if sid in self._blocks]
    
    def get_path(self, block_id: str) -> List[DocumentBlock]:
        """Get path from root to this block."""
        path = []
        current = block_id
        while current:
            block = self._blocks.get(current)
            if block:
                path.append(block)
                current = block.parent_id
            else:
                break
        return list(reversed(path))
    
    def find_by_concept(self, concept: str) -> List[DocumentBlock]:
        """Find blocks mentioning a concept."""
        block_ids = self._concept_index.get(concept.lower(), set())
        return [self._blocks[bid] for bid in block_ids if bid in self._blocks]
    
    def find_by_entity(self, entity: str) -> List[DocumentBlock]:
        """Find blocks mentioning an entity."""
        block_ids = self._entity_index.get(entity.lower(), set())
        return [self._blocks[bid] for bid in block_ids if bid in self._blocks]
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[DocumentBlock, float]]:
        """
        Simple keyword search with scoring.
        Returns (block, score) tuples sorted by score.
        """
        query_words = set(w.lower() for w in re.findall(r'\b\w+\b', query) if len(w) > 3)
        
        scores: Dict[str, float] = defaultdict(float)
        
        for word in query_words:
            # Title matches are high value
            for bid in self._title_words.get(word, []):
                scores[bid] += 3.0
            
            # Concept matches
            for bid in self._concept_index.get(word, []):
                scores[bid] += 2.5
            
            # Entity matches
            for bid in self._entity_index.get(word, []):
                scores[bid] += 2.0
            
            # Content matches
            for bid in self._word_index.get(word, []):
                scores[bid] += 1.0
        
        # Sort by score
        sorted_blocks = sorted(scores.items(), key=lambda x: -x[1])
        
        result = []
        for bid, score in sorted_blocks[:limit]:
            block = self._blocks.get(bid)
            if block:
                result.append((block, score))
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get document statistics."""
        return {
            "total_blocks": len(self._blocks),
            "by_type": {t.name: len(ids) for t, ids in self._sequence.items()},
            "total_word_estimate": sum(b.word_count for b in self._blocks.values()),
            "indexed_concepts": len(self._concept_index),
            "indexed_entities": len(self._entity_index),
        }


class DocumentManager:
    """
    Manages large documents through hierarchical decomposition and RLM integration.
    
    Enables the agent to:
    - Ingest documents (URLs, files, text) and create hierarchical structure
    - Navigate documents via [[MEMORY:...]] handles
    - Zoom in/out of detail levels
    - Maintain global awareness during local operations
    - Search and retrieve relevant sections
    """
    
    # Chunk sizes for different context budgets
    BUDGETS = {
        "micro": 500,      # ~125 tokens - minimal context
        "small": 2000,     # ~500 tokens - section summaries
        "medium": 8000,    # ~2000 tokens - chapter context  
        "large": 32000,    # ~8000 tokens - multiple chapters
        "full": 128000,    # ~32000 tokens - near-full document
    }
    
    def __init__(
        self,
        memory_manager: RLMMemoryManager,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._memory = memory_manager
        self._data_dir = data_dir or (Path.home() / ".lollmsbot" / "documents")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # Active documents
        self._documents: Dict[str, DocumentIndex] = {}
        self._document_roots: Dict[str, str] = {}  # doc_id -> root_block_id
        
        # Book projects
        self._book_projects: Dict[str, BookProject] = {}
        
        # Statistics
        self._stats: Dict[str, Any] = {}
    
    async def ingest_webpage(
        self,
        url: str,
        title: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> str:
        """
        Ingest a webpage and create hierarchical structure.
        
        Uses the HTTP tool to fetch, then analyzes structure to create
        a document tree stored in RLM.
        """
        doc_id = document_id or f"web_{hashlib.sha256(url.encode()).hexdigest()[:12]}"
        
        logger.info(f"Ingesting webpage: {url} as {doc_id}")
        
        # Store source in RLM first (via HTTP tool would do this)
        # Here we create the hierarchical structure
        
        # For now, create a simple structure - in production this would
        # use the HTTP tool's result and parse HTML
        
        # Create document root
        root = DocumentBlock(
            block_id=f"{doc_id}_root",
            block_type=BlockType.DOCUMENT,
            title=title or f"Document from {url}",
            content=f"Source: {url}\nIngested: {datetime.now().isoformat()}",
        )
        
        # Store in RLM with high importance (reference material)
        root.memory_chunk_id = await self._store_block(root, doc_id, importance=9.0)
        
        # Create index
        index = DocumentIndex(doc_id)
        index.add_block(root)
        
        self._documents[doc_id] = index
        self._document_roots[doc_id] = root.block_id
        
        # In production: parse content, extract structure, create children
        
        logger.info(f"Document {doc_id} ingested with root block {root.memory_chunk_id}")
        
        return doc_id
    
    async def ingest_text(
        self,
        content: str,
        title: str = "Untitled Document",
        source_hint: str = "user_upload",
        document_id: Optional[str] = None,
    ) -> str:
        """
        Ingest raw text and create hierarchical structure.
        
        Analyzes text structure (paragraphs, sections marked by headers)
        to build document tree.
        """
        doc_id = document_id or f"txt_{hashlib.sha256(content[:1000].encode()).hexdigest()[:12]}"
        
        logger.info(f"Ingesting text document: {title} as {doc_id}")
        
        # Parse structure
        blocks = self._parse_text_structure(content, doc_id)
        
        # Create index
        index = DocumentIndex(doc_id)
        
        # Store blocks in RLM and index
        for block in blocks:
            block.memory_chunk_id = await self._store_block(
                block, 
                doc_id, 
                importance=8.0 if block.block_type in (BlockType.DOCUMENT, BlockType.CHAPTER) else 7.0
            )
            index.add_block(block)
        
        self._documents[doc_id] = index
        self._document_roots[doc_id] = next(
            b.block_id for b in blocks if b.block_type == BlockType.DOCUMENT
        )
        
        stats = index.get_statistics()
        logger.info(f"Document {doc_id} ingested: {stats['total_blocks']} blocks, ~{stats['total_word_estimate']} words")
        
        return doc_id
    
    def _parse_text_structure(self, content: str, doc_id: str) -> List[DocumentBlock]:
        """
        Parse text into hierarchical blocks.
        
        Detects structure from:
        - Markdown headers (# ## ###)
        - LaTeX headers (\chapter{}, \section{})
        - Indentation patterns
        - Blank line separation
        """
        blocks: List[DocumentBlock] = []
        
        # Create root
        root_id = f"{doc_id}_root"
        root = DocumentBlock(
            block_id=root_id,
            block_type=BlockType.DOCUMENT,
            title="Document",  # Will be updated if h1 found
        )
        blocks.append(root)
        
        # Current parent stack: [root_id, chapter_id, section_id, ...]
        parent_stack = [root_id]
        current_level = 0  # 0=document, 1=part/chapter, 2=section, 3=subsection
        
        lines = content.split('\n')
        i = 0
        seq_counters: Dict[int, int] = defaultdict(int)
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check for headers
            level, title = self._detect_header(line)
            
            if level > 0:
                # It's a header - create new block
                block_type = self._level_to_block_type(level)
                
                # Pop stack to correct level
                while len(parent_stack) > level:
                    parent_stack.pop()
                
                # Create block
                seq_counters[level] += 1
                block_id = f"{doc_id}_{block_type.name.lower()}_{seq_counters[level]}"
                
                block = DocumentBlock(
                    block_id=block_id,
                    block_type=block_type,
                    title=title,
                    parent_id=parent_stack[-1] if parent_stack else root_id,
                    sequence=seq_counters[level],
                )
                
                # Add to parent's children
                parent_block = next((b for b in blocks if b.block_id == block.parent_id), None)
                if parent_block:
                    parent_block.children_ids.append(block_id)
                
                # Push to stack
                parent_stack.append(block_id)
                current_level = level
                
                blocks.append(block)
                i += 1
                
            elif stripped:
                # Content line - add to current leaf
                # Collect paragraph
                para_lines = [stripped]
                i += 1
                
                while i < len(lines) and lines[i].strip():
                    para_lines.append(lines[i].strip())
                    i += 1
                
                # Skip blank lines
                while i < len(lines) and not lines[i].strip():
                    i += 1
                
                # Create paragraph block
                para_content = ' '.join(para_lines)
                para_id = f"{doc_id}_para_{len([b for b in blocks if b.block_type == BlockType.PARAGRAPH])}"
                
                para = DocumentBlock(
                    block_id=para_id,
                    block_type=BlockType.PARAGRAPH,
                    content=para_content,
                    parent_id=parent_stack[-1] if parent_stack else root_id,
                    word_count=len(para_content.split()),
                    char_count=len(para_content),
                )
                
                # Add to parent's children
                parent = next((b for b in blocks if b.block_id == para.parent_id), None)
                if parent:
                    parent.children_ids.append(para_id)
                
                blocks.append(para)
                
            else:
                # Blank line
                i += 1
        
        return blocks
    
    def _detect_header(self, line: str) -> Tuple[int, str]:
        """Detect if line is a header and return (level, title)."""
        # Markdown headers
        md_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if md_match:
            return len(md_match.group(1)), md_match.group(2).strip()
        
        # LaTeX headers
        latex_match = re.match(r'^\\(part|chapter|section|subsection|subsubsection)\*?\{(.+)\}$', line)
        if latex_match:
            cmd = latex_match.group(1)
            title = latex_match.group(2)
            level_map = {
                'part': 1,
                'chapter': 1,
                'section': 2,
                'subsection': 3,
                'subsubsection': 4,
            }
            return level_map.get(cmd, 2), title
        
        # Underline headers (===, ---)
        if line and all(c == '=' for c in line.strip()):
            return 1, ""  # Title on previous line
        if line and all(c == '-' for c in line.strip()):
            return 2, ""
        
        return 0, ""
    
    def _level_to_block_type(self, level: int) -> BlockType:
        """Convert header level to block type."""
        mapping = {
            1: BlockType.CHAPTER,
            2: BlockType.SECTION,
            3: BlockType.SUBSECTION,
            4: BlockType.SUBSECTION,
            5: BlockType.SUBSECTION,
            6: BlockType.SUBSECTION,
        }
        return mapping.get(level, BlockType.SECTION)
    
    async def _store_block(self, block: DocumentBlock, doc_id: str, importance: float = 7.0) -> str:
        """Store block in RLM memory."""
        # Build content with structural metadata
        structural_context = self._build_structural_context(block, doc_id)
        
        full_content = f"""{structural_context}

TITLE: {block.title}
TYPE: {block.block_type.name}
WORD_COUNT: {block.word_count}

CONTENT:
{block.content}

SUMMARY: {block.summary}
CONCEPTS: {', '.join(block.key_concepts)}
ENTITIES: {', '.join(block.entities)}
"""
        
        # Store in RLM
        chunk_id = await self._memory.store_in_ems(
            content=full_content,
            chunk_type=MemoryChunkType.FACT,  # Reference material
            importance=importance,
            tags=["document", "block", block.block_type.name, doc_id],
            summary=f"[{block.block_type.name}] {block.title or 'Untitled'} ({block.word_count} words)",
            load_hints=[block.block_id, doc_id, block.title, doc_id] + block.key_concepts,
            source=f"document:{doc_id}:block:{block.block_id}",
        )
        
        return chunk_id
    
    def _build_structural_context(self, block: DocumentBlock, doc_id: str) -> str:
        """Build context string showing position in hierarchy."""
        # Get path from root
        # This would use the index, but for initial storage we don't have full index yet
        # So we build minimal context
        
        context_parts = [f"DOCUMENT: {doc_id}"]
        
        if block.parent_id:
            context_parts.append(f"PARENT: {block.parent_id}")
        
        return " | ".join(context_parts)
    
    async def get_context_lens(
        self,
        document_id: str,
        focus_block_id: Optional[str] = None,
        budget: str = "medium",
        include_surroundings: bool = True,
    ) -> ContextLens:
        """
        Get a zoomable view of document context.
        
        Creates a ContextLens that provides:
        - Global structure awareness
        - Local detail at focus point
        - Navigation handles to related blocks
        
        Args:
            document_id: Document to view
            focus_block_id: Specific block to focus on (zoom center)
            budget: Context size budget ("micro", "small", "medium", "large", "full")
            include_surroundings: Include adjacent blocks in context
        
        Returns:
            ContextLens ready for LLM consumption
        """
        index = self._documents.get(document_id)
        if not index:
            raise ValueError(f"Document {document_id} not found")
        
        max_tokens = self.BUDGETS.get(budget, self.BUDGETS["medium"])
        
        # Build context
        lens = ContextLens(
            document_id=document_id,
            focus_block_id=focus_block_id,
            budget_tokens=max_tokens,
            index=index,
        )
        
        # Fill lens with content
        await lens._build()
        
        return lens
    
    async def create_book_project(
        self,
        title: str,
        author: Optional[str] = None,
        description: Optional[str] = None,
        references: List[str] = None,  # Document IDs to use as sources
    ) -> str:
        """
        Create a new book writing project.
        
        Initializes a BookProject with hierarchical structure and
        connects to reference documents.
        """
        project_id = f"book_{hashlib.sha256(title.encode()).hexdigest()[:12]}_{int(datetime.now().timestamp())}"
        
        logger.info(f"Creating book project: {title} as {project_id}")
        
        # Create book structure
        book = BookProject(
            project_id=project_id,
            title=title,
            author=author,
            description=description,
            document_manager=self,
            reference_documents=references or [],
        )
        
        # Initialize with empty structure
        root = DocumentBlock(
            block_id=f"{project_id}_book",
            block_type=BlockType.DOCUMENT,
            title=title,
            content=f"Book: {title}\nAuthor: {author or 'TBD'}\nDescription: {description or 'TBD'}",
        )
        root.memory_chunk_id = await self._store_block(root, project_id, importance=10.0)
        
        # Create project index
        index = DocumentIndex(project_id)
        index.add_block(root)
        
        self._documents[project_id] = index
        self._document_roots[project_id] = root.block_id
        self._book_projects[project_id] = book
        
        # Connect references
        for ref_doc_id in (references or []):
            if ref_doc_id in self._documents:
                book.reference_indices.append(self._documents[ref_doc_id])
                logger.info(f"Connected reference: {ref_doc_id}")
        
        return project_id
    
    def get_book_project(self, project_id: str) -> Optional[BookProject]:
        """Retrieve a book project."""
        return self._book_projects.get(project_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all loaded documents."""
        result = []
        for doc_id, index in self._documents.items():
            stats = index.get_statistics()
            root = self._document_roots.get(doc_id)
            root_block = index.get_block(root) if root else None
            
            result.append({
                "document_id": doc_id,
                "title": root_block.title if root_block else "Unknown",
                "type": "book_project" if doc_id in self._book_projects else "reference",
                "blocks": stats["total_blocks"],
                "words": stats["total_word_estimate"],
                "concepts": stats["indexed_concepts"],
            })
        
        return result
    
    def search_across_documents(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        limit_per_doc: int = 5,
    ) -> Dict[str, List[Tuple[DocumentBlock, float]]]:
        """
        Search across multiple documents.
        
        Returns results grouped by document.
        """
        results: Dict[str, List[Tuple[DocumentBlock, float]]] = {}
        
        docs_to_search = document_ids or list(self._documents.keys())
        
        for doc_id in docs_to_search:
            index = self._documents.get(doc_id)
            if index:
                results[doc_id] = index.search(query, limit_per_doc)
        
        return results


class ContextLens:
    """
    A "view" into a document with specific focus and zoom level.
    
    Provides:
    - Global structure summary (always included)
    - Local detail at focus point (configurable depth)
    - Navigation handles to jump to related sections
    
    This is what the LLM actually "sees" when working on a document.
    """
    
    def __init__(
        self,
        document_id: str,
        focus_block_id: Optional[str],
        budget_tokens: int,
        index: DocumentIndex,
    ):
        self.document_id = document_id
        self.focus_block_id = focus_block_id
        self.budget_tokens = budget_tokens
        self.index = index
        
        self.global_summary: str = ""
        self.local_detail: str = ""
        self.navigation_handles: List[str] = []  # [[MEMORY:...]] strings
        self.current_tokens: int = 0
        
        # Tracking for RCB integration
        self._loaded_chunk_ids: Set[str] = set()
    
    async def _build(self) -> None:
        """Build the lens content."""
        # Always include global structure
        self._add_global_structure()
        
        # If no focus, we're done (overview mode)
        if not self.focus_block_id:
            return
        
        # Add focused content
        focus_block = self.index.get_block(self.focus_block_id)
        if not focus_block:
            return
        
        self._add_focus_block(focus_block)
        
        # Add surroundings if budget allows
        remaining = self.budget_tokens - self.current_tokens
        if remaining > self.index.get_block(self.focus_block_id).estimate_tokens() * 0.5:
            self._add_surroundings(focus_block)
        
        # Add related content via search
        remaining = self.budget_tokens - self.current_tokens
        if remaining > 500:
            self._add_related_content(focus_block)
    
    def _add_global_structure(self) -> None:
        """Add document structure summary."""
        stats = self.index.get_statistics()
        
        # Build TOC from sequence
        toc_lines = ["STRUCTURE OVERVIEW:"]
        
        for block_type in [BlockType.PART, BlockType.CHAPTER, BlockType.SECTION]:
            blocks = self.index._sequence.get(block_type, [])
            if blocks:
                toc_lines.append(f"\n{block_type.name.title()}s ({len(blocks)}):")
                for bid in blocks[:20]:  # Limit
                    block = self.index.get_block(bid)
                    if block:
                        status_emoji = "✓" if block.status == "final" else "○"
                        toc_lines.append(f"  {status_emoji} {block.title or 'Untitled'}")
        
        self.global_summary = '\n'.join(toc_lines)
        self.current_tokens += len(self.global_summary) // 4  # Rough estimate
    
    def _add_focus_block(self, block: DocumentBlock) -> None:
        """Add detailed content for focused block."""
        # Get path for context
        path = self.index.get_path(block.block_id)
        path_str = " > ".join(b.title or b.block_id[:8] for b in path[:-1])  # Exclude self
        
        detail_lines = [
            f"\n{'='*60}",
            f"FOCUS: {block.block_type.name} - {block.title or 'Untitled'}",
            f"PATH: {path_str}",
            f"STATUS: {block.status}",
            f"WORDS: {block.word_count}",
            f"{'='*60}",
        ]
        
        if block.summary:
            detail_lines.extend([f"\nSUMMARY: {block.summary}"])
        
        if block.notes:
            detail_lines.extend([f"\nEDITOR NOTES:"])
            for note in block.notes:
                detail_lines.append(f"  • {note}")
        
        # Full content
        detail_lines.extend([
            f"\nCONTENT:",
            block.content[:10000] if len(block.content) > 10000 else block.content,  # Cap single block
        ])
        
        self.local_detail = '\n'.join(detail_lines)
        self.current_tokens += len(self.local_detail) // 4
        
        # Add memory handle for this block
        if block.memory_chunk_id:
            self.navigation_handles.append(f"[[MEMORY:{block.memory_chunk_id}]]")
            self._loaded_chunk_ids.add(block.memory_chunk_id)
    
    def _add_surroundings(self, block: DocumentBlock) -> None:
        """Add preceding and following blocks."""
        # Get siblings
        siblings = self.index.get_siblings(block.block_id)
        
        # Find position
        try:
            all_siblings = sorted(siblings + [block], key=lambda b: b.sequence)
            idx = next(i for i, b in enumerate(all_siblings) if b.block_id == block.block_id)
        except StopIteration:
            return
        
        # Add previous
        if idx > 0:
            prev = all_siblings[idx - 1]
            prev_text = f"\n← PREVIOUS: {prev.title or prev.block_id[:8]}\n{prev.content[:500]}..."
            self.local_detail += prev_text
            self.current_tokens += len(prev_text) // 4
        
        # Add next
        if idx < len(all_siblings) - 1:
            next_b = all_siblings[idx + 1]
            next_text = f"\n→ NEXT: {next_b.title or next_b.block_id[:8]}\n{next_b.content[:500]}..."
            self.local_detail += next_text
            self.current_tokens += len(next_text) // 4
    
    def _add_related_content(self, block: DocumentBlock) -> None:
        """Add semantically related content via search."""
        # Search for similar content
        query = ' '.join(block.key_concepts[:3]) if block.key_concepts else block.content[:200]
        related = self.index.search(query, limit=3)
        
        if related:
            self.local_detail += "\n\nRELATED SECTIONS:"
            for rel_block, score in related[:2]:  # Top 2
                if rel_block.block_id != block.block_id:
                    rel_text = f"\n  ↳ {rel_block.title or rel_block.block_id[:8]} (relevance: {score:.1f})\n     {rel_block.summary or rel_block.content[:200]}..."
                    self.local_detail += rel_text
    
    def to_prompt_fragment(self) -> str:
        """Convert lens to text for LLM prompt."""
        parts = [
            f"DOCUMENT CONTEXT: {self.document_id}",
            f"TOKEN BUDGET: {self.current_tokens}/{self.budget_tokens}",
            "",
            self.global_summary,
            "",
            self.local_detail,
        ]
        
        if self.navigation_handles:
            parts.extend([
                "",
                "NAVIGATION:",
                "Use these handles to access other sections:",
            ])
            for handle in self.navigation_handles:
                parts.append(f"  {handle}")
        
        return '\n'.join(parts)
    
    def get_rcb_entries(self) -> List[RCBEntry]:
        """Get RCB entries for this lens."""
        entries = []
        
        # Add structure as system context
        entries.append(RCBEntry(
            entry_type="document_structure",
            content=self.global_summary,
            display_order=0,
        ))
        
        # Add focus content
        if self.focus_block_id:
            focus = self.index.get_block(self.focus_block_id)
            if focus and focus.memory_chunk_id:
                entries.append(RCBEntry(
                    entry_type="document_focus",
                    content=self.local_detail,
                    chunk_id=focus.memory_chunk_id,
                    display_order=1,
                ))
        
        return entries


@dataclass
class WritingTask:
    """A specific writing task within a book project."""
    task_id: str
    task_type: str  # "outline", "chapter", "section", "revise", "polish"
    target_block_id: Optional[str] = None
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    reference_queries: List[str] = field(default_factory=list)  # What to search for
    
    # State
    status: str = "pending"  # pending, active, completed, failed
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output_block_id: Optional[str] = None
    
    # Iteration
    revision_count: int = 0
    max_revisions: int = 3
    last_feedback: str = ""


class BookProject:
    """
    A book writing project with hierarchical workflow support.
    
    Manages:
    - Document structure (chapters, sections)
    - Reference materials and their integration
    - Writing tasks and their dependencies
    - Global consistency checks
    """
    
    def __init__(
        self,
        project_id: str,
        title: str,
        author: Optional[str],
        description: Optional[str],
        document_manager: DocumentManager,
        reference_documents: List[str] = None,
    ):
        self.project_id = project_id
        self.title = title
        self.author = author
        self.description = description
        self._doc_mgr = document_manager
        
        # References (other documents we can draw from)
        self.reference_documents: List[str] = reference_documents or []
        self.reference_indices: List[DocumentIndex] = []
        
        # Writing tasks
        self.tasks: Dict[str, WritingTask] = {}
        self.task_queue: List[str] = []
        
        # Global state
        self.themes: List[str] = []
        self.characters: List[Dict[str, Any]] = field(default_factory=list)
        self.worldbuilding: Dict[str, Any] = field(default_factory=dict)
        self.plot_points: List[Dict[str, Any]] = field(default_factory=list)
        
        # Revision history
        self.revisions: List[Dict[str, Any]] = field(default_factory=list)
    
    async def create_outline(
        self,
        num_chapters: int,
        chapter_titles: Optional[List[str]] = None,
        synopsis: Optional[str] = None,
    ) -> str:
        """
        Create book outline structure.
        
        Creates chapter and section blocks, ready for content.
        Returns task_id for tracking.
        """
        task_id = f"{self.project_id}_outline_{len(self.tasks)}"
        
        task = WritingTask(
            task_id=task_id,
            task_type="outline",
            description=f"Create {num_chapters}-chapter outline for '{self.title}'",
        )
        
        # In production: this would call the LLM to generate outline
        # based on synopsis and reference materials
        
        # For now, create structure
        root_id = self._doc_mgr._document_roots.get(self.project_id)
        if not root_id:
            raise ValueError("Book root not found")
        
        root = self._doc_mgr._documents[self.project_id].get_block(root_id)
        
        # Create chapters
        for i in range(1, num_chapters + 1):
            title = (chapter_titles[i-1] if chapter_titles and i <= len(chapter_titles) 
                    else f"Chapter {i}: [Title TBD]")
            
            chap_id = f"{self.project_id}_ch{i}"
            chap = DocumentBlock(
                block_id=chap_id,
                block_type=BlockType.CHAPTER,
                title=title,
                parent_id=root_id,
                sequence=i,
                status="draft",
            )
            
            chap.memory_chunk_id = await self._doc_mgr._store_block(
                chap, self.project_id, importance=9.5
            )
            
            self._doc_mgr._documents[self.project_id].add_block(chap)
            root.children_ids.append(chap_id)
            
            # Add placeholder sections
            for j, section_title in enumerate(["Introduction", "Development", "Conclusion"], 1):
                sect_id = f"{chap_id}_s{j}"
                sect = DocumentBlock(
                    block_id=sect_id,
                    block_type=BlockType.SECTION,
                    title=section_title,
                    parent_id=chap_id,
                    sequence=j,
                )
                sect.memory_chunk_id = await self._doc_mgr._store_block(
                    sect, self.project_id, importance=8.5
                )
                self._doc_mgr._documents[self.project_id].add_block(sect)
                chap.children_ids.append(sect_id)
        
        task.status = "completed"
        self.tasks[task_id] = task
        
        return task_id
    
    async def write_chapter(
        self,
        chapter_number: int,
        target_length_words: int = 3000,
        style_guidance: Optional[str] = None,
    ) -> str:
        """
        Create writing task for a chapter.
        
        Sets up context lens and returns task for execution.
        """
        chap_id = f"{self.project_id}_ch{chapter_number}"
        chapter = self._doc_mgr._documents[self.project_id].get_block(chap_id)
        
        if not chapter:
            raise ValueError(f"Chapter {chapter_number} not found")
        
        task_id = f"{self.project_id}_write_ch{chapter_number}"
        
        # Build requirements from global state
        requirements = [
            f"Target length: ~{target_length_words} words",
            f"Chapter title: {chapter.title}",
        ]
        
        # Add continuity requirements
        if chapter_number > 1:
            prev_chap = self._doc_mgr._documents[self.project_id].get_block(
                f"{self.project_id}_ch{chapter_number - 1}"
            )
            if prev_chap:
                requirements.append(f"Continue from: {prev_chap.title}")
        
        if chapter_number < len(chapter.parent_id or []):  # Simplified
            requirements.append(f"Setup for next chapter")
        
        # Add style guidance
        if style_guidance:
            requirements.append(f"Style: {style_guidance}")
        
        # Build reference queries from chapter title/content
        ref_queries = [chapter.title]
        # In production: extract themes, character names, etc.
        
        task = WritingTask(
            task_id=task_id,
            task_type="chapter",
            target_block_id=chap_id,
            description=f"Write Chapter {chapter_number}: {chapter.title}",
            requirements=requirements,
            reference_queries=ref_queries,
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        return task_id
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a writing task with full context setup.
        
        This is called by the agent to perform the actual writing.
        Returns the generated content and updates task state.
        """
        task = self.tasks.get(task_id)
        if not task:
            return {"success": False, "error": "Task not found"}
        
        task.status = "active"
        task.assigned_at = datetime.now()
        
        # Get context lens for this task
        lens = await self._doc_mgr.get_context_lens(
            document_id=self.project_id,
            focus_block_id=task.target_block_id,
            budget="large",  # Use large context for writing
            include_surroundings=True,
        )
        
        # Gather reference material
        ref_content = []
        for query in task.reference_queries:
            results = self._doc_mgr.search_across_documents(
                query=query,
                document_ids=self.reference_documents,
                limit_per_doc=2,
            )
            for doc_id, matches in results.items():
                for block, score in matches:
                    ref_content.append(f"[From {doc_id}: {block.title or 'ref'}]\n{block.summary or block.content[:500]}")
        
        # Build execution prompt
        execution_context = {
            "task": task,
            "document_context": lens.to_prompt_fragment(),
            "references": ref_content,
            "project_themes": self.themes,
            "continuity_requirements": self._get_continuity_requirements(task),
        }
        
        # This would be passed to the LLM
        # For now, return the context structure
        return {
            "success": True,
            "task_id": task_id,
            "context_ready": True,
            "context_tokens_estimate": lens.current_tokens,
            "execution_prompt": execution_context,
            "instruction": (
                f"Write {task.task_type} content for: {task.description}\n"
                f"Requirements: {'; '.join(task.requirements)}\n"
                f"Use the provided DOCUMENT CONTEXT to maintain consistency.\n"
                f"REFERENCE MATERIAL is available for factual support."
            ),
        }
    
    def _get_continuity_requirements(self, task: WritingTask) -> List[str]:
        """Build continuity requirements from project state."""
        requirements = []
        
        # Check what comes before
        if task.target_block_id:
            block = self._doc_mgr._documents[self.project_id].get_block(task.target_block_id)
            if block:
                path = self._doc_mgr._documents[self.project_id].get_path(block.block_id)
                if len(path) > 1:
                    parent = path[-2] if len(path) >= 2 else None
                    if parent:
                        requirements.append(f"Must align with parent: {parent.title}")
        
        # Add thematic requirements
        if self.themes:
            requirements.append(f"Themes to maintain: {', '.join(self.themes[:3])}")
        
        return requirements
    
    def submit_written_content(
        self,
        task_id: str,
        content: str,
        word_count: int,
    ) -> str:
        """
        Submit completed writing and update document structure.
        
        Creates paragraph blocks for the content and updates task.
        """
        task = self.tasks.get(task_id)
        if not task or not task.target_block_id:
            raise ValueError("Invalid task")
        
        target = self._doc_mgr._documents[self.project_id].get_block(task.target_block_id)
        if not target:
            raise ValueError("Target block not found")
        
        # Parse content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Create paragraph blocks
        created_ids = []
        for i, para_text in enumerate(paragraphs, 1):
            para_id = f"{task.target_block_id}_p{i}"
            para = DocumentBlock(
                block_id=para_id,
                block_type=BlockType.PARAGRAPH,
                content=para_text,
                parent_id=task.target_block_id,
                sequence=i,
                word_count=len(para_text.split()),
                char_count=len(para_text),
                status="draft",
            )
            
            # Store (async would be better but we need sync here)
            # In production: use asyncio.run_coroutine_threadsafe or similar
            
            # Update parent
            target.children_ids.append(para_id)
            
            created_ids.append(para_id)
        
        # Update task
        task.status = "completed"
        task.completed_at = datetime.now()
        task.output_block_id = task.target_block_id  # The chapter now has content
        
        # Record revision
        self.revisions.append({
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "word_count": word_count,
            "paragraphs": len(paragraphs),
        })
        
        return task.target_block_id
    
    def get_progress(self) -> Dict[str, Any]:
        """Get writing progress statistics."""
        all_tasks = list(self.tasks.values())
        
        return {
            "project_id": self.project_id,
            "title": self.title,
            "total_tasks": len(all_tasks),
            "completed": len([t for t in all_tasks if t.status == "completed"]),
            "in_progress": len([t for t in all_tasks if t.status == "active"]),
            "pending": len([t for t in all_tasks if t.status == "pending"]),
            "total_words_written": sum(
                r.get("word_count", 0) for r in self.revisions
            ),
            "chapters_started": len([t for t in all_tasks if t.task_type == "chapter"]),
            "chapters_completed": len([
                t for t in all_tasks 
                if t.task_type == "chapter" and t.status == "completed"
            ]),
        }


async def create_document_manager(
    memory_manager: RLMMemoryManager,
    data_dir: Optional[Path] = None,
) -> DocumentManager:
    """Factory for creating configured DocumentManager."""
    return DocumentManager(
        memory_manager=memory_manager,
        data_dir=data_dir,
    )
