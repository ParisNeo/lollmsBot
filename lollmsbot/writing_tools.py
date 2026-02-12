"""
Writing Tools - Specialized tools for document creation and editing.

Extends the base tool system with document-aware operations that
integrate with DocumentManager for hierarchical writing support.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lollmsbot.agent import Tool, ToolResult
from lollmsbot.document_manager import (
    DocumentManager, 
    BookProject, 
    WritingTask,
    ContextLens,
    BlockType,
    DocumentBlock,
)


class IngestDocumentTool(Tool):
    """
    Tool for ingesting external documents into the writing workspace.
    
    Creates hierarchical structure and makes content available via
    memory handles for reference during writing.
    """
    
    name: str = "ingest_document"
    description: str = (
        "Ingest a document (webpage URL, file path, or raw text) and make it "
        "available as structured, searchable reference material. "
        "Large documents are automatically chunked and indexed for retrieval. "
        "Returns a document handle for use in writing tasks."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "source_type": {
                "type": "string",
                "enum": ["url", "file", "text"],
                "description": "Type of source to ingest",
            },
            "source": {
                "type": "string",
                "description": "URL, file path, or raw text content",
            },
            "title": {
                "type": "string",
                "description": "Preferred title for the document",
            },
            "document_id": {
                "type": "string",
                "description": "Optional custom ID (auto-generated if not provided)",
            },
        },
        "required": ["source_type", "source"],
    }
    
    def __init__(self, document_manager: DocumentManager):
        self._doc_mgr = document_manager
    
    async def execute(
        self,
        source_type: str,
        source: str,
        title: Optional[str] = None,
        document_id: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute document ingestion."""
        try:
            if source_type == "url":
                doc_id = await self._doc_mgr.ingest_webpage(
                    url=source,
                    title=title,
                    document_id=document_id,
                )
                
            elif source_type == "text":
                doc_id = await self._doc_mgr.ingest_text(
                    content=source,
                    title=title or "Untitled Document",
                    document_id=document_id,
                )
                
            else:
                # file type - would read and parse
                return ToolResult(
                    success=False,
                    output=None,
                    error="File ingestion not yet implemented - use 'text' with file content",
                )
            
            # Get document info
            stats = self._doc_mgr._documents.get(doc_id, {}).get_statistics() if hasattr(
                self._doc_mgr._documents.get(doc_id), 'get_statistics'
            ) else {}
            
            return ToolResult(
                success=True,
                output={
                    "document_id": doc_id,
                    "title": title or "Untitled",
                    "blocks": stats.get("total_blocks", 0),
                    "words_estimate": stats.get("total_word_estimate", 0),
                    "access_handle": f"[[DOCUMENT:{doc_id}]]",
                    "usage": (
                        f"Document ingested. Use context lens to view: "
                        f"get_context_lens(document_id='{doc_id}', budget='medium')"
                    ),
                },
                error=None,
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Ingestion failed: {str(e)}",
            )


class CreateBookProjectTool(Tool):
    """
    Tool for initializing a new book writing project.
    
    Sets up hierarchical structure and connects reference materials.
    """
    
    name: str = "create_book_project"
    description: str = (
        "Create a new book writing project with title, author, and optional references. "
        "References are existing document IDs that will be available for research "
        "during writing. Returns a project ID for use with other writing tools."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Book title",
            },
            "author": {
                "type": "string",
                "description": "Author name (can be 'TBD')",
            },
            "description": {
                "type": "string",
                "description": "Brief description or premise",
            },
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Document IDs to use as reference materials",
            },
        },
        "required": ["title"],
    }
    
    def __init__(self, document_manager: DocumentManager):
        self._doc_mgr = document_manager
    
    async def execute(
        self,
        title: str,
        author: Optional[str] = None,
        description: Optional[str] = None,
        references: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolResult:
        """Create book project."""
        try:
            project_id = await self._doc_mgr.create_book_project(
                title=title,
                author=author,
                description=description,
                references=references or [],
            )
            
            return ToolResult(
                success=True,
                output={
                    "project_id": project_id,
                    "title": title,
                    "status": "created",
                    "references_connected": len(references or []),
                    "next_steps": [
                        "Use create_outline to structure the book",
                        "Use write_section to generate content",
                        "Use get_document_context to view structure",
                    ],
                    "access_handle": f"[[BOOK:{project_id}]]",
                },
                error=None,
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Project creation failed: {str(e)}",
            )


class CreateOutlineTool(Tool):
    """Tool for creating book/chapter outlines."""
    
    name: str = "create_outline"
    description: str = (
        "Create hierarchical outline for a book or chapter. "
        "Generates chapter structure with sections ready for content. "
        "Can be based on synopsis or created from scratch."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "Book project ID",
            },
            "num_chapters": {
                "type": "integer",
                "description": "Number of chapters",
                "minimum": 1,
                "maximum": 100,
            },
            "synopsis": {
                "type": "string",
                "description": "Overall book synopsis for guidance",
            },
            "chapter_titles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional specific chapter titles",
            },
        },
        "required": ["project_id", "num_chapters"],
    }
    
    def __init__(self, document_manager: DocumentManager):
        self._doc_mgr = document_manager
    
    async def execute(
        self,
        project_id: str,
        num_chapters: int,
        synopsis: Optional[str] = None,
        chapter_titles: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolResult:
        """Create outline."""
        project = self._doc_mgr.get_book_project(project_id)
        
        if not project:
            return ToolResult(
                success=False,
                output=None,
                error=f"Project {project_id} not found",
            )
        
        try:
            task_id = await project.create_outline(
                num_chapters=num_chapters,
                chapter_titles=chapter_titles,
                synopsis=synopsis,
            )
            
            # Get updated structure
            progress = project.get_progress()
            
            return ToolResult(
                success=True,
                output={
                    "task_id": task_id,
                    "project_id": project_id,
                    "chapters_created": num_chapters,
                    "progress": progress,
                    "instruction": (
                        "Outline created. Use write_section to generate content for chapters, "
                        "or get_document_context to review structure."
                    ),
                },
                error=None,
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Outline creation failed: {str(e)}",
            )


class GetDocumentContextTool(Tool):
    """
    Tool for retrieving hierarchical document context.
    
    Provides "zoomable" views of documents at appropriate detail levels
    for the task at hand.
    """
    
    name: str = "get_document_context"
    description: str = (
        "Get structured context from a document or book project. "
        "Returns hierarchical view with focus on specific sections if desired. "
        "Automatically manages context budget to fit available tokens. "
        "Use this to understand document structure before writing or editing."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "Document or project ID",
            },
            "focus": {
                "type": "string",
                "description": "Specific chapter, section, or block to focus on (optional)",
            },
            "detail_level": {
                "type": "string",
                "enum": ["overview", "summary", "detailed", "full"],
                "description": "How much detail to include",
            },
            "include_references": {
                "type": "boolean",
                "description": "Include relevant reference material",
                "default": True,
            },
        },
        "required": ["document_id"],
    }
    
    def __init__(self, document_manager: DocumentManager):
        self._doc_mgr = document_manager
    
    async def execute(
        self,
        document_id: str,
        focus: Optional[str] = None,
        detail_level: str = "summary",
        include_references: bool = True,
        **kwargs,
    ) -> ToolResult:
        """Get document context."""
        # Map detail level to budget
        budget_map = {
            "overview": "small",
            "summary": "medium", 
            "detailed": "large",
            "full": "full",
        }
        budget = budget_map.get(detail_level, "medium")
        
        # Find block ID if focus is a title
        focus_block_id = None
        if focus:
            # Try to find by title
            index = self._doc_mgr._documents.get(document_id)
            if index:
                for bid, block in index._blocks.items():
                    if focus.lower() in (block.title or "").lower():
                        focus_block_id = bid
                        break
        
        try:
            lens = await self._doc_mgr.get_context_lens(
                document_id=document_id,
                focus_block_id=focus_block_id,
                budget=budget,
                include_surroundings=True,
            )
            
            # Get reference material if requested
            references = []
            if include_references:
                # Check if it's a book project with references
                project = self._doc_mgr.get_book_project(document_id)
                if project:
                    # Search references for relevant content
                    # This is a simplified version
                    pass
            
            return ToolResult(
                success=True,
                output={
                    "document_id": document_id,
                    "context_text": lens.to_prompt_fragment(),
                    "tokens_used": lens.current_tokens,
                    "focus_block": focus_block_id,
                    "navigation_handles": lens.navigation_handles,
                    "instructions": (
                        "Use this context to maintain consistency with the document. "
                        "Navigation handles can be used to access other sections. "
                        "The STRUCTURE OVERVIEW shows global organization."
                    ),
                },
                error=None,
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Context retrieval failed: {str(e)}",
            )


class WriteSectionTool(Tool):
    """
    Tool for generating content for a specific section.
    
    Sets up proper context including surrounding material and references,
    then prepares execution context for the LLM.
    """
    
    name: str = "write_section"
    description: str = (
        "Generate content for a specific chapter or section. "
        "Automatically retrieves surrounding context and reference materials "
        "to ensure continuity. Creates a writing task that can be executed "
        "with full awareness of document structure."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "Book project ID",
            },
            "chapter_number": {
                "type": "integer",
                "description": "Which chapter to write",
                "minimum": 1,
            },
            "section_title": {
                "type": "string",
                "description": "Specific section to write (optional - writes whole chapter if not specified)",
            },
            "target_words": {
                "type": "integer",
                "description": "Target word count",
                "default": 2000,
            },
            "style_notes": {
                "type": "string",
                "description": "Specific style guidance for this section",
            },
        },
        "required": ["project_id", "chapter_number"],
    }
    
    def __init__(self, document_manager: DocumentManager):
        self._doc_mgr = document_manager
    
    async def execute(
        self,
        project_id: str,
        chapter_number: int,
        section_title: Optional[str] = None,
        target_words: int = 2000,
        style_notes: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Create writing task."""
        project = self._doc_mgr.get_book_project(project_id)
        
        if not project:
            return ToolResult(
                success=False,
                output=None,
                error=f"Project {project_id} not found",
            )
        
        try:
            # If section specified, find it
            target_id = None
            if section_title:
                # Find section in chapter
                chap_id = f"{project_id}_ch{chapter_number}"
                chapter = self._doc_mgr._documents[project_id].get_block(chap_id)
                if chapter:
                    for child_id in chapter.children_ids:
                        child = self._doc_mgr._documents[project_id].get_block(child_id)
                        if child and section_title.lower() in (child.title or "").lower():
                            target_id = child_id
                            break
            
            # Create task
            task_id = await project.write_chapter(
                chapter_number=chapter_number,
                target_length_words=target_words,
                style_guidance=style_notes,
            )
            
            # Execute to get context
            execution = await project.execute_task(task_id)
            
            return ToolResult(
                success=True,
                output={
                    "task_id": task_id,
                    "status": "context_ready",
                    "execution_context": execution.get("execution_prompt"),
                    "instruction": execution.get("instruction"),
                    "estimated_tokens": execution.get("context_tokens_estimate"),
                    "next_step": (
                        "Review the execution_context and write content. "
                        "Then use submit_written_content to save results."
                    ),
                },
                error=None,
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Writing task creation failed: {str(e)}",
            )


class SubmitWrittenContentTool(Tool):
    """Tool for submitting completed writing back to the project."""
    
    name: str = "submit_written_content"
    description: str = (
        "Submit completed writing for a section or chapter. "
        "Updates the document structure with new content and marks "
        "the writing task as complete. Content is parsed into paragraph "
        "blocks and integrated into the hierarchy."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "Book project ID",
            },
            "task_id": {
                "type": "string",
                "description": "Writing task ID from write_section",
            },
            "content": {
                "type": "string",
                "description": "The written content",
            },
        },
        "required": ["project_id", "task_id", "content"],
    }
    
    def __init__(self, document_manager: DocumentManager):
        self._doc_mgr = document_manager
    
    async def execute(
        self,
        project_id: str,
        task_id: str,
        content: str,
        **kwargs,
    ) -> ToolResult:
        """Submit content."""
        project = self._doc_mgr.get_book_project(project_id)
        
        if not project:
            return ToolResult(
                success=False,
                output=None,
                error=f"Project {project_id} not found",
            )
        
        word_count = len(content.split())
        
        try:
            block_id = project.submit_written_content(
                task_id=task_id,
                content=content,
                word_count=word_count,
            )
            
            # Update progress
            progress = project.get_progress()
            
            return ToolResult(
                success=True,
                output={
                    "task_id": task_id,
                    "block_id": block_id,
                    "word_count": word_count,
                    "status": "submitted",
                    "project_progress": progress,
                    "next_steps": [
                        "Continue to next section with write_section",
                        "Review with get_document_context",
                        "Revise if needed by submitting new content",
                    ],
                },
                error=None,
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Content submission failed: {str(e)}",
            )


class SearchReferencesTool(Tool):
    """Tool for searching across reference documents."""
    
    name: str = "search_references"
    description: str = (
        "Search for information across all reference documents in a project. "
        "Returns relevant passages with source attribution. "
        "Use this to find facts, quotes, or details from your research materials."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "Book project ID",
            },
            "query": {
                "type": "string",
                "description": "What to search for",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results per document",
                "default": 3,
            },
        },
        "required": ["project_id", "query"],
    }
    
    def __init__(self, document_manager: DocumentManager):
        self._doc_mgr = document_manager
    
    async def execute(
        self,
        project_id: str,
        query: str,
        limit: int = 3,
        **kwargs,
    ) -> ToolResult:
        """Search references."""
        project = self._doc_mgr.get_book_project(project_id)
        
        if not project:
            return ToolResult(
                success=False,
                output=None,
                error=f"Project {project_id} not found",
            )
        
        # Search across reference documents
        results = self._doc_mgr.search_across_documents(
            query=query,
            document_ids=project.reference_documents,
            limit_per_doc=limit,
        )
        
        # Format results
        formatted_results = []
        for doc_id, matches in results.items():
            doc_results = []
            for block, score in matches:
                doc_results.append({
                    "title": block.title or "Untitled",
                    "type": block.block_type.name,
                    "relevance_score": round(score, 2),
                    "summary": block.summary or block.content[:300],
                    "source_handle": f"[[MEMORY:{block.memory_chunk_id}]]" if block.memory_chunk_id else None,
                })
            
            formatted_results.append({
                "document_id": doc_id,
                "matches": doc_results,
            })
        
        # Also search in the book itself (already written content)
        book_results = self._doc_mgr._documents.get(project_id, DocumentIndex(project_id)).search(query, limit)
        
        return ToolResult(
            success=True,
            output={
                "query": query,
                "reference_results": formatted_results,
                "already_written_matches": [
                    {
                        "title": b.title,
                        "relevance": round(s, 2),
                    }
                    for b, s in book_results[:3]
                ],
                "instruction": (
                    "Use these results to inform your writing. "
                    "Source handles can be loaded for full context if needed."
                ),
            },
            error=None,
        )


class GetWritingProgressTool(Tool):
    """Tool for checking book writing progress."""
    
    name: str = "get_writing_progress"
    description: str = (
        "Get detailed progress report on a book project. "
        "Shows completed chapters, word counts, tasks remaining, "
        "and overall completion percentage."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "Book project ID",
            },
        },
        "required": ["project_id"],
    }
    
    def __init__(self, document_manager: DocumentManager):
        self._doc_mgr = document_manager
    
    async def execute(
        self,
        project_id: str,
        **kwargs,
    ) -> ToolResult:
        """Get progress."""
        project = self._doc_mgr.get_book_project(project_id)
        
        if not project:
            return ToolResult(
                success=False,
                output=None,
                error=f"Project {project_id} not found",
            )
        
        progress = project.get_progress()
        
        # Get detailed structure
        index = self._doc_mgr._documents.get(project_id)
        structure = []
        
        if index:
            for chap_id in index._sequence.get(BlockType.CHAPTER, []):
                chap = index.get_block(chap_id)
                if chap:
                    sections = []
                    for sect_id in chap.children_ids:
                        sect = index.get_block(sect_id)
                        if sect:
                            # Count paragraphs
                            para_count = len([
                                c for c in sect.children_ids 
                                if index.get_block(c) and index.get_block(c).block_type == BlockType.PARAGRAPH
                            ])
                            sections.append({
                                "title": sect.title,
                                "status": sect.status,
                                "paragraphs": para_count,
                            })
                    
                    structure.append({
                        "chapter": chap.title,
                        "status": chap.status,
                        "sections": sections,
                    })
        
        progress["structure"] = structure
        
        return ToolResult(
            success=True,
            output=progress,
            error=None,
        )


def get_writing_tools(document_manager: DocumentManager) -> List[Tool]:
    """Get all writing-related tools."""
    return [
        IngestDocumentTool(document_manager),
        CreateBookProjectTool(document_manager),
        CreateOutlineTool(document_manager),
        GetDocumentContextTool(document_manager),
        WriteSectionTool(document_manager),
        SubmitWrittenContentTool(document_manager),
        SearchReferencesTool(document_manager),
        GetWritingProgressTool(document_manager),
    ]
