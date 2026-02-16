"""
Knowledge Base Tools for SimplifiedAgant integration
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lollmsbot.agent import Tool, ToolResult


class KnowledgeQueryTool(Tool):
    """Query the knowledge base."""
    
    name: str = "knowledge_query"
    description: str = (
        "Search the knowledge base for information you've saved. "
        "Find articles, web pages, YouTube videos, and notes. "
        "Returns results with source attribution and relevance scores."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    
    risk_level: str = "low"
    
    def __init__(self, kb_manager: Optional[Any] = None):
        self._kb = kb_manager
    
    async def execute(self, query: str, limit: int = 5, **kwargs) -> ToolResult:
        if not self._kb:
            return ToolResult(
                success=False,
                output=None,
                error="Knowledge base not initialized",
            )
        
        try:
            results = await self._kb.query(query, limit)
            
            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "results": results,
                    "count": len(results),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Knowledge query failed: {str(e)}",
            )


class IngestContentTool(Tool):
    """Ingest content into knowledge base."""
    
    name: str = "ingest_content"
    description: str = (
        "Save web content, articles, or YouTube videos to your knowledge base. "
        "Content is automatically analyzed, tagged, and made searchable. "
        "Also extracts potential video ideas if the content is relevant."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to ingest",
            },
            "title": {
                "type": "string",
                "description": "Optional title override",
            },
            "content": {
                "type": "string",
                "description": "Optional content (if already fetched)",
            },
        },
        "required": ["url"],
    }
    
    risk_level: str = "low"
    
    def __init__(self, kb_manager: Optional[Any] = None):
        self._kb = kb_manager
    
    async def execute(self, url: str, title: Optional[str] = None, 
                     content: Optional[str] = None, **kwargs) -> ToolResult:
        if not self._kb:
            return ToolResult(
                success=False,
                output=None,
                error="Knowledge base not initialized",
            )
        
        try:
            # Check if YouTube
            if "youtube.com" in url or "youtu.be" in url:
                entry = await self._kb.ingest_youtube(url, transcript=content, title=title)
            else:
                entry = await self._kb.ingest_url(url, title, content)
            
            if not entry:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Failed to ingest content",
                )
            
            # Get video ideas if relevant
            ideas = []
            if "video" in entry.title.lower() or "youtube" in entry.source_type:
                ideas = await self._kb.get_video_ideas(entry.title)
            
            return ToolResult(
                success=True,
                output={
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "source_type": entry.source_type,
                    "topics": entry.topics,
                    "video_ideas_found": len(ideas),
                    "ideas": ideas[:3],
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Content ingestion failed: {str(e)}",
            )


__all__ = ["KnowledgeQueryTool", "IngestContentTool"]
