"""
Integrated Document Agent - Agent subclass with built-in document management.

This extends the base Agent with document-aware capabilities for
large-scale writing projects. The agent can:
- Ingest and structure reference materials
- Navigate hierarchical documents via REPL-style commands
- Write with global awareness of document structure
- Maintain consistency across long-form content
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Set

from lollmsbot.agent import Agent, PermissionLevel
from lollmsbot.agent.rlm import RLMMemoryManager, MemoryChunkType
from lollmsbot.document_manager import (
    DocumentManager,
    BookProject,
    ContextLens,
    create_document_manager,
)
from lollmsbot.writing_tools import get_writing_tools
from lollmsbot.config import BotConfig


class IntegratedDocumentAgent(Agent):
    """
    Agent with integrated document management for long-form writing.
    
    Extends base Agent with:
    - DocumentManager for hierarchical content
    - Writing-specific tools for book creation
    - Automatic context lens selection for document queries
    """
    
    def __init__(
        self,
        config: Optional[BotConfig] = None,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        default_permissions: PermissionLevel = PermissionLevel.BASIC,
        enable_guardian: bool = True,
        verbose_logging: bool = True,
        memory_db_path: Optional[Any] = None,
    ) -> None:
        super().__init__(
            config=config,
            agent_id=agent_id,
            name=name or "DocumentAgent",
            default_permissions=default_permissions,
            enable_guardian=enable_guardian,
            verbose_logging=verbose_logging,
            memory_db_path=memory_db_path,
        )
        
        self._doc_manager: Optional[DocumentManager] = None
        self._active_projects: Dict[str, BookProject] = {}
        self._current_focus: Optional[str] = None  # document_id:block_id
    
    async def initialize(
        self,
        gateway_mode: str = "unknown",
        host_bindings: Optional[List[str]] = None,
    ) -> None:
        """Initialize with document management."""
        # Initialize base agent (creates memory manager)
        await super().initialize(gateway_mode, host_bindings)
        
        # Initialize document manager with our memory
        if self._memory:
            self._doc_manager = await create_document_manager(
                memory_manager=self._memory,
            )
            
            # Register writing tools
            writing_tools = get_writing_tools(self._doc_manager)
            for tool in writing_tools:
                try:
                    await self.register_tool(tool)
                except ValueError:
                    # Tool might already be registered
                    pass
            
            self._logger.log(
                f"📚 Document management initialized with {len(writing_tools)} writing tools",
                "cyan",
                "📝"
            )
    
    async def chat(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced chat with document-aware context injection.
        
        If message mentions documents/projects, automatically injects
        relevant context lens.
        """
        # Check for document context requests
        enriched_context = dict(context) if context else {}
        
        # Detect document/project references
        doc_refs = self._extract_document_references(message)
        
        if doc_refs and self._doc_manager:
            # Inject document context
            doc_contexts = []
            for doc_id, focus, detail in doc_refs:
                try:
                    lens = await self._doc_manager.get_context_lens(
                        document_id=doc_id,
                        focus_block_id=focus,
                        budget=detail or "medium",
                    )
                    doc_contexts.append(lens.to_prompt_fragment())
                except Exception as e:
                    self._logger.log(f"Failed to get lens for {doc_id}: {e}", "yellow")
            
            if doc_contexts:
                enriched_context["document_contexts"] = doc_contexts
                self._logger.log(f"📖 Injected {len(doc_contexts)} document contexts", "cyan")
        
        # Call base chat with enriched context
        result = await super().chat(user_id, message, enriched_context)
        
        return result
    
    def _extract_document_references(
        self,
        message: str,
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """
        Extract document references from message.
        
        Patterns:
        - "document:doc_id" or "book:project_id"
        - "chapter X" (implies current project)
        - "section Y of document Z"
        
        Returns list of (doc_id, focus_block_id, detail_level)
        """
        refs = []
        
        # Pattern: document:xxx or book:xxx
        for match in re.finditer(r'(?:document|book|project):(\w+)', message, re.I):
            doc_id = match.group(1)
            refs.append((doc_id, None, None))
        
        # Pattern: chapter \d+
        chap_match = re.search(r'chapter\s+(\d+)', message, re.I)
        if chap_match and self._current_focus:
            chap_num = int(chap_match.group(1))
            doc_id, _ = self._current_focus.split(':', 1)
            focus_id = f"{doc_id}_ch{chap_num}"
            refs.append((doc_id, focus_id, "detailed"))
        
        return refs
    
    async def quick_ingest(
        self,
        content: str,
        title: str = "Quick Reference",
    ) -> str:
        """Quickly ingest text for immediate use."""
        if not self._doc_manager:
            raise RuntimeError("Document manager not initialized")
        
        doc_id = await self._doc_manager.ingest_text(
            content=content,
            title=title,
        )
        
        return doc_id
    
    async def create_book(
        self,
        title: str,
        author: Optional[str] = None,
    ) -> BookProject:
        """Create a new book project."""
        if not self._doc_manager:
            raise RuntimeError("Document manager not initialized")
        
        project_id = await self._doc_manager.create_book_project(
            title=title,
            author=author or "TBD",
        )
        
        project = self._doc_manager.get_book_project(project_id)
        self._active_projects[project_id] = project
        
        return project
    
    @property
    def document_manager(self) -> Optional[DocumentManager]:
        return self._doc_manager


# Re-export for convenience
__all__ = ["IntegratedDocumentAgent"]
