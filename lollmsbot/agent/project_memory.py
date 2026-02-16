"""
Project-Based Memory System for LollmsBot

Implements hierarchical memory architecture:
- Projects: Encapsulated memory containers for specific work streams
- Global Memory: Cross-project knowledge and project registry
- Working Memory (RCB): Dynamically loaded active context

The LLM can explicitly control memory through project operations:
- Create/load/unload projects
- Push/pop memory segments
- Query project metadata
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable

from lollmsbot.agent.rlm import RLMMemoryManager, MemoryChunk, MemoryChunkType, RCBEntry

import logging

logger = logging.getLogger(__name__)


class ProjectStatus(Enum):
    """Status of a project in memory."""
    INACTIVE = auto()      # Stored in EMS only
    LOADED = auto()        # Basic metadata in RCB
    ACTIVE = auto()        # Full working context loaded
    ARCHIVED = auto()      # Compressed, rarely accessed


@dataclass
class MemorySegment:
    """
    A named chunk of memory within a project.
    Can be independently loaded/unloaded from RCB.
    """
    segment_id: str
    segment_type: str  # 'context', 'facts', 'conversations', 'files', 'notes'
    chunk_ids: List[str] = field(default_factory=list)
    priority: float = 1.0  # 1.0 = normal, higher = keep in RCB longer
    auto_load: bool = True  # Load into RCB when project activated
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "segment_type": self.segment_type,
            "chunk_count": len(self.chunk_ids),
            "priority": self.priority,
            "auto_load": self.auto_load,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "description": self.description,
        }


@dataclass
class Project:
    """
    A project encapsulates related memory for a specific work stream.
    Projects have segments that can be loaded/unloaded independently.
    """
    project_id: str
    name: str
    description: str = ""
    status: ProjectStatus = ProjectStatus.INACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    # Memory organization
    segments: Dict[str, MemorySegment] = field(default_factory=dict)
    active_segment_ids: Set[str] = field(default_factory=set)
    
    # Project metadata for quick loading
    summary: str = ""  # AI-generated project summary
    key_facts: List[str] = field(default_factory=list)  # Important facts
    tags: List[str] = field(default_factory=list)
    
    # Cross-project references
    related_projects: List[str] = field(default_factory=list)
    parent_project: Optional[str] = None
    
    # Statistics
    total_chunks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "segments": {k: v.to_dict() for k, v in self.segments.items()},
            "active_segments": list(self.active_segment_ids),
            "summary": self.summary,
            "key_facts": self.key_facts,
            "tags": self.tags,
            "related_projects": self.related_projects,
            "parent_project": self.parent_project,
            "total_chunks": self.total_chunks,
        }
    
    def get_active_segments(self) -> List[MemorySegment]:
        """Get currently active segments."""
        return [self.segments[sid] for sid in self.active_segment_ids if sid in self.segments]


class ProjectMemoryManager:
    """
    Manages project-based memory with dynamic loading/unloading.
    
    Architecture:
    - Projects are stored in EMS as compressed chunks
    - Project metadata is indexed for fast discovery
    - Active project segments are loaded into RCB as needed
    - Global memory persists across project switches
    """
    
    def __init__(
        self,
        memory_manager: RLMMemoryManager,
        max_active_projects: int = 3,
        max_segments_per_project: int = 5,
        global_memory_capacity: int = 20,
    ) -> None:
        self._memory = memory_manager
        
        # Project storage
        self._projects: Dict[str, Project] = {}
        self._project_chunk_map: Dict[str, str] = {}  # project_id -> EMS chunk_id
        
        # Active state
        self._active_project_id: Optional[str] = None
        self._max_active_projects = max_active_projects
        self._max_segments_per_project = max_segments_per_project
        
        # Global memory (always loaded)
        self._global_memory_tags: Set[str] = {"global", "cross_project", "project_registry"}
        self._global_memory_capacity = global_memory_capacity
        
        # Segment cache in RCB
        self._loaded_segments: Dict[str, datetime] = {}  # segment_id -> loaded_at
        self._segment_lock: asyncio.Lock = asyncio.Lock()
        
        # Event callbacks
        self._project_callbacks: List[Callable[[str, str, Dict], Awaitable[None]]] = []
        self._segment_callbacks: List[Callable[[str, str, Dict], Awaitable[None]]] = []
        
        # Initialization
        self._initialized = False
        
        # Memory structure cache
        self._structure_cache: Optional[Dict[str, Any]] = None
        self._structure_timestamp: Optional[datetime] = None
    
    async def initialize(self) -> None:
        """Initialize project manager and load project registry."""
        if self._initialized:
            return
        
        logger.info("Initializing Project Memory Manager...")
        
        # Load existing projects from EMS
        await self._load_project_registry()
        
        self._initialized = True
        active_count = len([p for p in self._projects.values() if p.status != ProjectStatus.INACTIVE])
        logger.info(f"Project Memory ready: {len(self._projects)} projects, {active_count} active")
    
    def on_project_event(self, callback: Callable[[str, str, Dict], Awaitable[None]]) -> None:
        """Register callback for project events: (event_type, project_id, data)."""
        self._project_callbacks.append(callback)
    
    def on_segment_event(self, callback: Callable[[str, str, Dict], Awaitable[None]]) -> None:
        """Register callback for segment events: (event_type, segment_id, data)."""
        self._segment_callbacks.append(callback)
    
    async def _notify_project(self, event_type: str, project_id: str, data: Dict) -> None:
        for cb in self._project_callbacks:
            try:
                await cb(event_type, project_id, data)
            except Exception as e:
                logger.warning(f"Project callback failed: {e}")
    
    async def _notify_segment(self, event_type: str, segment_id: str, data: Dict) -> None:
        for cb in self._segment_callbacks:
            try:
                await cb(event_type, segment_id, data)
            except Exception as e:
                logger.warning(f"Segment callback failed: {e}")
    
    # ========== PROJECT LIFECYCLE ==========
    
    async def create_project(
        self,
        name: str,
        description: str = "",
        initial_context: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Project:
        """
        Create a new project with optional initial context.
        
        Returns the created Project object.
        """
        project_id = f"proj_{hashlib.sha256(name.encode()).hexdigest()[:16]}"
        
        # Check if exists
        if project_id in self._projects:
            raise ValueError(f"Project '{name}' already exists (ID: {project_id})")
        
        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            tags=tags or [],
            status=ProjectStatus.INACTIVE,
        )
        
        # Create initial segments
        if initial_context:
            context_segment = MemorySegment(
                segment_id=f"{project_id}_context",
                segment_type="context",
                description="Initial project context and goals",
                priority=10.0,  # High priority
            )
            project.segments[context_segment.segment_id] = context_segment
            project.active_segment_ids.add(context_segment.segment_id)
            
            # Store initial context in EMS
            chunk_id = await self._memory.store_in_ems(
                content=initial_context,
                chunk_type=MemoryChunkType.WORKING_MEMORY,
                importance=8.0,
                tags=["project", "context", project_id],
                summary=f"Project '{name}' initial context",
                load_hints=[name, project_id],
                source=f"project:{project_id}:context",
            )
            context_segment.chunk_ids.append(chunk_id)
        
        # Always create a facts segment
        facts_segment = MemorySegment(
            segment_id=f"{project_id}_facts",
            segment_type="facts",
            description="Key facts and knowledge about this project",
            priority=9.0,
        )
        project.segments[facts_segment.segment_id] = facts_segment
        
        # Store project metadata in EMS
        await self._save_project_to_ems(project)
        
        # Add to registry
        self._projects[project_id] = project
        await self._update_project_registry()
        
        await self._notify_project("created", project_id, project.to_dict())
        logger.info(f"Created project '{name}' ({project_id})")
        
        return project
    
    async def load_project(self, project_id: str, load_segments: bool = True) -> Project:
        """
        Load a project into working memory.
        
        If another project is active, it may be deactivated based on max_active_projects.
        """
        if project_id not in self._projects:
            # Try to load from EMS
            project = await self._load_project_from_ems(project_id)
            if project:
                self._projects[project_id] = project
            else:
                raise ValueError(f"Project '{project_id}' not found")
        
        project = self._projects[project_id]
        
        # Check if we need to unload other projects
        active_projects = [
            pid for pid, p in self._projects.items()
            if p.status in (ProjectStatus.LOADED, ProjectStatus.ACTIVE)
        ]
        
        if len(active_projects) >= self._max_active_projects and project_id not in active_projects:
            # Unload oldest active project
            oldest = min(
                active_projects,
                key=lambda pid: self._projects[pid].modified_at
            )
            await self.unload_project(oldest)
        
        # Set as active
        self._active_project_id = project_id
        project.status = ProjectStatus.ACTIVE
        project.modified_at = datetime.now()
        
        # Load segments into RCB if requested
        if load_segments:
            for segment in project.segments.values():
                if segment.auto_load or segment.segment_id in project.active_segment_ids:
                    await self.load_segment(project_id, segment.segment_id)
        
        await self._notify_project("activated", project_id, project.to_dict())
        logger.info(f"Activated project '{project.name}' ({project_id})")
        
        return project
    
    async def unload_project(self, project_id: str, save_state: bool = True) -> None:
        """
        Unload a project from working memory.
        Saves current state to EMS.
        """
        if project_id not in self._projects:
            return
        
        project = self._projects[project_id]
        
        # Unload all segments
        for segment_id in list(project.active_segment_ids):
            await self.unload_segment(project_id, segment_id)
        
        if save_state:
            await self._save_project_to_ems(project)
        
        project.status = ProjectStatus.INACTIVE
        project.active_segment_ids.clear()
        
        if self._active_project_id == project_id:
            self._active_project_id = None
        
        await self._notify_project("deactivated", project_id, project.to_dict())
        logger.info(f"Deactivated project '{project.name}' ({project_id})")
    
    async def delete_project(self, project_id: str) -> bool:
        """
        Permanently delete a project and all its memory.
        """
        if project_id not in self._projects:
            return False
        
        # Unload first
        await self.unload_project(project_id, save_state=False)
        
        project = self._projects[project_id]
        
        # Delete all segment chunks from EMS
        for segment in project.segments.values():
            for chunk_id in segment.chunk_ids:
                await self._memory._db.delete_chunk(chunk_id)
        
        # Delete project metadata
        if project_id in self._project_chunk_map:
            await self._memory._db.delete_chunk(self._project_chunk_map[project_id])
            del self._project_chunk_map[project_id]
        
        del self._projects[project_id]
        await self._update_project_registry()
        
        await self._notify_project("deleted", project_id, {"name": project.name})
        logger.info(f"Deleted project '{project.name}' ({project_id})")
        
        return True
    
    # ========== SEGMENT MANAGEMENT ==========
    
    async def create_segment(
        self,
        project_id: str,
        segment_name: str,
        segment_type: str = "notes",
        initial_content: Optional[str] = None,
    ) -> MemorySegment:
        """Create a new memory segment in a project."""
        if project_id not in self._projects:
            raise ValueError(f"Project '{project_id}' not found")
        
        project = self._projects[project_id]
        
        # Generate segment ID
        segment_id = f"{project_id}_{segment_type}_{len(project.segments)}"
        
        segment = MemorySegment(
            segment_id=segment_id,
            segment_type=segment_type,
            description=segment_name,
        )
        
        if initial_content:
            chunk_id = await self._memory.store_in_ems(
                content=initial_content,
                chunk_type=MemoryChunkType.WORKING_MEMORY,
                importance=5.0,
                tags=["project", "segment", project_id, segment_type],
                summary=f"Segment '{segment_name}' in project '{project.name}'",
                source=f"project:{project_id}:segment:{segment_id}",
            )
            segment.chunk_ids.append(chunk_id)
        
        project.segments[segment_id] = segment
        project.total_chunks += len(segment.chunk_ids)
        
        await self._save_project_to_ems(project)
        
        await self._notify_segment("created", segment_id, segment.to_dict())
        logger.info(f"Created segment '{segment_name}' in project '{project.name}'")
        
        return segment
    
    async def load_segment(self, project_id: str, segment_id: str) -> bool:
        """
        Load a segment's contents into RCB (working memory).
        """
        async with self._segment_lock:
            if project_id not in self._projects:
                return False
            
            project = self._projects[project_id]
            
            if segment_id not in project.segments:
                return False
            
            segment = project.segments[segment_id]
            
            # Already loaded?
            if segment_id in self._loaded_segments:
                segment.last_accessed = datetime.now()
                return True
            
            # Check capacity
            if len(self._loaded_segments) >= self._max_segments_per_project * self._max_active_projects:
                # Unload lowest priority segment
                await self._unload_lowest_priority_segment()
            
            # Load all chunk contents into RCB
            for chunk_id in segment.chunk_ids:
                await self._memory.load_from_ems(chunk_id, add_to_rcb=True)
            
            segment.last_accessed = datetime.now()
            self._loaded_segments[segment_id] = datetime.now()
            project.active_segment_ids.add(segment_id)
            
            await self._notify_segment("loaded", segment_id, {
                "project_id": project_id,
                "chunk_count": len(segment.chunk_ids),
            })
            
            logger.info(f"Loaded segment '{segment_id}' ({len(segment.chunk_ids)} chunks)")
            return True
    
    async def unload_segment(self, project_id: str, segment_id: str) -> bool:
        """Unload a segment from RCB."""
        async with self._segment_lock:
            if project_id not in self._projects:
                return False
            
            project = self._projects[project_id]
            
            # Remove from loaded tracking
            if segment_id in self._loaded_segments:
                del self._loaded_segments[segment_id]
            
            project.active_segment_ids.discard(segment_id)
            
            # Note: We don't actually remove from RCB here - that's managed by RLM
            # We just stop tracking it as "loaded by us"
            
            await self._notify_segment("unloaded", segment_id, {"project_id": project_id})
            logger.info(f"Unloaded segment '{segment_id}'")
            return True
    
    async def add_to_segment(
        self,
        project_id: str,
        segment_id: str,
        content: str,
        importance: float = 5.0,
    ) -> str:
        """
        Add content to a segment. Creates new chunk in EMS.
        Returns the new chunk_id.
        """
        if project_id not in self._projects:
            raise ValueError(f"Project '{project_id}' not found")
        
        project = self._projects[project_id]
        
        if segment_id not in project.segments:
            raise ValueError(f"Segment '{segment_id}' not found in project '{project_id}'")
        
        segment = project.segments[segment_id]
        
        # Store in EMS
        chunk_id = await self._memory.store_in_ems(
            content=content,
            chunk_type=MemoryChunkType.WORKING_MEMORY,
            importance=importance,
            tags=["project", segment.segment_type, project_id],
            summary=content[:200],
            source=f"project:{project_id}:segment:{segment_id}",
        )
        
        segment.chunk_ids.append(chunk_id)
        project.total_chunks += 1
        
        # If segment is active, load the new chunk
        if segment_id in project.active_segment_ids:
            await self._memory.load_from_ems(chunk_id, add_to_rcb=True)
        
        return chunk_id
    
    # ========== MEMORY OPERATIONS ==========
    
    async def promote_to_global(self, project_id: str, content: str, category: str = "cross_project") -> str:
        """
        Promote project-specific information to global memory.
        This information persists across project switches.
        """
        chunk_id = await self._memory.store_in_ems(
            content=content,
            chunk_type=MemoryChunkType.FACT,
            importance=7.0,  # Higher importance for global knowledge
            tags=["global", category, "cross_project"],
            summary=f"Global: {content[:100]}",
            load_hints=[category],
            source="global:promoted",
        )
        
        # Ensure it's loaded into RCB as global memory
        await self._memory.load_from_ems(chunk_id, add_to_rcb=True)
        
        logger.info(f"Promoted content to global memory ({category})")
        return chunk_id
    
    async def query_project_memory(
        self,
        project_id: str,
        query: str,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search memory within a specific project.
        """
        if project_id not in self._projects:
            return []
        
        project = self._projects[project_id]
        
        # Collect all chunk IDs from project segments
        all_chunk_ids = []
        for segment in project.segments.values():
            if include_inactive or segment.segment_id in project.active_segment_ids:
                all_chunk_ids.extend(segment.chunk_ids)
        
        # Search in EMS with project filter
        results = await self._memory.search_ems(
            query=query,
            limit=20,
        )
        
        # Filter to project chunks
        project_results = [
            {
                "chunk_id": chunk.chunk_id,
                "summary": chunk.summary,
                "relevance": score,
                "type": chunk.chunk_type.name,
            }
            for chunk, score in results
            if chunk.chunk_id in all_chunk_ids
        ]
        
        return project_results
    
    async def get_active_memory_context(self) -> Dict[str, Any]:
        """
        Get complete memory context for LLM.
        Includes: active project + loaded segments + global memory
        """
        context = {
            "active_project": None,
            "loaded_segments": [],
            "global_memory_summary": "",
            "available_projects": [],
        }
        
        # Active project info
        if self._active_project_id and self._active_project_id in self._projects:
            project = self._projects[self._active_project_id]
            context["active_project"] = {
                "id": project.project_id,
                "name": project.name,
                "description": project.description,
                "status": project.status.name,
                "summary": project.summary,
                "key_facts": project.key_facts[:5],
                "active_segments": [
                    {
                        "id": seg.segment_id,
                        "type": seg.segment_type,
                        "description": seg.description,
                        "chunk_count": len(seg.chunk_ids),
                    }
                    for seg in project.get_active_segments()
                ],
            }
        
        # Global memory hints
        context["global_memory_summary"] = await self._get_global_memory_summary()
        
        # Available projects (quick list)
        context["available_projects"] = [
            {
                "id": p.project_id,
                "name": p.name,
                "status": p.status.name,
                "description": p.description[:100],
            }
            for p in sorted(self._projects.values(), key=lambda x: x.modified_at, reverse=True)[:10]
        ]
        
        return context
    
    async def _get_global_memory_summary(self) -> str:
        """Get summary of global memory for context."""
        # Get high-importance global chunks
        results = await self._memory.search_ems(
            query="global cross_project",
            limit=5,
        )
        
        summaries = []
        for chunk, score in results:
            if any(tag in (chunk.tags or []) for tag in self._global_memory_tags):
                summaries.append(chunk.summary or "Unknown")
        
        return "; ".join(summaries) if summaries else "No global memory loaded"
    
    # ========== PROJECT REGISTRY ==========
    
    async def _save_project_to_ems(self, project: Project) -> str:
        """Serialize project metadata to EMS."""
        project_data = project.to_dict()
        
        # Store as JSON
        chunk_id = await self._memory.store_in_ems(
            content=json.dumps(project_data, indent=2),
            chunk_type=MemoryChunkType.SELF_KNOWLEDGE,
            importance=6.0,
            tags=["project_metadata", "project_registry", project.project_id],
            summary=f"Project '{project.name}' metadata",
            load_hints=[project.name, project.project_id],
            source=f"project:metadata:{project.project_id}",
        )
        
        self._project_chunk_map[project.project_id] = chunk_id
        return chunk_id
    
    async def _load_project_from_ems(self, project_id: str) -> Optional[Project]:
        """Deserialize project from EMS."""
        if project_id not in self._project_chunk_map:
            return None
        
        chunk_id = self._project_chunk_map[project_id]
        row = await self._memory._db.get_chunk(chunk_id)
        
        if not row:
            return None
        
        try:
            content = row.get("content_compressed", b"")
            import zlib
            project_data = json.loads(zlib.decompress(content).decode('utf-8'))
            
            # Reconstruct project
            project = Project(
                project_id=project_data["project_id"],
                name=project_data["name"],
                description=project_data.get("description", ""),
                status=ProjectStatus[project_data.get("status", "INACTIVE")],
                created_at=datetime.fromisoformat(project_data["created_at"]),
                modified_at=datetime.fromisoformat(project_data["modified_at"]),
                summary=project_data.get("summary", ""),
                key_facts=project_data.get("key_facts", []),
                tags=project_data.get("tags", []),
                related_projects=project_data.get("related_projects", []),
                parent_project=project_data.get("parent_project"),
                total_chunks=project_data.get("total_chunks", 0),
            )
            
            # Reconstruct segments
            for seg_id, seg_data in project_data.get("segments", {}).items():
                segment = MemorySegment(
                    segment_id=seg_data["segment_id"],
                    segment_type=seg_data["segment_type"],
                    priority=seg_data.get("priority", 1.0),
                    auto_load=seg_data.get("auto_load", True),
                    created_at=datetime.fromisoformat(seg_data["created_at"]),
                    last_accessed=datetime.fromisoformat(seg_data["last_accessed"]),
                    description=seg_data.get("description", ""),
                )
                project.segments[seg_id] = segment
            
            return project
            
        except Exception as e:
            logger.error(f"Failed to deserialize project {project_id}: {e}")
            return None
    
    async def _load_project_registry(self) -> None:
        """Load list of all projects from EMS."""
        # Search for project metadata chunks
        results = await self._memory.search_ems(
            query="project_metadata project_registry",
            limit=100,
        )
        
        for chunk, _ in results:
            if "project_metadata" in (chunk.tags or []):
                # Extract project ID from source
                source = chunk.source or ""
                if source.startswith("project:metadata:"):
                    project_id = source.split(":")[-1]
                    self._project_chunk_map[project_id] = chunk.chunk_id
                    
                    # Load basic info without full segments
                    try:
                        content = chunk.decompress_content()
                        data = json.loads(content)
                        self._projects[project_id] = Project(
                            project_id=data["project_id"],
                            name=data["name"],
                            description=data.get("description", ""),
                            status=ProjectStatus.INACTIVE,  # Start inactive
                            created_at=datetime.fromisoformat(data["created_at"]),
                            modified_at=datetime.fromisoformat(data["modified_at"]),
                            summary=data.get("summary", ""),
                            tags=data.get("tags", []),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to parse project {project_id}: {e}")
    
    async def _update_project_registry(self) -> None:
        """Update global registry of all projects."""
        registry = {
            "projects": [
                {
                    "id": p.project_id,
                    "name": p.name,
                    "description": p.description,
                    "status": p.status.name,
                    "tags": p.tags,
                    "modified_at": p.modified_at.isoformat(),
                }
                for p in self._projects.values()
            ],
            "active_project": self._active_project_id,
            "updated_at": datetime.now().isoformat(),
        }
        
        await self._memory.store_in_ems(
            content=json.dumps(registry),
            chunk_type=MemoryChunkType.SELF_KNOWLEDGE,
            importance=8.0,
            tags=["global", "project_registry", "metadata"],
            summary=f"Project registry: {len(self._projects)} projects",
            source="global:project_registry",
        )
    
    async def _unload_lowest_priority_segment(self) -> None:
        """Unload the least recently used, lowest priority segment."""
        if not self._loaded_segments:
            return
        
        # Find project with this segment
        candidates = []
        for seg_id, loaded_at in self._loaded_segments.items():
            # Find project and segment
            for project in self._projects.values():
                if seg_id in project.segments:
                    segment = project.segments[seg_id]
                    candidates.append((seg_id, project.project_id, segment.priority, loaded_at))
                    break
        
        if not candidates:
            return
        
        # Sort by (priority desc, loaded_at asc)
        candidates.sort(key=lambda x: (-x[2], x[3]))
        
        # Unload lowest priority/oldest
        seg_id, proj_id, _, _ = candidates[-1]
        await self.unload_segment(proj_id, seg_id)
    
    # ========== QUERY INTERFACE ==========
    
    async def search_all_projects(self, query: str) -> List[Dict[str, Any]]:
        """Search across all projects for relevant information."""
        results = []
        
        for project in self._projects.values():
            project_results = await self.query_project_memory(project.project_id, query, include_inactive=True)
            if project_results:
                results.append({
                    "project_id": project.project_id,
                    "project_name": project.name,
                    "matches": project_results[:3],  # Top 3 per project
                })
        
        return results
    
    async def get_project_list(self) -> List[Dict[str, Any]]:
        """Get list of all projects with basic info."""
        return [
            {
                "id": p.project_id,
                "name": p.name,
                "description": p.description,
                "status": p.status.name,
                "segment_count": len(p.segments),
                "total_chunks": p.total_chunks,
                "modified_at": p.modified_at.isoformat(),
                "is_active": p.project_id == self._active_project_id,
            }
            for p in sorted(self._projects.values(), key=lambda x: x.modified_at, reverse=True)
        ]
    
    async def get_project_details(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get full details of a specific project."""
        if project_id not in self._projects:
            return None
        
        project = self._projects[project_id]
        return project.to_dict()
    
    # ========== FORMATTING FOR LLM ==========
    
    def format_memory_structure_for_prompt(self, max_length: int = 4000) -> str:
        """Format current memory structure as REPL-style prompt."""
        lines = [
            "",
            "=" * 60,
            "🧠 PROJECT MEMORY ARCHITECTURE",
            "=" * 60,
            "",
            "Your memory is organized into PROJECTS with DYNAMIC SEGMENTS.",
            "You control memory loading through explicit operations.",
            "",
            "-" * 60,
            "ACTIVE PROJECT",
            "-" * 60,
        ]
        
        if self._active_project_id and self._active_project_id in self._projects:
            project = self._projects[self._active_project_id]
            lines.extend([
                f"",
                f"Project: {project.name} (ID: {project.project_id})",
                f"Status: {project.status.name}",
                f"Description: {project.description[:100] if project.description else 'None'}",
                f"",
                "ACTIVE SEGMENTS (loaded in RCB):",
            ])
            
            for seg_id in project.active_segment_ids:
                if seg_id in project.segments:
                    seg = project.segments[seg_id]
                    lines.append(f"  • [{seg.segment_type}] {seg.description}")
                    lines.append(f"    Chunks: {len(seg.chunk_ids)} | Priority: {seg.priority}")
            
            lines.append("")
            lines.append("INACTIVE SEGMENTS (in EMS, load on demand):")
            for seg_id, seg in project.segments.items():
                if seg_id not in project.active_segment_ids:
                    lines.append(f"  • [{seg.segment_type}] {seg.description} ({len(seg.chunk_ids)} chunks)")
        else:
            lines.append("No active project. Use project operations to activate one.")
        
        lines.extend([
            "",
            "-" * 60,
            "OTHER PROJECTS",
            "-" * 60,
        ])
        
        other_projects = [
            p for p in self._projects.values()
            if p.project_id != self._active_project_id
        ]
        
        for project in other_projects[:5]:
            status_icon = "●" if project.status != ProjectStatus.INACTIVE else "○"
            lines.append(f"  {status_icon} {project.name} ({project.status.name.lower()})")
        
        lines.extend([
            "",
            "-" * 60,
            "MEMORY OPERATIONS",
            "-" * 60,
            "",
            "You can control memory using these patterns:",
            "",
            "CREATE PROJECT:",
            "  <tool>project_memory</tool>",
            "  <operation>create_project</operation>",
            "  <name>My Project Name</name>",
            "  <description>What this project is about</description>",
            "",
            "LOAD PROJECT:",
            "  <tool>project_memory</tool>",
            "  <operation>load_project</operation>",
            "  <project_id>proj_...</project_id>",
            "",
            "CREATE SEGMENT:",
            "  <tool>project_memory</tool>",
            "  <operation>create_segment</operation>",
            "  <project_id>current</project_id>",
            "  <segment_name>Research Notes</segment_name>",
            "  <segment_type>notes</segment_type>",
            "",
            "LOAD SEGMENT:",
            "  <tool>project_memory</tool>",
            "  <operation>load_segment</operation>",
            "  <project_id>current</project_id>",
            "  <segment_id>proj_..._notes_0</segment_id>",
            "",
            "PROMOTE TO GLOBAL:",
            "  <tool>project_memory</tool>",
            "  <operation>promote_to_global</operation>",
            "  <project_id>current</project_id>",
            "  <content>This fact should persist across projects</content>",
            "",
            "QUERY PROJECT MEMORY:",
            "  <tool>project_memory</tool>",
            "  <operation>query_memory</operation>",
            "  <project_id>current</project_id>",
            "  <query>what do I know about X</query>",
            "",
            "=" * 60,
        ])
        
        result = "\n".join(lines)
        if len(result) > max_length:
            result = result[:max_length - 100] + "\n\n... [Memory structure truncated]"
        
        return result


# Global singleton
_project_memory_instance: Optional[ProjectMemoryManager] = None

def get_project_memory_manager(memory_manager: RLMMemoryManager) -> ProjectMemoryManager:
    """Get or create the singleton ProjectMemoryManager."""
    global _project_memory_instance
    if _project_memory_instance is None:
        _project_memory_instance = ProjectMemoryManager(memory_manager)
    return _project_memory_instance
