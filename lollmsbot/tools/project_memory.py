"""
Project Memory Tool - LLM-controllable project management.

Allows the LLM to:
- Create and manage projects
- Load/unload memory segments
- Query project-specific memory
- Promote information to global memory
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lollmsbot.agent import Tool, ToolResult


class ProjectMemoryTool(Tool):
    """
    Tool for managing project-based memory.
    
    This gives the LLM explicit control over memory organization.
    """
    
    name: str = "project_memory"
    description: str = (
        "Manage project-based memory. Projects encapsulate related work with "
        "segments that can be loaded/unloaded dynamically. Use this to organize "
        "long-running work, switch contexts, or query project-specific information."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "create_project",
                    "load_project",
                    "unload_project",
                    "delete_project",
                    "list_projects",
                    "get_project_details",
                    "create_segment",
                    "load_segment",
                    "unload_segment",
                    "add_to_segment",
                    "promote_to_global",
                    "query_memory",
                    "search_all_projects",
                    "get_active_context",
                    "find_or_open_project",
                ],
                "description": "The operation to perform. Use 'find_or_open_project' when user says 'open [project name]' - it will find existing or create new.",
            },
            "project_id": {
                "type": "string",
                "description": "Project ID (use 'current' for active project)",
            },
            "name": {
                "type": "string",
                "description": "Name for new project or project to open (use this or project_name)",
            },
            "project_name": {
                "type": "string",
                "description": "Alternative name parameter for find_or_open_project operation",
            },
            "description": {
                "type": "string",
                "description": "Description of project or content",
            },
            "segment_name": {
                "type": "string",
                "description": "Name for new segment",
            },
            "segment_type": {
                "type": "string",
                "enum": ["context", "facts", "conversations", "files", "notes"],
                "description": "Type of segment",
            },
            "segment_id": {
                "type": "string",
                "description": "Segment ID to load/unload",
            },
            "content": {
                "type": "string",
                "description": "Content to add",
            },
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "importance": {
                "type": "number",
                "description": "Importance 1-10",
                "default": 5.0,
            },
        },
        "required": ["operation"],
    }
    
    risk_level: str = "low"
    
    def __init__(self, project_memory_manager: Optional[Any] = None):
        self._project_memory = project_memory_manager
    
    def set_project_memory(self, manager: Any) -> None:
        """Set the project memory manager."""
        self._project_memory = manager
    
    async def execute(self, **params: Any) -> ToolResult:
        """Execute project memory operation."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"ProjectMemoryTool.execute called with params: {list(params.keys())}")
        
        if self._project_memory is None:
            logger.error("Project memory manager not initialized")
            return ToolResult(
                success=False,
                output=None,
                error="Project memory manager not initialized",
            )
        
        operation = params.get("operation")
        logger.info(f"Operation: {operation}")
        
        try:
            if operation == "create_project":
                return await self._create_project(params)
            elif operation == "load_project":
                return await self._load_project(params)
            elif operation == "unload_project":
                return await self._unload_project(params)
            elif operation == "delete_project":
                return await self._delete_project(params)
            elif operation == "list_projects":
                return await self._list_projects()
            elif operation == "get_project_details":
                return await self._get_project_details(params)
            elif operation == "create_segment":
                return await self._create_segment(params)
            elif operation == "load_segment":
                return await self._load_segment(params)
            elif operation == "unload_segment":
                return await self._unload_segment(params)
            elif operation == "add_to_segment":
                return await self._add_to_segment(params)
            elif operation == "promote_to_global":
                return await self._promote_to_global(params)
            elif operation == "query_memory":
                return await self._query_memory(params)
            elif operation == "search_all_projects":
                return await self._search_all_projects(params)
            elif operation == "get_active_context":
                return await self._get_active_context()
            elif operation == "find_or_open_project":
                return await self._find_or_open_project(params)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}",
                )
        
        except Exception as e:
            logger.exception(f"Project memory operation failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Project memory operation failed: {str(e)}",
            )
    
    def _resolve_project_id(self, project_id: str) -> Optional[str]:
        """Resolve 'current' to active project ID."""
        if project_id == "current":
            return self._project_memory._active_project_id
        return project_id
    
    async def _create_project(self, params: Dict[str, Any]) -> ToolResult:
        """Create a new project."""
        name = params.get("name") or params.get("project_name")
        description = params.get("description", "")
        
        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"_create_project called with params: {params}")
        logger.info(f"Extracted name: '{name}'")
        
        if not name:
            return ToolResult(
                success=False,
                output=None,
                error=f"Project name required. Received params: {list(params.keys())}. Use 'name' or 'project_name' parameter.",
            )
        
        # Check if project already exists
        import hashlib
        project_id = f"proj_{hashlib.sha256(name.encode()).hexdigest()[:16]}"
        if project_id in self._project_memory._projects:
            existing = self._project_memory._projects[project_id]
            return ToolResult(
                success=False,
                output={"existing_project": existing.to_dict()},
                error=f"Project '{name}' already exists (ID: {project_id}). Use 'find_or_open_project' operation to open it, or choose a different name.",
            )
        
        try:
            project = await self._project_memory.create_project(
                name=name,
                description=description,
                initial_context=params.get("content"),  # Optional initial context
            )

            # Auto-load the new project
            await self._project_memory.load_project(project.project_id)

            return ToolResult(
                success=True,
                output={
                    "project_id": project.project_id,
                    "name": project.name,
                    "status": "created_and_loaded",
                    "segments": list(project.segments.keys()),
                },
            )

        except ValueError as e:
            # Project already exists or other validation error
            return ToolResult(
                success=False,
                output=None,
                error=f"Cannot create project: {str(e)}",
            )
        except Exception as e:
            logger.exception(f"Failed to create project: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to create project '{name}': {str(e)}",
            )
    
    async def _load_project(self, params: Dict[str, Any]) -> ToolResult:
        """Load a project into active memory."""
        # Try project_id first, then name/project_name as lookup
        project_id = params.get("project_id")
        if not project_id or project_id == "current":
            # Try to find by name
            name = params.get("name") or params.get("project_name")
            if name and name != "current":
                # Search for project by name
                all_projects = await self._project_memory.get_project_list()
                name_lower = name.lower()
                for proj in all_projects:
                    if proj.get("name", "").lower() == name_lower:
                        project_id = proj["id"]
                        break
        
        project_id = self._resolve_project_id(project_id)
        
        if not project_id:
            return ToolResult(
                success=False,
                output=None,
                error="No project ID provided and no active project. Use 'name' or 'project_name' to search by name.",
            )
        
        project = await self._project_memory.load_project(project_id)
        
        return ToolResult(
            success=True,
            output={
                "project_id": project.project_id,
                "name": project.name,
                "status": "loaded",
                "active_segments": [
                    {
                        "id": seg.segment_id,
                        "type": seg.segment_type,
                        "description": seg.description,
                    }
                    for seg in project.get_active_segments()
                ],
            },
        )
    
    async def _unload_project(self, params: Dict[str, Any]) -> ToolResult:
        """Unload active project."""
        project_id = self._resolve_project_id(params.get("project_id", "current"))
        
        if not project_id:
            return ToolResult(
                success=False,
                output=None,
                error="No active project to unload",
            )
        
        await self._project_memory.unload_project(project_id)
        
        return ToolResult(
            success=True,
            output={
                "project_id": project_id,
                "status": "unloaded",
            },
        )
    
    async def _delete_project(self, params: Dict[str, Any]) -> ToolResult:
        """Delete a project permanently."""
        project_id = self._resolve_project_id(params.get("project_id"))
        
        if not project_id:
            return ToolResult(
                success=False,
                output=None,
                error="Project ID required",
            )
        
        success = await self._project_memory.delete_project(project_id)
        
        return ToolResult(
            success=success,
            output={"project_id": project_id, "deleted": success},
        )
    
    async def _list_projects(self) -> ToolResult:
        """List all projects."""
        projects = await self._project_memory.get_project_list()
        
        return ToolResult(
            success=True,
            output={
                "projects": projects,
                "count": len(projects),
                "active_project": self._project_memory._active_project_id,
            },
        )
    
    async def _get_project_details(self, params: Dict[str, Any]) -> ToolResult:
        """Get detailed info about a project."""
        project_id = self._resolve_project_id(params.get("project_id", "current"))
        
        if not project_id:
            return ToolResult(
                success=False,
                output=None,
                error="No project specified",
            )
        
        details = await self._project_memory.get_project_details(project_id)
        
        return ToolResult(
            success=details is not None,
            output=details or {"error": "Project not found"},
        )
    
    async def _create_segment(self, params: Dict[str, Any]) -> ToolResult:
        """Create a new segment in a project."""
        project_id = self._resolve_project_id(params.get("project_id", "current"))
        
        if not project_id:
            return ToolResult(
                success=False,
                output=None,
                error="No active project",
            )
        
        segment = await self._project_memory.create_segment(
            project_id=project_id,
            segment_name=params.get("segment_name", "New Segment"),
            segment_type=params.get("segment_type", "notes"),
            initial_content=params.get("content"),
        )
        
        # Auto-load the new segment
        await self._project_memory.load_segment(project_id, segment.segment_id)
        
        return ToolResult(
            success=True,
            output={
                "segment_id": segment.segment_id,
                "segment_type": segment.segment_type,
                "status": "created_and_loaded",
            },
        )
    
    async def _load_segment(self, params: Dict[str, Any]) -> ToolResult:
        """Load a segment into RCB."""
        project_id = self._resolve_project_id(params.get("project_id", "current"))
        segment_id = params.get("segment_id")
        
        if not project_id or not segment_id:
            return ToolResult(
                success=False,
                output=None,
                error="Project ID and segment ID required",
            )
        
        success = await self._project_memory.load_segment(project_id, segment_id)
        
        return ToolResult(
            success=success,
            output={
                "project_id": project_id,
                "segment_id": segment_id,
                "loaded": success,
            },
        )
    
    async def _unload_segment(self, params: Dict[str, Any]) -> ToolResult:
        """Unload a segment from RCB."""
        project_id = self._resolve_project_id(params.get("project_id", "current"))
        segment_id = params.get("segment_id")
        
        if not project_id or not segment_id:
            return ToolResult(
                success=False,
                output=None,
                error="Project ID and segment ID required",
            )
        
        success = await self._project_memory.unload_segment(project_id, segment_id)
        
        return ToolResult(
            success=success,
            output={
                "project_id": project_id,
                "segment_id": segment_id,
                "unloaded": success,
            },
        )
    
    async def _add_to_segment(self, params: Dict[str, Any]) -> ToolResult:
        """Add content to a segment."""
        project_id = self._resolve_project_id(params.get("project_id", "current"))
        segment_id = params.get("segment_id")
        content = params.get("content")
        
        if not all([project_id, segment_id, content]):
            return ToolResult(
                success=False,
                output=None,
                error="Project ID, segment ID, and content required",
            )
        
        chunk_id = await self._project_memory.add_to_segment(
            project_id=project_id,
            segment_id=segment_id,
            content=content,
            importance=params.get("importance", 5.0),
        )
        
        return ToolResult(
            success=True,
            output={
                "chunk_id": chunk_id,
                "added_to_segment": segment_id,
            },
        )
    
    async def _promote_to_global(self, params: Dict[str, Any]) -> ToolResult:
        """Promote project content to global memory."""
        project_id = self._resolve_project_id(params.get("project_id", "current"))
        content = params.get("content")
        
        if not content:
            return ToolResult(
                success=False,
                output=None,
                error="Content required",
            )
        
        chunk_id = await self._project_memory.promote_to_global(
            project_id=project_id or "unknown",
            content=content,
            category=params.get("category", "cross_project"),
        )
        
        return ToolResult(
            success=True,
            output={
                "chunk_id": chunk_id,
                "promoted_to": "global_memory",
            },
        )
    
    async def _query_memory(self, params: Dict[str, Any]) -> ToolResult:
        """Query memory within a specific project."""
        project_id = self._resolve_project_id(params.get("project_id", "current"))
        query = params.get("query")
        
        if not project_id or not query:
            return ToolResult(
                success=False,
                output=None,
                error="Project ID and query required",
            )
        
        results = await self._project_memory.query_project_memory(
            project_id=project_id,
            query=query,
            include_inactive=params.get("include_inactive", False),
        )
        
        return ToolResult(
            success=True,
            output={
                "query": query,
                "project_id": project_id,
                "results": results,
                "count": len(results),
            },
        )
    
    async def _search_all_projects(self, params: Dict[str, Any]) -> ToolResult:
        """Search across all projects."""
        query = params.get("query")
        
        if not query:
            return ToolResult(
                success=False,
                output=None,
                error="Query required",
            )
        
        results = await self._project_memory.search_all_projects(query)
        
        return ToolResult(
            success=True,
            output={
                "query": query,
                "results": results,
                "projects_matched": len(results),
            },
        )
    
    async def _get_active_context(self) -> ToolResult:
        """Get complete active memory context."""
        context = await self._project_memory.get_active_memory_context()
        
        return ToolResult(
            success=True,
            output=context,
        )
    
    async def _find_or_open_project(self, params: Dict[str, Any]) -> ToolResult:
        """
        Find existing project by name (fuzzy match) or create new one.
        Implements 'open project' semantics - checks existence first.
        """
        # Accept both 'name' and 'project_name' for flexibility
        name = params.get("name") or params.get("project_name")
        if not name:
            return ToolResult(
                success=False,
                output=None,
                error="Project name required for find_or_open_project (use 'name' or 'project_name' parameter)",
            )
        
        # First, list all projects and search for match
        all_projects = await self._project_memory.get_project_list()
        
        # Case-insensitive exact match first
        name_lower = name.lower()
        for proj in all_projects:
            if proj.get("name", "").lower() == name_lower:
                # Found exact match - load it
                project_id = proj["id"]
                loaded = await self._project_memory.load_project(project_id)
                return ToolResult(
                    success=True,
                    output={
                        "action": "loaded_existing",
                        "project_id": project_id,
                        "name": loaded.name,
                        "description": loaded.description,
                        "status": "loaded",
                        "message": f"Opened existing project '{loaded.name}'",
                        "segments": [
                            {
                                "id": seg.segment_id,
                                "type": seg.segment_type,
                                "description": seg.description,
                            }
                            for seg in loaded.get_active_segments()
                        ],
                    },
                )
        
        # No exact match - check for partial matches
        partial_matches = []
        for proj in all_projects:
            proj_name = proj.get("name", "").lower()
            # Check if name is contained in project name or vice versa
            if name_lower in proj_name or proj_name in name_lower:
                partial_matches.append(proj)
        
        if partial_matches:
            # Return suggestions rather than creating
            return ToolResult(
                success=False,  # Not a failure, but requires user choice
                output={
                    "action": "suggest_existing",
                    "requested_name": name,
                    "similar_projects": [
                        {
                            "id": p["id"],
                            "name": p["name"],
                            "description": p.get("description", "")[:100],
                        }
                        for p in partial_matches[:3]
                    ],
                    "message": f"Did you mean one of these existing projects? Use load_project with the correct ID, or create_project to make a new '{name}' project.",
                },
                error=None,  # Not really an error, just needs user input
            )
        
        # No matches at all - create new project
        description = params.get("description", f"Project: {name}")
        auto_create = params.get("auto_create", True)
        
        if not auto_create:
            return ToolResult(
                success=False,
                output={
                    "action": "not_found",
                    "requested_name": name,
                    "existing_projects": len(all_projects),
                    "message": f"No project named '{name}' found. Set auto_create=true to create it.",
                },
            )
        
        # Create new project
        project = await self._project_memory.create_project(
            name=name,
            description=description,
            initial_context=params.get("content"),
        )
        
        # Auto-load the new project
        await self._project_memory.load_project(project.project_id)
        
        return ToolResult(
            success=True,
            output={
                "action": "created_new",
                "project_id": project.project_id,
                "name": project.name,
                "description": project.description,
                "status": "created_and_loaded",
                "message": f"Created and opened new project '{project.name}'",
                "segments": list(project.segments.keys()),
            },
        )
