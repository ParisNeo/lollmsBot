"""
Integration layer connecting SimplifiedAgantStyle with LollmsBot Agent.

Provides:
- Tool registration for Pi-style operations
- Session management integration
- Daily workflow automation (YouTube, CRM, Business Analysis)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from lollmsbot.agent import Tool, ToolResult

logger = logging.getLogger(__name__)


class SimplifiedAgantTool(Tool):
    """
    Unified tool for all OpenClaw-style operations.
    
    Single tool that handles: read, write, edit, bash, extension management,
    and session branching.
    """
    
    name: str = "simplified_agant"
    description: str = (
        "OpenClaw-style unified tool for minimal operations and extension management. "
        "Operations: read, write, edit, bash, create_extension, list_extensions, "
        "reload_extensions, create_branch, switch_branch, get_branches."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "read", "write", "edit", "bash",
                    "create_extension", "list_extensions", "reload_extensions",
                    "create_branch", "switch_branch", "get_branches", "get_branch_context",
                ],
                "description": "The operation to perform",
            },
            "args": {
                "type": "object",
                "description": "Arguments specific to the operation",
            },
        },
        "required": ["operation"],
    }
    
    risk_level: str = "medium"  # Can execute bash commands
    
    def __init__(self):
        self._style: Optional[Any] = None
    
    def set_style(self, style: Any) -> None:
        """Set the SimplifiedAgantStyle instance."""
        self._style = style
    
    async def execute(self, operation: str, args: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        """Execute OpenClaw-style operation."""
        if self._style is None:
            return ToolResult(
                success=False,
                output=None,
                error="SimplifiedAgantStyle not initialized",
            )
        
        args = args or {}
        
        try:
            if operation == "read":
                return await self._execute_read(**args)
            elif operation == "write":
                return await self._execute_write(**args)
            elif operation == "edit":
                return await self._execute_edit(**args)
            elif operation == "bash":
                return await self._execute_bash(**args)
            elif operation == "create_extension":
                return await self._execute_create_extension(**args)
            elif operation == "list_extensions":
                return await self._execute_list_extensions()
            elif operation == "reload_extensions":
                return await self._execute_reload_extensions(**args)
            elif operation == "create_branch":
                return await self._execute_create_branch(**args)
            elif operation == "switch_branch":
                return await self._execute_switch_branch(**args)
            elif operation == "get_branches":
                return await self._execute_get_branches()
            elif operation == "get_branch_context":
                return await self._execute_get_branch_context(**args)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}",
                )
        
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Operation failed: {str(e)}",
            )
    
    async def _execute_read(self, source: str, limit: Optional[int] = None, **kwargs) -> ToolResult:
        """Execute read operation."""
        import aiohttp
        import aiofiles
        
        try:
            # Check if URL or file
            if source.startswith(('http://', 'https://')):
                async with aiohttp.ClientSession() as session:
                    async with session.get(source, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        content = await response.text()
                        if limit:
                            content = content[:limit]
                        return ToolResult(
                            success=True,
                            output={"content": content, "source": source, "type": "url"},
                        )
            else:
                # File read
                async with aiofiles.open(source, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    if limit:
                        lines = content.split('\n')[:limit]
                        content = '\n'.join(lines)
                    return ToolResult(
                        success=True,
                        output={"content": content, "source": source, "type": "file"},
                    )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Read failed: {str(e)}",
            )
    
    async def _execute_write(self, path: str, content: str, **kwargs) -> ToolResult:
        """Execute write operation."""
        import aiofiles
        
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            return ToolResult(
                success=True,
                output={"path": path, "bytes_written": len(content.encode('utf-8'))},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Write failed: {str(e)}",
            )
    
    async def _execute_edit(self, path: str, old_string: str, new_string: str, **kwargs) -> ToolResult:
        """Execute edit operation."""
        import aiofiles
        
        try:
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            if old_string not in content:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"String not found in file: {old_string[:50]}...",
                )
            
            new_content = content.replace(old_string, new_string, 1)
            
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(new_content)
            
            return ToolResult(
                success=True,
                output={
                    "path": path,
                    "replacements": 1,
                    "old_length": len(old_string),
                    "new_length": len(new_string),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Edit failed: {str(e)}",
            )
    
    async def _execute_bash(self, command: str, working_dir: Optional[str] = None, **kwargs) -> ToolResult:
        """Execute bash command."""
        import asyncio
        import os
        
        try:
            # Security: Basic command validation
            dangerous = ['rm -rf /', 'mkfs', 'dd if=', '> /dev/sda', 'shutdown', 'reboot']
            for d in dangerous:
                if d in command.lower():
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Command blocked for security: contains '{d}'",
                    )
            
            cwd = working_dir or os.getcwd()
            
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            
            stdout, stderr = await proc.communicate()
            
            return ToolResult(
                success=proc.returncode == 0,
                output={
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "returncode": proc.returncode,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Bash execution failed: {str(e)}",
            )
    
    async def _execute_create_extension(self, name: str, code: str, description: str = "", **kwargs) -> ToolResult:
        """Create a new extension."""
        if self._style is None:
            return ToolResult(
                success=False,
                output=None,
                error="SimplifiedAgantStyle not initialized. Please wait for initialization to complete.",
            )
        try:
            ext = self._style.create_extension(name, code, description)
            return ToolResult(
                success=True,
                output={
                    "extension": ext.to_dict(),
                    "message": f"Created extension '{name}'. Use /reload to activate.",
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )
    
    async def _execute_list_extensions(self) -> ToolResult:
        """List all extensions."""
        if self._style is None:
            return ToolResult(
                success=False,
                output=None,
                error="SimplifiedAgantStyle not initialized. Please wait for initialization to complete.",
            )
        extensions = self._style.list_extensions()
        return ToolResult(
            success=True,
            output={"extensions": extensions, "count": len(extensions)},
        )
    
    async def _execute_reload_extensions(self, name: Optional[str] = None, **kwargs) -> ToolResult:
        """Reload extensions."""
        if self._style is None:
            return ToolResult(
                success=False,
                output=None,
                error="SimplifiedAgantStyle not initialized. Please wait for initialization to complete.",
            )
        result = await self._style.hot_reload(name)
        return ToolResult(
            success=len(result.get("failed", [])) == 0,
            output=result,
        )
    
    async def _execute_create_branch(self, summary: str = "", **kwargs) -> ToolResult:
        """Create a new branch."""
        if self._style is None:
            return ToolResult(
                success=False,
                output=None,
                error="SimplifiedAgantStyle not initialized. Please wait for initialization to complete.",
            )
        branch = self._style.create_branch(summary)
        return ToolResult(
            success=True,
            output={"branch": branch.to_dict()},
        )
    
    async def _execute_switch_branch(self, branch_id: str, **kwargs) -> ToolResult:
        """Switch to a branch."""
        if self._style is None:
            return ToolResult(
                success=False,
                output=None,
                error="SimplifiedAgantStyle not initialized. Please wait for initialization to complete.",
            )
        branch = self._style.switch_branch(branch_id)
        if branch:
            return ToolResult(
                success=True,
                output={"branch": branch.to_dict()},
            )
        return ToolResult(
            success=False,
            output=None,
            error=f"Branch {branch_id} not found",
        )
    
    async def _execute_get_branches(self) -> ToolResult:
        """Get branch tree."""
        if self._style is None:
            return ToolResult(
                success=False,
                output=None,
                error="SimplifiedAgantStyle not initialized. Please wait for initialization to complete.",
            )
        tree = self._style.get_branch_tree()
        return ToolResult(
            success=True,
            output=tree,
        )
    
    async def _execute_get_branch_context(self, branch_id: Optional[str] = None, **kwargs) -> ToolResult:
        """Get branch conversation context."""
        if self._style is None:
            return ToolResult(
                success=False,
                output=None,
                error="SimplifiedAgantStyle not initialized. Please wait for initialization to complete.",
            )
        context = self._style.get_branch_context(branch_id)
        return ToolResult(
            success=True,
            output={"context": context, "message_count": len(context)},
        )


class SimplifiedAgantIntegration:
    """
    High-level integration for OpenClaw-style features.
    
    Manages:
    - Daily workflows (YouTube, CRM, Business Analysis)
    - Session branching
    - Extension lifecycle
    - TUI display
    """
    
    def __init__(self, agent: Any):
        self.agent = agent
        self.style: Optional[Any] = None
        self.tool: Optional[SimplifiedAgantTool] = None
        
        # Workflow components (lazy loaded)
        self._youtube: Optional[Any] = None
        self._crm: Optional[Any] = None
        self._business_council: Optional[Any] = None
        self._knowledge_base: Optional[Any] = None
        self._task_manager: Optional[Any] = None
    
    async def initialize(self) -> None:
        """Initialize OpenClaw-style components."""
        from lollmsbot.agent.simplified_agant_style import SimplifiedAgantStyle
        
        self.style = SimplifiedAgantStyle(
            agent=self.agent,
            enable_tui=True,
        )
        
        self.tool = SimplifiedAgantTool()
        self.tool.set_style(self.style)
        
        # Create initial branch
        self.style.create_branch("Main conversation")
        
        logger.info("SimplifiedAgantIntegration initialized")
    
    def get_tool(self) -> SimplifiedAgantTool:
        """Get the unified tool for registration."""
        if self.tool is None:
            raise RuntimeError("SimplifiedAgantIntegration not initialized. Call initialize() first.")
        return self.tool
    
    def get_system_prompt_addon(self) -> str:
        """Get system prompt addon if style is initialized."""
        if self.style:
            return self.style.get_system_prompt_addon()
        return ""
    
    # ========== DAILY WORKFLOW (OpenClaw-style automation) ==========
    
    async def daily_workflow(self) -> Dict[str, Any]:
        """
        Execute daily OpenClaw-style workflow.
        
        Inspired by OpenClaw's autonomous capabilities:
        1. Check YouTube analytics
        2. Process CRM tasks
        3. Run business analysis
        4. Update knowledge base
        """
        results = {
            "steps_completed": [],
            "youtube_snapshot": None,
            "crm_tasks": [],
            "business_analysis": None,
            "errors": [],
        }
        
        # Step 1: YouTube Analytics (if available)
        try:
            if self._youtube is None:
                try:
                    from lollmsbot.youtube_analytics import YouTubeAnalyticsManager
                    self._youtube = YouTubeAnalyticsManager(
                        memory_manager=self.agent._memory if self.agent else None
                    )
                except ImportError:
                    pass
            
            if self._youtube:
                snapshot = await self._youtube.get_channel_snapshot()
                results["youtube_snapshot"] = snapshot
                results["steps_completed"].append("youtube_analytics")
        except Exception as e:
            results["errors"].append(f"YouTube: {str(e)}")
        
        # Step 2: CRM Tasks (if available)
        try:
            if self._crm is None:
                try:
                    from lollmsbot.crm import CRMManager
                    self._crm = CRMManager(
                        memory_manager=self.agent._memory if self.agent else None
                    )
                except ImportError:
                    pass
            
            if self._crm:
                tasks = await self._crm.get_today_tasks()
                results["crm_tasks"] = tasks
                results["steps_completed"].append("crm_tasks")
        except Exception as e:
            results["errors"].append(f"CRM: {str(e)}")
        
        # Step 3: Business Analysis (if available)
        try:
            if self._business_council is None:
                try:
                    from lollmsbot.business_analysis import BusinessAnalysisCouncil
                    self._business_council = BusinessAnalysisCouncil(
                        memory_manager=self.agent._memory if self.agent else None
                    )
                except ImportError:
                    pass
            
            if self._business_council:
                analysis = await self._business_council.generate_daily_report()
                results["business_analysis"] = analysis
                results["steps_completed"].append("business_analysis")
        except Exception as e:
            results["errors"].append(f"Business: {str(e)}")
        
        # Step 4: Knowledge Base Update (if available)
        try:
            if self._knowledge_base is None:
                try:
                    from lollmsbot.knowledge_base import KnowledgeBaseManager
                    self._knowledge_base = KnowledgeBaseManager(
                        memory_manager=self.agent._memory if self.agent else None
                    )
                except ImportError:
                    pass
            
            if self._knowledge_base:
                # Ingest any pending sources
                ingested = await self._knowledge_base.process_pending()
                results["steps_completed"].append(f"knowledge_base ({ingested} items)")
        except Exception as e:
            results["errors"].append(f"Knowledge: {str(e)}")
        
        return results
    
    def get_branch_tree_display(self) -> str:
        """Get formatted branch tree for display."""
        if not self.style:
            return "Branching not initialized"
        
        tree = self.style.get_branch_tree()
        
        if self.style.enable_tui and self.style.console:
            self.style.display_tree(tree, title="Session Branches")
            return ""
        else:
            # Text fallback
            lines = ["Session Branches:"]
            for node in tree.get("tree", []):
                self._format_branch_node(lines, node, 0)
            return "\n".join(lines)
    
    def _format_branch_node(self, lines: List[str], node: Dict[str, Any], depth: int) -> None:
        """Recursively format branch node."""
        indent = "  " * depth
        marker = "● " if node.get("is_active") else "○ "
        lines.append(f"{indent}{marker}{node.get('id', 'unknown')}: {node.get('summary', '')[:50]}")
        
        for child in node.get("children", []):
            self._format_branch_node(lines, child, depth + 1)


# Integration helper
def integrate_simplified_agant(agent: Any) -> Tuple[SimplifiedAgantIntegration, SimplifiedAgantTool]:
    """
    Integrate OpenClaw-style features into LollmsBot Agent.
    
    Returns:
        Tuple of (integration, tool) for registration
    """
    integration = SimplifiedAgantIntegration(agent)
    return integration, integration.get_tool()