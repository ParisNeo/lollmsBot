"""
Integration layer to add simplified_agent-style capabilities to lollmsbot's Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from lollmsbot.agent import Agent, Tool, ToolResult
from lollmsbot.agent.simplified_agant_style import SimplifiedAgantStyle, CodeExtension


class simplified_agentTool(Tool):
    """
    Tool that bridges to simplified_agent-style minimal tool execution.
    
    This wraps the 4 core tools (read, write, edit, bash) and
    any self-written extensions.
    """
    
    name: str = "simplified_agant"
    description: str = (
        "SimplifiedAgant-style minimal tool execution. Use for file operations, "
        "code editing, shell commands, or any self-written extensions. "
        "Core tools: read, write, edit, bash. Extensions: dynamic."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "edit", "bash", "create_extension", "list_extensions", "branch", "hot_reload"],
                "description": "The operation to perform",
            },
            "path": {
                "type": "string",
                "description": "File path for read/write/edit operations",
            },
            "content": {
                "type": "string",
                "description": "Content for write operations or extension code",
            },
            "old_string": {
                "type": "string",
                "description": "Text to replace in edit operations",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text in edit operations",
            },
            "command": {
                "type": "string",
                "description": "Shell command for bash operations",
            },
            "extension_name": {
                "type": "string",
                "description": "Name for extension operations",
            },
            "extension_description": {
                "type": "string",
                "description": "Description for new extension",
            },
            "test_command": {
                "type": "string",
                "description": "Command to test new extension",
            },
        },
        "required": ["operation"],
    }
    
    risk_level: str = "high"  # Bash is high risk
    
    def __init__(self, openclaw_agent: SimplifiedAgantStyle):
        self._openclaw = openclaw_agent
    
    async def execute(self, **params: Any) -> ToolResult:
        """Execute SimplifiedAgant operation."""
        operation = params.get("operation")
        
        if operation == "read":
            return await self._openclaw.execute_core_tool("read", **{
                k: v for k, v in params.items() 
                if k in ["path", "offset", "limit"]
            })
        
        elif operation == "write":
            return await self._openclaw.execute_core_tool("write", **{
                k: v for k, v in params.items()
                if k in ["path", "content", "append"]
            })
        
        elif operation == "edit":
            return await self._openclaw.execute_core_tool("edit", **{
                k: v for k, v in params.items()
                if k in ["path", "old_string", "new_string"]
            })
        
        elif operation == "bash":
            return await self._openclaw.execute_core_tool("bash", **{
                k: v for k, v in params.items()
                if k in ["command", "timeout", "cwd"]
            })
        
        elif operation == "create_extension":
            try:
                extension = await self._openclaw.create_extension(
                    name=params.get("extension_name", "unnamed"),
                    description=params.get("extension_description", ""),
                    code=params.get("content", ""),
                    test_command=params.get("test_command"),
                )
                return ToolResult(
                    success=True,
                    output={
                        "extension_name": extension.name,
                        "version": extension.version,
                        "is_active": extension.is_active,
                        "test_results": extension.test_results,
                    }
                )
            except Exception as e:
                return ToolResult(success=False, output=None, error=str(e))
        
        elif operation == "list_extensions":
            return ToolResult(
                success=True,
                output={
                    "extensions": [
                        {
                            "name": ext.name,
                            "description": ext.description,
                            "version": ext.version,
                            "is_active": ext.is_active,
                        }
                        for ext in self._openclaw.extensions.values()
                    ],
                    "count": len(self._openclaw.extensions),
                }
            )
        
        elif operation == "branch":
            branch = self._openclaw.branch_session(
                summary=params.get("extension_description", "New branch")
            )
            return ToolResult(
                success=True,
                output={
                    "branch_id": branch.branch_id,
                    "parent": branch.parent_branch_id,
                    "active_branch": self._openclaw.active_branch_id,
                    "tree": self._openclaw.get_branch_tree(),
                }
            )
        
        elif operation == "hot_reload":
            result = await self._openclaw.hot_reload(
                params.get("extension_name")
            )
            return ToolResult(success=True, output=result)
        
        else:
            # Try as extension name
            return await self._openclaw.execute_core_tool(operation, **params)


def integrate_simplified_agent(agent: Agent, working_dir: Optional[Path] = None) -> SimplifiedAgantStyle:
    """
    Integrate simplified_agent-style capabilities into an existing lollmsbot Agent.
    
    Returns the SimplifiedAgantStyle instance for direct access.
    """
    # Create simplified_agent workspace
    oc_working_dir = working_dir or Path.home() / ".lollmsbot" / "simplified_agent"
    
    # Create simplified_agent agent with access to lollmsbot's memory
    simplified_agent = SimplifiedAgantStyle(
        working_dir=oc_working_dir,
        memory_manager=agent._memory if hasattr(agent, '_memory') else None,
    )
    
    # Create and register the bridge tool
    simplified_agent_tool = simplified_agentTool(simplified_agent)
    
    # Register with lollmsbot agent
    # Note: This would need to be async in real usage
    # For now, return the tool for manual registration
    
    return simplified_agent, simplified_agent_tool
