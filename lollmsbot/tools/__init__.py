"""
Tools package for LollmsBot.

This package provides tool implementations that can be registered with the Agent.
Tools are the primary way the Agent interacts with external systems and performs actions.

Available tools:
- filesystem: File operations and HTML app generation
- http: Web requests with memory integration
- calendar: Event management
- shell: Command execution (high risk)
- power: Windows power management
- project_memory: Project-based memory management
"""

from lollmsbot.agent import Tool, ToolResult, ToolError

# Import all tool classes
from lollmsbot.tools.filesystem import FilesystemTool
from lollmsbot.tools.http import HttpTool
from lollmsbot.tools.calendar import CalendarTool
from lollmsbot.tools.shell import ShellTool
from lollmsbot.tools.power import PowerTool
from lollmsbot.tools.project_memory import ProjectMemoryTool

# SimplifiedAgant integration tools
try:
    from lollmsbot.tools.crm_tools import CRMQueryTool, MeetingPrepTool
    from lollmsbot.tools.knowledge_tools import KnowledgeQueryTool, IngestContentTool
    from lollmsbot.tools.task_tools import CreateTaskTool, GetTasksTool
    from lollmsbot.tools.youtube_tools import YouTubeReportTool
    from lollmsbot.tools.business_tools import BusinessReportTool
    OPENCLAW_TOOLS_AVAILABLE = True
except ImportError:
    OPENCLAW_TOOLS_AVAILABLE = False

class ToolRegistry:
    """Dynamic registry for tool registration and discovery.
    
    The ToolRegistry provides a centralized way to manage tool instances,
    allowing dynamic registration, lookup, and enumeration of available tools.
    
    Attributes:
        _tools: Dictionary mapping tool names to tool instances.
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(FilesystemTool())
        >>> tool = registry.get("filesystem")
        >>> all_tools = registry.list_tools()
    """
    
    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool in the registry.
        
        Args:
            tool: Tool instance to register.
            
        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> Tool | None:
        """Remove a tool from the registry.
        
        Args:
            tool_name: Name of the tool to remove.
            
        Returns:
            The removed tool if found, None otherwise.
        """
        return self._tools.pop(tool_name, None)
    
    def get(self, tool_name: str) -> Tool | None:
        """Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve.
            
        Returns:
            The tool instance if found, None otherwise.
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> list[Tool]:
        """List all registered tools.
        
        Returns:
            List of all registered tool instances.
        """
        return list(self._tools.values())
    
    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool name is registered.
        
        Args:
            tool_name: Name to check.
            
        Returns:
            True if the tool is registered, False otherwise.
        """
        return tool_name in self._tools
    
    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)
    
    def __repr__(self) -> str:
        return f"ToolRegistry({list(self._tools.keys())})"


def get_default_tools() -> list[Tool]:
    """Get a list of default tool instances.
    
    Returns a list containing instantiated default tools:
    - FilesystemTool: File and directory operations
    - HttpTool: HTTP requests and API calls
    - CalendarTool: Date and time management
    - ShellTool: Safe shell command execution
    - PowerTool: Power management operations
    - ProjectMemoryTool: Project-based memory management
    
    Returns:
        List of default tool instances.
        
    Example:
        >>> tools = get_default_tools()
        >>> for tool in tools:
        ...     print(f"Loaded: {tool.name}")
    """
    return [
        FilesystemTool(),
        HttpTool(),
        CalendarTool(),
        ShellTool(),
        PowerTool(),
        ProjectMemoryTool(),
    ]


__all__ = [
    # Base classes from agent module
    "Tool",
    "ToolResult",
    "ToolError",
    # Tool registry
    "ToolRegistry",
    # Tool classes
    "FilesystemTool",
    "HttpTool",
    "CalendarTool",
    "ShellTool",
    "PowerTool",
    "ProjectMemoryTool",
    # Utility functions
    "get_default_tools",
]


if OPENCLAW_TOOLS_AVAILABLE:
    __all__.extend([
        "CRMQueryTool",
        "MeetingPrepTool",
        "KnowledgeQueryTool",
        "IngestContentTool",
        "CreateTaskTool",
        "GetTasksTool",
        "YouTubeReportTool",
        "BusinessReportTool",
    ])
