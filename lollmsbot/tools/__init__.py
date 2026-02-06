"""
Tools package for LollmsBot.

This package provides a collection of built-in tools for the Agent framework,
including filesystem operations, HTTP requests, calendar management, and
shell command execution. It also provides the ToolRegistry for dynamic
tool registration and discovery.

Example:
    >>> from lollmsbot.tools import get_default_tools, ToolRegistry
    >>> tools = get_default_tools()
    >>> registry = ToolRegistry()
    >>> for tool in tools:
    ...     registry.register(tool)
"""

from lollmsbot.agent import Tool, ToolResult, ToolError

# Import all tool classes
from lollmsbot.tools.filesystem import FilesystemTool
from lollmsbot.tools.http import HttpTool
from lollmsbot.tools.calendar import CalendarTool
from lollmsbot.tools.shell import ShellTool


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
    # Utility functions
    "get_default_tools",
]
