"""
LollmsBot - AI Agent Framework

A modular AI assistant with:
- Configurable LLM backends
- Tool system for capabilities
- Skill framework for reusable workflows
- Memory and identity management
- Security and ethics layers
"""

__version__ = "0.2.0"

# Core agent exports - now properly imported from config
from lollmsbot.agent.config import (
    AgentState,
    PermissionLevel,
    Tool,
    ToolResult,
    SkillResult,
    UserPermissions,
    ConversationTurn,
    ToolError,
    AgentError,
)
from lollmsbot.agent import Agent

# Configuration
from lollmsbot.config import BotConfig, LollmsSettings, GatewaySettings

# Optional components (import as needed)
from lollmsbot.agent.memory import MemoryManager
from lollmsbot.agent.identity import IdentityDetector
from lollmsbot.agent.llm import PromptBuilder
from lollmsbot.agent.tools import ToolParser, FileGenerator
from lollmsbot.agent.logging import AgentLogger

__all__ = [
    # Version
    "__version__",
    
    # Core agent
    "Agent",
    "AgentState",
    "PermissionLevel",
    "Tool",
    "ToolResult",
    "SkillResult",
    "UserPermissions",
    "ConversationTurn",
    "ToolError",
    "AgentError",
    
    # Configuration
    "BotConfig",
    "LollmsSettings",
    "GatewaySettings",
    
    # Components (optional direct use)
    "MemoryManager",
    "IdentityDetector",
    "PromptBuilder",
    "ToolParser",
    "FileGenerator",
    "AgentLogger",
]
