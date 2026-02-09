"""
Agent configuration dataclasses and enums.

Contains all type definitions, enums, and dataclasses used by the Agent system.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from lollmsbot.guardian import SecurityEvent


class AgentState(Enum):
    """Enumeration of possible agent states."""
    IDLE = auto()
    PROCESSING = auto()
    ERROR = auto()
    QUARANTINED = auto()
    LEARNING = auto()


class PermissionLevel(Enum):
    """Permission levels for users."""
    NONE = auto()
    BASIC = auto()
    TOOLS = auto()
    SKILLS = auto()
    SKILL_CREATE = auto()
    ADMIN = auto()


class ToolError(Exception):
    """Exception raised when a tool execution fails."""
    pass


class AgentError(Exception):
    """Exception raised when agent processing fails."""
    pass


class Tool(ABC):
    """Abstract base class for tools that can be registered with the Agent.
    
    All tools must inherit from this class and implement the execute method.
    Tools are the primary way the Agent interacts with the external world.
    
    Attributes:
        name: Unique identifier for the tool (lowercase, no spaces).
        description: Human-readable description of what the tool does.
        parameters: JSON Schema describing the tool's parameters.
        risk_level: Risk classification for security purposes.
    """
    
    name: str = "abstract_tool"
    description: str = "Abstract tool - do not use directly"
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    risk_level: str = "low"  # low, medium, high, critical
    
    @abstractmethod
    async def execute(self, **params: Any) -> "ToolResult":
        """Execute the tool with the given parameters.
        
        Args:
            **params: Parameters as defined in the tool's JSON schema.
            
        Returns:
            ToolResult indicating success/failure with output data.
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's JSON schema for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
    
    def __repr__(self) -> str:
        return f"Tool({self.name}, risk={self.risk_level})"


@dataclass
class UserPermissions:
    """Permissions configuration for a specific user."""
    level: PermissionLevel = PermissionLevel.BASIC
    allowed_tools: Optional[Set[str]] = None
    allowed_skills: Optional[Set[str]] = None
    denied_tools: Set[str] = field(default_factory=set)
    denied_skills: Set[str] = field(default_factory=set)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    files_to_send: List[Dict[str, Any]] = field(default_factory=list)
    security_event: Optional[SecurityEvent] = None


@dataclass
class SkillResult:
    """Result from skill execution."""
    success: bool
    result: Any
    skill_name: str
    skill_version: str
    execution_id: str
    steps_taken: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    skills_called: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ConversationTurn:
    """Single turn in conversation history."""
    timestamp: datetime = field(default_factory=datetime.now)
    user_message: str = ""
    agent_response: str = ""
    tools_used: List[str] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_screened: bool = True
    security_flags: List[str] = field(default_factory=list)
    importance_score: float = 1.0  # 1.0 = normal, higher = more important
    memory_tags: List[str] = field(default_factory=list)  # e.g., ["creator_identity", "personal_info"]


ToolEventCallback = Callable[[str, str, Optional[Dict[str, Any]]], Awaitable[None]]
