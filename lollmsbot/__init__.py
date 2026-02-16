"""
LollmsBot - AI Agent Framework

A modular AI assistant with:
- Configurable LLM backends
- Tool system for capabilities
- Skill framework for reusable workflows
- Memory and identity management
- Security and ethics layers
- Document management for long-form writing
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
from lollmsbot.agent.integrated_document_agent import IntegratedDocumentAgent

# Document management
from lollmsbot.document_manager import (
    DocumentManager,
    DocumentIndex,
    DocumentBlock,
    BookProject,
    ContextLens,
    BlockType,
    create_document_manager,
)

# Writing tools
from lollmsbot.writing_tools import (
    IngestDocumentTool,
    CreateBookProjectTool,
    CreateOutlineTool,
    GetDocumentContextTool,
    WriteSectionTool,
    SubmitWrittenContentTool,
    SearchReferencesTool,
    GetWritingProgressTool,
    get_writing_tools,
)

# Configuration
from lollmsbot.config import BotConfig, LollmsSettings, GatewaySettings

# Optional components (import as needed)
from lollmsbot.agent.memory import MemoryManager
from lollmsbot.agent.identity import IdentityDetector
from lollmsbot.agent.llm import PromptBuilder
from lollmsbot.agent.tools import ToolParser, FileGenerator
from lollmsbot.agent.logging import AgentLogger
from lollmsbot.agent.rlm.manager import (
    RLMMemoryManager,
    MemoryMap,
)
from lollmsbot.agent.rlm.memory_map import (
    MemoryCategorySummary,
    MemoryAnchor,
    KnowledgeGap
)

# NEW: Power management
from lollmsbot.power_management import (
    PowerManager,
    PowerState,
    ExecutionState,
    get_power_manager,
)

# NEW: Project Memory
from lollmsbot.agent.project_memory import (
    ProjectMemoryManager,
    Project,
    MemorySegment,
    ProjectStatus,
)

# NEW: SimplifiedAgant Integration (when available)
try:
    from lollmsbot.openclaw_integration import SimplifiedAgantIntegration
    from lollmsbot.crm import CRMManager, Contact, ContactRole
    from lollmsbot.knowledge_base import KnowledgeBaseManager, KnowledgeEntry
    from lollmsbot.task_manager import TaskManager, Task, TaskPriority
    from lollmsbot.youtube_analytics import YouTubeAnalyticsManager
    from lollmsbot.business_analysis import BusinessAnalysisCouncil, CouncilReport
    OPENCLAW_AVAILABLE = True
except ImportError:
    OPENCLAW_AVAILABLE = False

__all__ = [
    # Version
    "__version__",
    
    # Core agent
    "Agent",
    "IntegratedDocumentAgent",
    "AgentState",
    "PermissionLevel",
    "Tool",
    "ToolResult",
    "SkillResult",
    "UserPermissions",
    "ConversationTurn",
    "ToolError",
    "AgentError",
    
    # Project Memory
    "ProjectMemoryManager",
    "Project",
    "MemorySegment",
    "ProjectStatus",
    
    # Document management
    "DocumentManager",
    "DocumentIndex",
    "DocumentBlock",
    "BookProject",
    "ContextLens",
    "BlockType",
    "create_document_manager",
    
    # Writing tools
    "IngestDocumentTool",
    "CreateBookProjectTool",
    "CreateOutlineTool",
    "GetDocumentContextTool",
    "WriteSectionTool",
    "SubmitWrittenContentTool",
    "SearchReferencesTool",
    "GetWritingProgressTool",
    "get_writing_tools",
    
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
    
    # RLM Memory
    "RLMMemoryManager",
    "MemoryMap",
    "MemoryCategorySummary",
    "MemoryAnchor",
    "KnowledgeGap",
    
    # NEW: Power management
    "PowerManager",
    "PowerState", 
    "ExecutionState",
    "get_power_manager",
]

# Add SimplifiedAgant exports if available
if OPENCLAW_AVAILABLE:
    __all__.extend([
        "SimplifiedAgantIntegration",
        "CRMManager",
        "Contact",
        "ContactRole",
        "KnowledgeBaseManager",
        "KnowledgeEntry",
        "TaskManager",
        "Task",
        "TaskPriority",
        "YouTubeAnalyticsManager",
        "BusinessAnalysisCouncil",
        "CouncilReport",
    ])
