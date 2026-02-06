"""
Agent module for LollmsBot - Skills-Integrated Edition

Uses lazy imports to avoid circular dependency with skills module.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable, cast, TYPE_CHECKING

from lollmsbot.config import BotConfig
from lollmsbot.lollms_client import LollmsClient, build_lollms_client
from lollmsbot.guardian import get_guardian, Guardian, SecurityEvent, ThreatLevel

# Rich imports for colored logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.color import Color
import logging

if TYPE_CHECKING:
    # Only for type checking, not runtime
    from lollmsbot.skills import Skill, SkillRegistry, SkillExecutor, SkillLearner, SkillComplexity


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


class Tool(ABC):
    """Abstract base class for agent tools."""
    
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise ValueError(f"Tool {cls.__name__} must define a 'name' attribute")
        if not cls.description:
            raise ValueError(f"Tool {cls.__name__} must define a 'description' attribute")
    
    @abstractmethod
    async def execute(self, **params: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def __repr__(self) -> str:
        return f"Tool({self.name}, risk={self.risk_level})"


ToolEventCallback = Callable[[str, str, Optional[Dict[str, Any]]], Awaitable[None]]


class Agent:
    """
    Core AI agent with Skills and Guardian integration.
    Enhanced with detailed colored logging for debugging.
    """
    
    def __init__(
        self,
        config: Optional[BotConfig] = None,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        default_permissions: PermissionLevel = PermissionLevel.BASIC,
        enable_guardian: bool = True,
        enable_skills: bool = True,
        verbose_logging: bool = True,
    ) -> None:
        self.config: BotConfig = config or BotConfig()
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.name: str = name or self.config.name
        
        # Initialize colored console for debugging
        self._console = Console()
        self._verbose_logging = verbose_logging
        
        # Guardian integration
        self._guardian_enabled = enable_guardian
        self._guardian: Optional[Guardian] = None
        if enable_guardian:
            self._guardian = get_guardian()
        
        # Skills integration - LAZY initialization
        self._skills_enabled = enable_skills
        self._skill_registry: Any = None  # Optional['SkillRegistry']
        self._skill_executor: Any = None   # Optional['SkillExecutor']
        self._skill_learner: Any = None    # Optional['SkillLearner']
        
        self._state: AgentState = AgentState.IDLE
        self._tools: Dict[str, Tool] = {}
        self._memory: Dict[str, Any] = {
            "conversation_history": cast(List[ConversationTurn], []),
            "working_memory": cast(Dict[str, Any], {}),
            "context": cast(Dict[str, Any], {}),
        }
        
        self._user_histories: Dict[str, List[ConversationTurn]] = {}
        self._user_permissions: Dict[str, UserPermissions] = {}
        self._default_permissions: UserPermissions = UserPermissions(level=default_permissions)
        
        # Initialize LoLLMS client for actual LLM generation
        self._lollms_client: Optional[LollmsClient] = None
        self._lollms_client_initialized = False
        
        self._state_lock: asyncio.Lock = asyncio.Lock()
        self._tool_lock: asyncio.Lock = asyncio.Lock()
        self._skill_lock: asyncio.Lock = asyncio.Lock()
        self._permission_lock: asyncio.Lock = asyncio.Lock()
        
        self._file_delivery_callback: Optional[Callable[[str, List[Dict[str, Any]]], Awaitable[bool]]] = None
        self._tool_event_callback: Optional[ToolEventCallback] = None
        self._skill_event_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[None]]] = None
        
        # Standard logger for file logging
        self._logger = logging.getLogger(__name__)
        
        # Print startup banner
        if self._verbose_logging:
            self._log_startup()
        
        if self._guardian and self._guardian.is_quarantined:
            self._state = AgentState.QUARANTINED
            self._log_critical("🚨 Agent initialized in QUARANTINED state")
    
    def _log(self, message: str, style: str = "white", emoji: str = "", level: str = "info") -> None:
        """Internal colored logging method."""
        if not self._verbose_logging:
            return
        
        # Map style names to rich styles
        style_map = {
            "blue": "bold blue",
            "cyan": "bold cyan",
            "green": "bold green",
            "yellow": "bold yellow",
            "orange": "bold rgb(255,165,0)",
            "red": "bold red",
            "magenta": "bold magenta",
            "purple": "bold purple",
            "dim": "dim",
            "white": "white",
        }
        
        rich_style = style_map.get(style, style)
        prefix = f"{emoji} " if emoji else ""
        formatted = f"{prefix}[{rich_style}]{message}[/{rich_style}]"
        
        self._console.print(formatted)
        
        # Also log to standard logger
        if level == "error":
            self._logger.error(message)
        elif level == "warning":
            self._logger.warning(message)
        elif level == "critical":
            self._logger.critical(message)
        else:
            self._logger.info(message)
    
    def _log_startup(self) -> None:
        """Print agent startup banner."""
        panel = Panel.fit(
            f"[bold cyan]🤖 {self.name}[/bold cyan]\n"
            f"[dim]Agent ID: {self.agent_id}[/dim]\n"
            f"[dim]Guardian: {'✅' if self._guardian else '❌'} | "
            f"Skills: {'✅' if self._skills_enabled else '❌'}[/dim]",
            title="[bold blue]Agent Initialized[/bold blue]",
            border_style="blue"
        )
        self._console.print(panel)
    
    def _log_command_received(self, user_id: str, message: str, context: Optional[Dict[str, Any]]) -> None:
        """Log when a command is received."""
        channel = context.get("channel", "unknown") if context else "unknown"
        msg_preview = message[:100] + "..." if len(message) > 100 else message
        
        panel = Panel(
            f"[bold white]User:[/bold white] [cyan]{user_id}[/cyan]\n"
            f"[bold white]Channel:[/bold white] [yellow]{channel}[/yellow]\n"
            f"[bold white]Message:[/bold white] [white]{msg_preview}[/white]\n"
            f"[dim]Length: {len(message)} chars | Tools available: {len(self._tools)}[/dim]",
            title="[bold blue]📥 COMMAND RECEIVED[/bold blue]",
            border_style="blue"
        )
        self._console.print(panel)
    
    def _log_security_check(self, is_safe: bool, event: Optional[SecurityEvent]) -> None:
        """Log security screening results."""
        if is_safe:
            self._log("✅ Input passed security screening", "green", "🛡️")
        else:
            self._log(f"🚫 SECURITY BLOCK: {event.description if event else 'Unknown violation'}", "red", "🛡️", "warning")
    
    def _log_tool_detection(self, tool_count: int, tools_found: List[str]) -> None:
        """Log tool detection in LLM response."""
        if tool_count > 0:
            panel = Panel(
                f"[bold white]Found {tool_count} tool call(s):[/bold white]\n" +
                "\n".join([f"  [purple]• {t}[/purple]" for t in tools_found]),
                title="[bold purple]🔧 TOOLS DETECTED[/bold purple]",
                border_style="purple"
            )
            self._console.print(panel)
        else:
            self._log("No tool calls detected in response", "dim", "🔧")
    
    def _log_tool_execution(self, tool_name: str, params: Dict[str, Any], success: bool, error: Optional[str] = None) -> None:
        """Log tool execution with parameters."""
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])[:200]
        
        if success:
            self._log(f"✅ Tool '{tool_name}' executed successfully", "green", "🔧")
            self._log(f"   Parameters: {params_str}", "dim")
        else:
            self._log(f"❌ Tool '{tool_name}' failed: {error}", "red", "🔧", "error")
    
    def _log_llm_call(self, prompt_length: int, system_prompt: str) -> None:
        """Log LLM invocation."""
        sys_preview = system_prompt[:80] + "..." if len(system_prompt) > 80 else system_prompt
        
        panel = Panel(
            f"[bold white]Prompt length:[/bold white] [cyan]{prompt_length}[/cyan] chars\n"
            f"[bold white]System prompt:[/bold white] [dim]{sys_preview}[/dim]",
            title="[bold orange]🧠 LLM CALL[/bold orange]",
            border_style="rgb(255,165,0)"
        )
        self._console.print(panel)
    
    def _log_llm_response(self, response_length: int, has_tools: bool) -> None:
        """Log LLM response received."""
        tool_status = "🟢 contains tools" if has_tools else "🔵 text only"
        self._log(f"📤 LLM response: {response_length} chars ({tool_status})", "orange")
    
    def _log_file_generation(self, file_count: int, filenames: List[str]) -> None:
        """Log file generation events."""
        if file_count > 0:
            panel = Panel(
                f"[bold white]Generated {file_count} file(s):[/bold white]\n" +
                "\n".join([f"  [green]📄 {name}[/green]" for name in filenames]),
                title="[bold green]📦 FILES CREATED[/bold green]",
                border_style="green"
            )
            self._console.print(panel)
    
    def _log_state_change(self, old_state: AgentState, new_state: AgentState, reason: str = "") -> None:
        """Log agent state transitions."""
        reason_str = f" ({reason})" if reason else ""
        self._log(f"🔄 State: {old_state.name} → {new_state.name}{reason_str}", "yellow", "ℹ️")
    
    def _log_response_sent(self, user_id: str, response_length: int, tools_used: List[str]) -> None:
        """Log final response delivery."""
        tools_str = f" | Tools: {', '.join(tools_used)}" if tools_used else " | No tools"
        self._log(f"📤 Response sent to {user_id}: {response_length} chars{tools_str}", "green", "✅")
    
    def _log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log errors with optional exception details."""
        exc_str = f": {str(exception)}" if exception else ""
        self._log(f"💥 ERROR: {message}{exc_str}", "red", "❌", "error")
    
    def _log_critical(self, message: str) -> None:
        """Log critical security/system events."""
        self._log(message, "red", "🚨", "critical")
    
    def _log_skill_execution(self, skill_name: str, inputs: Dict[str, Any], success: bool) -> None:
        """Log skill execution events."""
        if success:
            self._log(f"🎯 Skill '{skill_name}' executed successfully", "magenta", "📚")
        else:
            self._log(f"❌ Skill '{skill_name}' execution failed", "red", "📚", "error")
    
    def _ensure_lollms_client(self) -> Optional[LollmsClient]:
        """Lazy initialization of LoLLMS client with logging."""
        if not self._lollms_client_initialized:
            self._log("Initializing LoLLMS client...", "cyan", "🔗")
            try:
                self._lollms_client = build_lollms_client()
                self._log("✅ LoLLMS client connected", "green", "🔗")
            except Exception as e:
                self._log_error("Failed to initialize LoLLMS client", e)
                self._lollms_client = None
            self._lollms_client_initialized = True
        return self._lollms_client
    
    def _ensure_skills_initialized(self) -> bool:
        """Lazy initialization of skills subsystems with logging."""
        if not self._skills_enabled:
            return False
        
        if self._skill_registry is None:
            self._log("Initializing skills subsystems...", "magenta", "📚")
            try:
                from lollmsbot.skills import get_skill_registry, SkillExecutor, SkillLearner
                
                self._skill_registry = get_skill_registry()
                self._skill_executor = SkillExecutor(self, self._skill_registry, self._guardian)
                self._skill_learner = SkillLearner(self._skill_registry, self._skill_executor)
                
                skill_count = len(self._skill_registry._skills) if self._skill_registry else 0
                self._log(f"✅ Skills loaded: {skill_count} skills available", "green", "📚")
                return True
            except Exception as e:
                self._log_error("Failed to initialize skills", e)
                self._skills_enabled = False
                return False
        
        return True
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    @property
    def guardian(self) -> Optional[Guardian]:
        return self._guardian
    
    @property
    def skill_registry(self) -> Any:  # Optional[SkillRegistry]
        self._ensure_skills_initialized()
        return self._skill_registry
    
    def set_file_delivery_callback(self, callback: Callable[[str, List[Dict[str, Any]]], Awaitable[bool]]) -> None:
        self._file_delivery_callback = callback
    
    def set_tool_event_callback(self, callback: ToolEventCallback) -> None:
        self._tool_event_callback = callback
    
    def set_skill_event_callback(self, callback: Callable[[str, str, Dict[str, Any]], Awaitable[None]]) -> None:
        self._skill_event_callback = callback
    
    async def register_tool(self, tool: Tool) -> None:
        """Register a tool with the agent - with logging."""
        async with self._tool_lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' already registered")
            self._tools[tool.name] = tool
            self._log(f"🔧 Tool registered: {tool.name} (risk={tool.risk_level})", "purple", "➕")
    
    @property
    def tools(self) -> Dict[str, Tool]:
        return dict(self._tools)
    
    async def list_available_skills(
        self,
        user_id: str,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List skills available to this user."""
        if not self._ensure_skills_initialized():
            return []
        
        perms = self._user_permissions.get(user_id, self._default_permissions)
        if perms.level.value < PermissionLevel.SKILLS.value:
            return []
        
        if search_query:
            skills = self._skill_registry.search(search_query)
            skills = [s for s, _ in skills]
        elif category:
            skills = self._skill_registry.list_skills(category=category)
        else:
            skills = list(self._skill_registry._skills.values())
        
        if perms.allowed_skills is not None:
            skills = [s for s in skills if s.name in perms.allowed_skills]
        skills = [s for s in skills if s.name not in perms.denied_skills]
        
        return [
            {
                "name": s.name,
                "version": s.metadata.version,
                "description": s.metadata.description,
                "complexity": s.metadata.complexity.name,
                "tags": s.metadata.tags,
                "confidence": s.metadata.confidence_score,
                "can_execute": s.check_dependencies(set(self._tools.keys()), set(self._skill_registry._skills.keys()))[0],
            }
            for s in skills
        ]
    
    async def execute_skill(
        self,
        user_id: str,
        skill_name: str,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> SkillResult:
        """Execute a skill with full permission checking and logging."""
        if not self._ensure_skills_initialized():
            return SkillResult(
                success=False, result=None, skill_name=skill_name, skill_version="unknown",
                execution_id="", error="Skills not initialized",
            )
        
        perms = self._user_permissions.get(user_id, self._default_permissions)
        
        # Log permission check
        self._log(f"🔒 Checking permissions for skill '{skill_name}' (user: {user_id})", "yellow", "🛡️")
        
        if perms.level.value < PermissionLevel.SKILLS.value:
            self._log(f"🚫 Permission denied: {user_id} lacks SKILLS level", "red", "❌")
            return SkillResult(
                success=False, result=None, skill_name=skill_name, skill_version="unknown",
                execution_id="", error="Permission denied: skills not available",
            )
        
        if perms.denied_skills and skill_name in perms.denied_skills:
            self._log(f"🚫 Skill '{skill_name}' is in user's denylist", "red", "❌")
            return SkillResult(
                success=False, result=None, skill_name=skill_name, skill_version="unknown",
                execution_id="", error=f"Skill '{skill_name}' is blocked",
            )
        
        if perms.allowed_skills is not None and skill_name not in perms.allowed_skills:
            self._log(f"🚫 Skill '{skill_name}' not in user's allowlist", "red", "❌")
            return SkillResult(
                success=False, result=None, skill_name=skill_name, skill_version="unknown",
                execution_id="", error=f"Skill '{skill_name}' not in allowed list",
            )
        
        if self._skill_event_callback:
            await self._skill_event_callback("skill_start", skill_name, {"inputs": list(inputs.keys())})
        
        self._log(f"🎯 Executing skill '{skill_name}'...", "magenta", "📚")
        start = datetime.now()
        result = await self._skill_executor.execute(skill_name, inputs, context)
        duration = (datetime.now() - start).total_seconds()
        
        skill_result = SkillResult(
            success=result.get("success", False),
            result=result.get("result"),
            skill_name=skill_name,
            skill_version=result.get("skill_version", "unknown"),
            execution_id=result.get("execution_id", ""),
            steps_taken=result.get("steps_executed", []),
            tools_used=result.get("tools_used", []),
            skills_called=result.get("skills_called", []),
            duration_seconds=result.get("duration_seconds", duration),
            error=result.get("error"),
        )
        
        # Log skill completion
        self._log_skill_execution(skill_name, inputs, skill_result.success)
        if skill_result.success:
            self._log(f"   Duration: {skill_result.duration_seconds:.2f}s | Steps: {len(skill_result.steps_taken)}", "dim")
        else:
            self._log(f"   Error: {skill_result.error}", "red")
        
        if self._skill_event_callback:
            await self._skill_event_callback(
                "skill_complete" if skill_result.success else "skill_error",
                skill_name,
                {"duration": skill_result.duration_seconds, "steps": len(skill_result.steps_taken)},
            )
        
        return skill_result
    
    def _is_informational_query(self, message: str) -> bool:
        """Check if the message is asking about capabilities/tools (no tools needed)."""
        msg_lower = message.lower()
        info_patterns = [
            # Direct capability questions
            "what tools do you have",
            "what tools can you use",
            "what tools are available",
            "what can you do",
            "what are your capabilities",
            "list your tools",
            "show me your tools",
            "tell me about your tools",
            "what functions do you have",
            "what can you help me with",
            "how can you help",
            "what do you do",
            # Tool-specific questions
            "do you have a tool",
            "do you have tools",
            "can you use tools",
            "are there any tools",
        ]
        return any(pattern in msg_lower for pattern in info_patterns)
    
    async def chat(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process chat with detailed colored logging at every step."""
        # Log command reception
        self._log_command_received(user_id, message, context)
        
        # Check quarantine state
        if self._state == AgentState.QUARANTINED or (self._guardian and self._guardian.is_quarantined):
            self._log_critical("Request blocked: Agent is in QUARANTINED state")
            return {
                "success": False,
                "response": "Security Alert: System is in protective quarantine.",
                "error": "System quarantined by Guardian",
                "tools_used": [], "skills_used": [], "files_to_send": [],
            }
        
        # Security screening
        security_flags = []
        if self._guardian:
            self._log("🔍 Running security screening...", "yellow", "🛡️")
            is_safe, event = self._guardian.check_input(message, source=f"user:{user_id}")
            self._log_security_check(is_safe, event)
            
            if not is_safe:
                return {
                    "success": False,
                    "response": "Message blocked by security screening.",
                    "error": f"Security blocked: {event.description if event else 'policy violation'}",
                    "security_blocked": True, "tools_used": [], "skills_used": [], "files_to_send": [],
                }
            if event:
                security_flags.append(event.event_type)
        
        # Permission check
        if not await self.check_permission(user_id, PermissionLevel.BASIC):
            self._log(f"🚫 Permission denied for user {user_id}", "red", "❌")
            return {
                "success": False, "response": "", "error": "Access denied.",
                "tools_used": [], "skills_used": [], "files_to_send": [], "permission_denied": True,
            }
        
        # State management with logging
        old_state = self._state
        async with self._state_lock:
            if self._state == AgentState.PROCESSING:
                self._log("⚠️ Agent busy - rejecting concurrent request", "yellow", "⏳")
                return {
                    "success": False, "response": "Processing another request.", "error": "Agent busy",
                    "tools_used": [], "skills_used": [], "files_to_send": [],
                }
            self._state = AgentState.PROCESSING
            self._log_state_change(old_state, AgentState.PROCESSING, "processing new message")
        
        turn = ConversationTurn(user_message=message, input_screened=True, security_flags=security_flags)
        tools_used: List[str] = []
        skills_used: List[str] = []
        files_to_send: List[Dict[str, Any]] = []
        tool_events: List[Dict[str, Any]] = []
        
        try:
            # Analyze message type
            self._log("🔍 Analyzing message intent...", "cyan", "🤔")
            
            # Check if this is a simple informational query about tools/capabilities
            is_info_query = self._is_informational_query(message)
            if is_info_query:
                self._log("ℹ️ Detected informational query - bypassing tool logic", "cyan", "📝")
                # Generate simple response without LLM tool calls
                tool_list = "\n".join([f"- **{name}**: {tool.description[:80]}" for name, tool in self._tools.items()])
                response = f"I have access to the following tools:\n\n{tool_list}\n\nJust ask me to use any of these and I'll help you out!"
                turn.agent_response = response
                self._add_to_user_history(user_id, turn)
                self._log_response_sent(user_id, len(response), [])
                return {
                    "success": True,
                    "response": response,
                    "tools_used": [],
                    "skills_used": [],
                    "files_to_send": [],
                }
            
            # Check if this is a file generation request by analyzing the message
            file_generation_keywords = [
                "create", "make", "build", "generate", "write", "save",
                "file", "html", "game", "app", "script", "code",
                ".html", ".js", ".css", ".py", ".txt", ".json"
            ]
            is_file_request = any(kw in message.lower() for kw in file_generation_keywords)
            
            if is_file_request:
                self._log(f"📄 Detected file generation request (keywords found)", "green", "🎯")
                system_prompt = self._build_file_generation_prompt(context)
            else:
                system_prompt = self._build_system_prompt(context)
            
            # Get response from LLM with tool awareness
            client = self._ensure_lollms_client()
            
            if client:
                # Build conversation context
                history = self._get_user_history(user_id)
                prompt = self._format_prompt_for_lollms(system_prompt, history, message)
                
                self._log_llm_call(len(prompt), system_prompt)
                
                # Call LLM
                llm_response = client.generate_text(
                    prompt=prompt,
                    temperature=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1,
                )
                
                has_tools = "[[TOOL:" in llm_response or "<function_calls>" in llm_response
                self._log_llm_response(len(llm_response), has_tools)
                
                # Parse response for tool calls
                response, extracted_tools, extracted_files = await self._parse_and_execute_tools(
                    llm_response, user_id, context
                )
                
                tools_used.extend(extracted_tools)
                files_to_send.extend(extracted_files)
            else:
                # Fallback: try direct tool execution for simple requests
                self._log("⚠️ No LLM client - attempting direct tool execution", "yellow", "🔧")
                response = await self._try_direct_tool_execution(message, user_id, context)
                if response:
                    # Check if tools were used and get files
                    if hasattr(self, '_last_tool_result') and self._last_tool_result:
                        tools_used.append(self._last_tool_name or "unknown")
                        files_to_send.extend(self._last_tool_result.files_to_send or [])
                else:
                    response = f"I received: '{message}'. However, I'm not connected to a language model backend."
            
            # Trigger file delivery callback if files were generated
            if files_to_send and self._file_delivery_callback:
                self._log_file_generation(len(files_to_send), [f.get("filename", "unnamed") for f in files_to_send])
                self._log(f"📦 Delivering {len(files_to_send)} file(s) to user {user_id}", "green")
                delivery_success = await self._file_delivery_callback(user_id, files_to_send)
                if not delivery_success:
                    self._log("⚠️ File delivery callback reported failure", "yellow", "❌")
            
            # Build response with file information
            final_response = response
            if files_to_send and not any("file" in response.lower() for kw in ["created", "saved", "generated", "built"]):
                file_names = [f.get("filename", "unnamed") for f in files_to_send]
                final_response += f"\n\n📁 I've created: {', '.join(file_names)}"
            
            turn.agent_response = final_response
            turn.tools_used = tools_used
            turn.skills_used = skills_used
            self._add_to_user_history(user_id, turn)
            
            # Log successful completion
            self._log_response_sent(user_id, len(final_response), tools_used)
            
            return {
                "success": True,
                "response": final_response,
                "tools_used": tools_used,
                "skills_used": skills_used,
                "files_to_send": files_to_send,
                "tool_events": tool_events,
            }
            
        except Exception as exc:
            self._log_error(f"Error in chat processing", exc)
            async with self._state_lock:
                self._state = AgentState.ERROR
                self._log_state_change(AgentState.PROCESSING, AgentState.ERROR, f"exception: {str(exc)[:50]}")
            return {
                "success": False,
                "error": f"Processing error: {str(exc)}",
                "tools_used": tools_used, "skills_used": skills_used,
                "files_to_send": files_to_send,
            }
        
        finally:
            async with self._state_lock:
                if self._state != AgentState.ERROR:
                    old_state = self._state
                    self._state = AgentState.IDLE
                    self._log_state_change(old_state, AgentState.IDLE, "processing complete")
    
    def _build_file_generation_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Build a prompt optimized for file generation tasks."""
        # Get actual available tools with descriptions
        available_tools = []
        for name, tool in self._tools.items():
            available_tools.append(f"- {name}: {tool.description[:100]}")
        
        tool_list = "\n".join(available_tools) if available_tools else "No tools available"
        
        parts = [
            f"You are {self.name}, a helpful AI assistant that can create files and code.",
            "",
            "AVAILABLE TOOLS (USE THESE EXACT NAMES):",
            tool_list,
            "",
            "CRITICAL INSTRUCTION: When the user asks you to create, build, generate, or write a file,",
            "you MUST use the filesystem tool. Do NOT output the code directly in your response.",
            "",
            "TOOL CALL FORMAT (USE EXACTLY AS SHOWN):",
            "[[TOOL:filesystem|{\"operation\": \"create_html_app\", \"filename\": \"game.html\", \"html_content\": \"<!DOCTYPE html>...\"}]]",
            "",
            "EXAMPLE TOOL CALLS:",
            "1. For HTML games: [[TOOL:filesystem|{\"operation\": \"create_html_app\", \"filename\": \"mygame.html\", \"html_content\": \"<!DOCTYPE html>...\"}]]",
            "2. For text files: [[TOOL:filesystem|{\"operation\": \"write_file\", \"path\": \"output.txt\", \"content\": \"file content here\"}]]",
            "3. For listing files: [[TOOL:filesystem|{\"operation\": \"list_dir\", \"path\": \".\"}]]",
            "",
            "RULES:",
            "- ONLY use tool names from the AVAILABLE TOOLS list above",
            "- NEVER use placeholder names like 'toolname' - use the actual tool name",
            "- ALWAYS wrap tool calls in [[TOOL:...|{...}]] format",
            "- CREATE THE FILE using the tool, then briefly confirm in text",
        ]
        
        if context:
            channel = context.get("channel", "unknown")
            parts.append(f"\nCurrent channel: {channel}")
        
        return "\n".join(parts)
    
    async def _parse_and_execute_tools(
        self,
        llm_response: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
    ) -> tuple[str, List[str], List[Dict[str, Any]]]:
        """Parse LLM response and execute any embedded tool calls with detailed logging."""
        import re
        import json
        
        tools_used: List[str] = []
        files_generated: List[Dict[str, Any]] = []
        
        self._log("🔍 Parsing LLM response for tool calls...", "cyan", "🔧")
        
        # Check for native [[TOOL:...]] format first (most explicit)
        native_pattern = r'\[\[TOOL:(\w+)\|(\{.*?\})\]\]'
        native_matches = list(re.finditer(native_pattern, llm_response, re.DOTALL))
        
        # Check for XML-style tool calls (common with some LLMs)
        xml_pattern = r'<function_calls>\s*<invoke\s+name="([^"]+)">\s*(?:<parameter[^>]*>)?\s*<!\[CDATA\[(.*?)\]\]>\s*(?:</parameter>)?\s*</invoke>\s*</function_calls>'
        xml_matches = list(re.finditer(xml_pattern, llm_response, re.DOTALL | re.IGNORECASE))
        
        if not xml_matches:
            # Try alternative XML format without CDATA
            xml_pattern2 = r'<function_calls>\s*<invoke\s+name="([^"]+)">(.*?)</invoke>\s*</function_calls>'
            xml_matches = list(re.finditer(xml_pattern2, llm_response, re.DOTALL | re.IGNORECASE))
        
        # Also check for markdown code blocks that might contain tool calls
        code_block_pattern = r'```(?:xml|json)?\s*(<function_calls>.*?</function_calls>)```'
        code_matches = list(re.finditer(code_block_pattern, llm_response, re.DOTALL | re.IGNORECASE))
        
        all_matches = []
        
        # Process native matches first (they're most explicit)
        for match in native_matches:
            tool_name = match.group(1).lower()
            params_str = match.group(2)
            all_matches.append((match.start(), match.end(), 'native', tool_name, params_str))
        
        # Process XML matches
        for match in xml_matches:
            tool_name = match.group(1).lower()
            content = match.group(2)
            all_matches.append((match.start(), match.end(), 'xml', tool_name, content))
        
        # Process code block matches (extract inner content)
        for match in code_matches:
            inner = match.group(1)
            # Re-parse for actual tool calls inside code block
            inner_xml = re.search(r'<invoke\s+name="([^"]+)".*?>(.*?)</invoke>', inner, re.DOTALL | re.IGNORECASE)
            if inner_xml:
                tool_name = inner_xml.group(1).lower()
                content = inner_xml.group(2)
                all_matches.append((match.start(), match.end(), 'xml', tool_name, content))
        
        # Sort by position to maintain order
        all_matches.sort(key=lambda x: x[0])
        
        # Log detection results
        tools_found = list(set([m[3] for m in all_matches]))
        self._log_tool_detection(len(all_matches), tools_found)
        
        if not all_matches:
            # No tool calls found, return response as-is
            self._log("No executable tool calls found", "dim", "🔧")
            return llm_response.strip(), tools_used, files_generated
        
        # Execute each tool call
        response_parts = []
        last_end = 0
        
        for start, end, format_type, tool_name, content in all_matches:
            # Add text before this tool call
            response_parts.append(llm_response[last_end:start])
            
            # Parse parameters based on format
            tool_params = {}
            
            if format_type == 'xml':
                # Parse XML parameters
                cdata_match = re.search(r'<!\[CDATA\[(.*?)\]\]>', content, re.DOTALL)
                if cdata_match:
                    param_content = cdata_match.group(1)
                    
                    if "operation" not in content:
                        # Need to infer operation from context
                        if "<!DOCTYPE html>" in param_content or "<html" in param_content:
                            tool_params["operation"] = "create_html_app"
                            tool_params["filename"] = self._extract_filename_from_context(llm_response, "app.html")
                            tool_params["html_content"] = param_content
                        else:
                            tool_params["operation"] = "write_file"
                            tool_params["path"] = "output.txt"
                            tool_params["content"] = param_content
                    else:
                        # Extract all parameters from XML
                        param_pattern = r'<parameter name="([^"]+)">\s*(?:<!\[CDATA\[(.*?)\]\]>|\s*([^<]*))\s*</parameter>'
                        for pmatch in re.finditer(param_pattern, content, re.DOTALL):
                            p_name = pmatch.group(1)
                            p_value = pmatch.group(2) if pmatch.group(2) else pmatch.group(3)
                            tool_params[p_name] = p_value.strip() if p_value else ""
                else:
                    # No CDATA, parse regular XML parameters
                    param_pattern = r'<parameter name="([^"]+)">(.*?)</parameter>'
                    for pmatch in re.finditer(param_pattern, content, re.DOTALL):
                        p_name = pmatch.group(1)
                        p_value = pmatch.group(2).strip()
                        tool_params[p_name] = p_value
            
            elif format_type == 'native':
                # Parse JSON parameters
                try:
                    tool_params = json.loads(content)
                except json.JSONDecodeError as e:
                    self._log(f"❌ Failed to parse tool parameters for {tool_name}: {e}", "red", "🔧")
                    # Don't append error to user response - just skip silently
                    last_end = end
                    continue
            
            # CRITICAL FIX: Validate tool name against available tools
            if tool_name not in self._tools:
                self._log(f"⚠️ Tool '{tool_name}' not available", "yellow", "❌")
                
                # Special handling for common LLM hallucinations - don't expose errors to user
                if tool_name == "toolname" or tool_name == "tool":
                    # Silently skip this tool call - it's a placeholder, not a real request
                    self._log(f"   Skipping placeholder '{tool_name}' - not a real tool call", "yellow")
                    last_end = end
                    continue
                else:
                    # For genuinely unknown tools, still don't expose to user in final response
                    # but we logged it above for debugging
                    last_end = end
                    continue
            
            # Execute the tool
            tool = self._tools[tool_name]
            
            self._log(f"🔧 Executing tool: {tool_name}", "purple", "⚡")
            if self._tool_event_callback:
                await self._tool_event_callback("tool_start", tool_name, tool_params)
            
            try:
                result = await tool.execute(**tool_params)
                
                if self._tool_event_callback:
                    await self._tool_event_callback(
                        "tool_complete" if result.success else "tool_error",
                        tool_name,
                        {"success": result.success, "duration": result.execution_time}
                    )
                
                # Log execution result
                self._log_tool_execution(tool_name, tool_params, result.success, result.error)
                tools_used.append(tool_name)
                
                if result.success:
                    # Capture files from tool result
                    if result.files_to_send:
                        files_generated.extend(result.files_to_send)
                        file_names = [f.get("filename", "unnamed") for f in result.files_to_send]
                        self._log(f"   Generated files: {', '.join(file_names)}", "green", "📄")
                        # Don't add tool execution markers to user response
                    # Don't add "[Tool X executed]" to user response - keep it natural
                else:
                    # Tool failed but don't expose raw error to user
                    self._log(f"   Tool failed silently: {result.error}", "yellow")
                    
            except Exception as e:
                self._log(f"💥 Exception executing tool {tool_name}: {e}", "red", "❌")
                # Don't expose exception details to user
            
            last_end = end
        
        # Add remaining text
        response_parts.append(llm_response[last_end:])
        
        # Clean up the response - remove all tool call syntax and error markers
        final_response = "".join(response_parts).strip()
        
        # Remove all tool call formats from user-facing response
        final_response = re.sub(r'\[\[TOOL:[^\]]+\]\]', '', final_response)
        final_response = re.sub(r'<function_calls>.*?</function_calls>', '', final_response, flags=re.DOTALL | re.IGNORECASE)
        final_response = re.sub(r'\[Error: [^\]]+\]', '', final_response)
        final_response = re.sub(r'\[Tool [^\]]+\]', '', final_response)
        
        # Clean up empty lines and whitespace
        final_response = '\n'.join(line for line in final_response.split('\n') if line.strip())
        final_response = final_response.strip()
        
        # Summary
        self._log(f"✅ Tool processing complete: {len(tools_used)} tool(s) executed, {len(files_generated)} file(s) generated", "green", "🔧")
        
        return final_response, tools_used, files_generated
    
    def _extract_filename_from_context(self, full_response: str, default_name: str) -> str:
        """Try to extract a filename from the response context."""
        import re
        
        patterns = [
            r'"filename"\s*:\s*"([^"]+)"',
            r'filename=["\']?([^"\']+)["\']?',
            r'(\w+\.html?)',
            r'named?\s+["\']?([^"\']+\.[^"\']+)["\']?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, full_response, re.IGNORECASE)
            if match:
                filename = match.group(1)
                if '.' in filename:
                    return filename
        
        return default_name
    
    async def _try_direct_tool_execution(
        self,
        message: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Try to directly execute tools based on message patterns without LLM - with logging."""
        message_lower = message.lower()
        
        self._log("🔍 Attempting direct tool execution pattern matching...", "cyan", "🎯")
        
        # Check for HTML/game creation requests
        html_keywords = ["html", "game", "app", "page", "website"]
        is_html_request = any(kw in message_lower for kw in html_keywords)
        
        # Check for file creation requests
        file_keywords = ["create file", "make file", "write file", "save file", "generate file"]
        is_file_request = any(kw in message_lower for kw in file_keywords)
        
        if is_html_request and "filesystem" in self._tools:
            self._log(f"🎮 Detected HTML/game request: '{message[:50]}...'", "cyan", "🎯")
            
            # Try to extract what kind of game/app from message
            game_type = "game"
            if "snake" in message_lower:
                game_type = "snake game"
            elif "pong" in message_lower:
                game_type = "pong game"
            elif "tetris" in message_lower:
                game_type = "tetris game"
            elif "calculator" in message_lower:
                game_type = "calculator"
            elif "todo" in message_lower:
                game_type = "todo app"
            elif "star" in message_lower:
                game_type = "catch_the_stars"
            
            self._log(f"🎯 Selected game type: {game_type}", "cyan", "🎮")
            
            # Create a simple HTML5 game template
            filename = f"{game_type.replace(' ', '_')}.html"
            
            html_content = self._generate_html_game(game_type)
            
            # Use filesystem tool
            self._log(f"🔧 Calling filesystem tool to create {filename}", "purple", "⚡")
            tool = self._tools["filesystem"]
            result = await tool.execute(
                operation="create_html_app",
                filename=filename,
                html_content=html_content,
            )
            
            # Store result for retrieval
            self._last_tool_result = result
            self._last_tool_name = "filesystem"
            
            if result.success:
                files = result.files_to_send or []
                file_names = [f.get("filename", "unnamed") for f in files]
                self._log(f"✅ Direct execution successful: created {', '.join(file_names)}", "green", "✅")
                return f"I've created a {game_type} for you! The file is ready for download: {', '.join(file_names)}"
            else:
                self._log(f"❌ Direct execution failed: {result.error}", "red", "❌")
                return f"I tried to create the {game_type} but encountered an error: {result.error}"
        
        elif is_file_request and "filesystem" in self._tools:
            self._log("📄 File request detected but too complex for direct execution", "yellow", "⚠️")
            return None  # Let LLM handle complex cases
        
        self._log("No direct execution pattern matched", "dim", "❌")
        return None  # No direct execution pattern matched
    
    def _generate_html_game(self, game_type: str) -> str:
        """Generate HTML5 game content based on type."""
        
        if "snake" in game_type.lower():
            return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Game</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: #1a1a2e;
            font-family: Arial, sans-serif;
        }
        canvas {
            border: 2px solid #0f3460;
            background: #16213e;
        }
        .info {
            color: #fff;
            text-align: center;
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div>
        <div class="info">
            <h1>🐍 Snake Game</h1>
            <p>Use arrow keys to play</p>
        </div>
        <canvas id="game" width="400" height="400"></canvas>
    </div>
    <script>
        const canvas = document.getElementById('game');
        const ctx = canvas.getContext('2d');
        const grid = 20;
        let snake = [{x: 10, y: 10}];
        let food = {x: 15, y: 15};
        let dx = 1, dy = 0;
        let score = 0;

        function draw() {
            ctx.fillStyle = '#16213e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#0f0';
            snake.forEach(s => ctx.fillRect(s.x*grid, s.y*grid, grid-2, grid-2));
            
            ctx.fillStyle = '#f00';
            ctx.fillRect(food.x*grid, food.y*grid, grid-2, grid-2);
            
            ctx.fillStyle = '#fff';
            ctx.font = '20px Arial';
            ctx.fillText('Score: ' + score, 10, 30);
        }

        function update() {
            const head = {x: snake[0].x + dx, y: snake[0].y + dy};
            
            if (head.x < 0) head.x = 19;
            if (head.x > 19) head.x = 0;
            if (head.y < 0) head.y = 19;
            if (head.y > 19) head.y = 0;
            
            if (snake.some(s => s.x === head.x && s.y === head.y)) {
                snake = [{x: 10, y: 10}];
                score = 0;
                return;
            }
            
            snake.unshift(head);
            
            if (head.x === food.x && head.y === food.y) {
                score += 10;
                food = {x: Math.floor(Math.random()*20), y: Math.floor(Math.random()*20)};
            } else {
                snake.pop();
            }
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp' && dy === 0) { dx = 0; dy = -1; }
            if (e.key === 'ArrowDown' && dy === 0) { dx = 0; dy = 1; }
            if (e.key === 'ArrowLeft' && dx === 0) { dx = -1; dy = 0; }
            if (e.key === 'ArrowRight' && dx === 0) { dx = 1; dy = 0; }
        });

        setInterval(() => { update(); draw(); }, 100);
        draw();
    </script>
</body>
</html>'''
        
        elif "pong" in game_type.lower():
            return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pong Game</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: #1a1a2e;
        }
        canvas {
            border: 2px solid #0f3460;
            background: #16213e;
        }
    </style>
</head>
<body>
    <canvas id="game" width="600" height="400"></canvas>
    <script>
        const canvas = document.getElementById('game');
        const ctx = canvas.getContext('2d');
        let ball = {x: 300, y: 200, dx: 4, dy: 4, radius: 10};
        let paddle1 = {x: 10, y: 150, width: 10, height: 100};
        let paddle2 = {x: 580, y: 150, width: 10, height: 100};
        let score1 = 0, score2 = 0;

        function draw() {
            ctx.fillStyle = '#16213e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.beginPath();
            ctx.arc(ball.x, ball.y, ball.radius, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
            
            ctx.fillStyle = '#0f0';
            ctx.fillRect(paddle1.x, paddle1.y, paddle1.width, paddle1.height);
            ctx.fillStyle = '#f00';
            ctx.fillRect(paddle2.x, paddle2.y, paddle2.width, paddle2.height);
            
            ctx.fillStyle = '#fff';
            ctx.font = '30px Arial';
            ctx.fillText(score1 + ' - ' + score2, 270, 40);
        }

        function update() {
            ball.x += ball.dx;
            ball.y += ball.dy;
            
            if (ball.y < ball.radius || ball.y > canvas.height - ball.radius) ball.dy *= -1;
            
            if (paddle2.y + paddle2.height/2 < ball.y) paddle2.y += 3;
            if (paddle2.y + paddle2.height/2 > ball.y) paddle2.y -= 3;
            
            if (ball.x - ball.radius < paddle1.x + paddle1.width && 
                ball.y > paddle1.y && ball.y < paddle1.y + paddle1.height) ball.dx = Math.abs(ball.dx);
            if (ball.x + ball.radius > paddle2.x && 
                ball.y > paddle2.y && ball.y < paddle2.y + paddle2.height) ball.dx = -Math.abs(ball.dx);
            
            if (ball.x < 0) { score2++; ball = {x: 300, y: 200, dx: 4, dy: 4, radius: 10}; }
            if (ball.x > canvas.width) { score1++; ball = {x: 300, y: 200, dx: -4, dy: 4, radius: 10}; }
        }

        document.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            paddle1.y = e.clientY - rect.top - paddle1.height/2;
        });

        setInterval(() => { update(); draw(); }, 16);
        draw();
    </script>
</body>
</html>'''
        
        elif "star" in game_type.lower():
            # Catch the falling stars game
            return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Catch the Falling Stars</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
        }
        #gameContainer {
            position: relative;
            width: 800px;
            height: 600px;
            max-width: 95vw;
            max-height: 95vh;
        }
        #gameCanvas {
            border-radius: 15px;
            box-shadow: 0 0 30px rgba(233, 69, 96, 0.3);
            background: linear-gradient(180deg, #0a0a1a 0%, #1a0a2e 100%);
        }
        #ui {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            pointer-events: none;
        }
        .ui-element {
            background: rgba(0, 0, 0, 0.6);
            padding: 10px 20px;
            border-radius: 25px;
            color: #fff;
            font-size: 18px;
            font-weight: bold;
        }
        #score { color: #ffd700; }
        #lives { color: #e94560; }
        #level { color: #00d9ff; }
        #startScreen, #gameOverScreen {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background: rgba(0, 0, 0, 0.85);
            border-radius: 15px;
            z-index: 10;
        }
        #gameOverScreen { display: none; }
        h1 {
            color: #ffd700;
            font-size: 48px;
            margin-bottom: 10px;
            text-shadow: 0 0 20px #ffd700;
        }
        p {
            color: #aaa;
            font-size: 18px;
            margin-bottom: 30px;
            text-align: center;
        }
        .btn {
            padding: 15px 40px;
            font-size: 20px;
            font-weight: bold;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            text-transform: uppercase;
        }
        #startBtn {
            background: linear-gradient(45deg, #e94560, #ff6b6b);
            color: white;
        }
        #restartBtn {
            background: linear-gradient(45deg, #00d9ff, #00ff88);
            color: #1a1a2e;
        }
    </style>
</head>
<body>
    <div id="gameContainer">
        <canvas id="gameCanvas" width="800" height="600"></canvas>
        <div id="ui">
            <div class="ui-element" id="score">⭐ Score: 0</div>
            <div class="ui-element" id="level">📈 Level: 1</div>
            <div class="ui-element" id="lives">❤️ Lives: 3</div>
        </div>
        <div id="startScreen">
            <h1>⭐ Catch the Stars ⭐</h1>
            <p>Move your mouse to control the basket.<br>Catch golden stars for points (+10)<br>Avoid red bombs or lose a life!</p>
            <button class="btn" id="startBtn">Start Game</button>
        </div>
        <div id="gameOverScreen">
            <h1>Game Over!</h1>
            <div id="finalScore">Score: 0</div>
            <button class="btn" id="restartBtn">Play Again</button>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        const basket = { x: 350, y: 520, width: 100, height: 60 };
        let mouseX = 400;
        let score = 0, lives = 3, level = 1, gameRunning = false;
        let objects = [];
        let particles = [];
        
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
        });
        
        function drawBasket() {
            ctx.fillStyle = '#e94560';
            ctx.fillRect(basket.x, basket.y, basket.width, basket.height);
        }
        
        function update() {
            if (!gameRunning) return;
            basket.x = mouseX - basket.width / 2;
            basket.x = Math.max(0, Math.min(canvas.width - basket.width, basket.x));
            
            // Spawn objects
            if (Math.random() < 0.02 + level * 0.005) {
                objects.push({
                    x: Math.random() * (canvas.width - 30) + 15,
                    y: -30,
                    size: 20,
                    speed: 2 + level * 0.5,
                    type: Math.random() < 0.2 ? 'bomb' : 'star'
                });
            }
            
            // Update objects
            objects.forEach(obj => {
                obj.y += obj.speed;
                
                // Collision with basket
                if (obj.y + obj.size > basket.y && obj.x > basket.x && obj.x < basket.x + basket.width) {
                    if (obj.type === 'star') score += 10;
                    else lives--;
                    obj.collected = true;
                }
                
                // Missed stars
                if (obj.y > canvas.height && obj.type === 'star') lives--;
            });
            
            objects = objects.filter(obj => !obj.collected && obj.y < canvas.height + 50);
            
            if (lives <= 0) {
                gameRunning = false;
                document.getElementById('gameOverScreen').style.display = 'flex';
                document.getElementById('finalScore').textContent = 'Score: ' + score;
            }
            
            document.getElementById('score').textContent = '⭐ Score: ' + score;
            document.getElementById('lives').textContent = '❤️ Lives: ' + lives;
            document.getElementById('level').textContent = '📈 Level: ' + level;
        }
        
        function draw() {
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            objects.forEach(obj => {
                ctx.beginPath();
                ctx.arc(obj.x, obj.y, obj.size, 0, Math.PI * 2);
                ctx.fillStyle = obj.type === 'star' ? '#ffd700' : '#ff3333';
                ctx.fill();
            });
            
            drawBasket();
        }
        
        function loop() {
            update();
            draw();
            requestAnimationFrame(loop);
        }
        
        document.getElementById('startBtn').addEventListener('click', () => {
            gameRunning = true;
            document.getElementById('startScreen').style.display = 'none';
        });
        
        document.getElementById('restartBtn').addEventListener('click', () => {
            score = 0; lives = 3; level = 1; objects = [];
            gameRunning = true;
            document.getElementById('gameOverScreen').style.display = 'none';
        });
        
        loop();
    </script>
</body>
</html>'''
        
        else:
            # Default simple interactive page
            return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My App</title>
    <style>
        body {
            font-family: system-ui, sans-serif;
            text-align: center;
            padding: 50px;
            background: linear-gradient(135deg, #1a1a2e, #0f3460);
            color: white;
            min-height: 100vh;
        }
        button {
            padding: 15px 30px;
            font-size: 18px;
            background: #e94560;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        button:hover {
            background: #ff6b6b;
        }
    </style>
</head>
<body>
    <h1>Welcome!</h1>
    <p>This is your custom HTML app.</p>
    <button onclick="alert('Hello!')">Click Me!</button>
</body>
</html>'''
    
    def _build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the system prompt for the AI with explicit tool documentation."""
        # Get actual available tools with detailed info
        available_tools = []
        for name, tool in self._tools.items():
            desc = tool.description[:80] if len(tool.description) > 80 else tool.description
            available_tools.append(f"- {name}: {desc}")
        
        tool_list = "\n".join(available_tools) if available_tools else "No tools currently available."
        
        parts = [
            f"You are {self.name}, a helpful AI assistant with access to tools.",
            "",
            "AVAILABLE TOOLS (USE THESE EXACT NAMES):",
            tool_list,
            "",
            "TOOL USAGE INSTRUCTIONS:",
            "When you need to use a tool, output EXACTLY this format:",
            "[[TOOL:toolname|{\"param1\": \"value1\", \"param2\": \"value2\"}]]",
            "",
            "CRITICAL RULES:",
            "1. ONLY use tool names from the AVAILABLE TOOLS list above",
            "2. NEVER use placeholder names like 'toolname' - use the actual tool name (e.g., 'filesystem', 'http', 'calendar', 'shell')",
            "3. ALWAYS wrap tool calls in [[TOOL:...|{...}]] format",
            "4. For the 'filesystem' tool, common operations are: read_file, write_file, list_dir, create_html_app",
            "5. For the 'http' tool, specify method and url: {\"method\": \"get\", \"url\": \"https://example.com\"}",
            "6. For the 'calendar' tool, operations include: get_events, add_event, delete_event",
            "7. For the 'shell' tool, use {\"operation\": \"execute\", \"command\": \"your command\"}",
            "",
            "EXAMPLES OF CORRECT TOOL CALLS:",
            "[[TOOL:filesystem|{\"operation\": \"create_html_app\", \"filename\": \"game.html\", \"html_content\": \"<!DOCTYPE html>...\"}]]",
            "[[TOOL:http|{\"method\": \"get\", \"url\": \"https://api.example.com/data\"}]]",
            "[[TOOL:calendar|{\"operation\": \"add_event\", \"title\": \"Meeting\", \"start\": \"2024-01-01T10:00\", \"end\": \"2024-01-01T11:00\"}]]",
            "",
            "When NOT to use tools:",
            "- For simple conversational responses",
            "- When answering questions about your capabilities (just list them)",
            "- For greetings, small talk, or explanations",
            "",
            "Guidelines:",
            "- Be helpful, accurate, and concise",
            "- If you're unsure about something, say so",
            "- Use tools when they would DIRECTLY help fulfill the user's request",
            "- Always use the exact tool names from the available list",
        ]
        
        # Add skills info
        if self._skill_registry:
            skill_count = len(self._skill_registry._skills)
            parts.append(f"\nYou also have access to {skill_count} specialized skills for complex tasks.")
            parts.append("Skills can be invoked through natural language requests.")
        
        # Add channel context if available
        if context:
            channel = context.get("channel", "unknown")
            parts.append(f"\nCurrent channel: {channel}")
        
        return "\n".join(parts)
    
    def _format_prompt_for_lollms(
        self,
        system_prompt: str,
        history: List[ConversationTurn],
        current_message: str,
    ) -> str:
        """Format the complete prompt for LoLLMS generation."""
        parts = [f"### System:\n{system_prompt}\n"]
        
        # Add recent history (last 5 turns to stay within context)
        for turn in history[-5:]:
            parts.append(f"### User:\n{turn.user_message}\n")
            parts.append(f"### Assistant:\n{turn.agent_response}\n")
        
        # Add current message
        parts.append(f"### User:\n{current_message}\n")
        parts.append("### Assistant:\n")
        
        return "\n".join(parts)
    
    def _format_skill_response(self, skills_used: List[str], skill_plan: Dict[str, Any]) -> str:
        """Format a response when skills were successfully executed."""
        if len(skills_used) == 1:
            return f"I used the '{skills_used[0]}' skill to help with your request. The task has been completed successfully."
        else:
            return f"I used multiple skills ({', '.join(skills_used)}) to complete your request. Everything worked as expected!"
    
    async def _plan_with_skills(
        self,
        message: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Plan execution using skills as primary building blocks."""
        if not self._ensure_skills_initialized():
            return {"use_skills": False, "skills": []}
        
        available_tools = set(self._tools.keys())
        recommended = self._skill_registry.recommend(message, available_tools)
        
        perms = self._user_permissions.get(user_id, self._default_permissions)
        allowed = []
        for skill, reason in recommended:
            if perms.level.value >= PermissionLevel.SKILLS.value:
                if perms.denied_skills and skill.name in perms.denied_skills:
                    continue
                if perms.allowed_skills is not None and skill.name not in perms.allowed_skills:
                    continue
                allowed.append((skill, reason))
        
        if not allowed:
            return {"use_skills": False, "skills": []}
        
        # Simplified heuristic
        from lollmsbot.skills import SkillComplexity
        use_skills = len(allowed) > 0 and any(
            s.metadata.complexity.value >= SkillComplexity.MODERATE.value 
            for s, _ in allowed[:2]
        )
        
        return {
            "use_skills": use_skills,
            "skills": [{"skill": s.name, "inputs": {}} for s, _ in allowed[:2]],
            "candidates": [{"name": s.name, "relevance_reason": r} for s, r in allowed[:5]],
        }
    
    async def check_permission(self, user_id: str, required: PermissionLevel) -> bool:
        """Check if user has required permission level."""
        perms = self._user_permissions.get(user_id, self._default_permissions)
        return perms.level.value >= required.value
    
    async def can_use_tool(self, user_id: str, tool_name: str) -> bool:
        """Check if user can use a specific tool."""
        perms = self._user_permissions.get(user_id, self._default_permissions)
        
        if perms.level.value < PermissionLevel.TOOLS.value:
            return False
        
        if tool_name in perms.denied_tools:
            return False
        
        if perms.allowed_tools is not None:
            return tool_name in perms.allowed_tools
        
        return True
    
    def _get_user_history(self, user_id: str) -> List[ConversationTurn]:
        """Get conversation history for user."""
        return self._user_histories.get(user_id, [])
    
    def _add_to_user_history(self, user_id: str, turn: ConversationTurn) -> None:
        """Add turn to user history."""
        if user_id not in self._user_histories:
            self._user_histories[user_id] = []
        self._user_histories[user_id].append(turn)
        
        max_history = getattr(self.config, 'max_history', 10)
        while len(self._user_histories[user_id]) > max_history * 2:
            self._user_histories[user_id].pop(0)
    
    def __repr__(self) -> str:
        guardian_status = "🛡️" if self._guardian and not self._guardian.is_quarantined else "🚨" if self._guardian else "⚪"
        skill_count = len(self._skill_registry._skills) if self._skill_registry else 0
        skill_status = f"📚{skill_count}" if self._skills_enabled else "📭"
        lollms_status = "🔗" if self._lollms_client else "❌"
        return f"Agent({self.agent_id}, {self.name}, {guardian_status}, {skill_status}, {lollms_status}, {len(self._tools)} tools)"
