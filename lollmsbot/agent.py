"""
Agent module for LollmsBot.

This module provides the core Agent class and Tool framework for building
AI agents with state management, tool registration, and conversation memory.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type

from lollmsbot.config import BotConfig
from lollmsbot.lollms_client import LollmsClient


class AgentState(Enum):
    """Enumeration of possible agent states."""
    IDLE = auto()
    PROCESSING = auto()
    ERROR = auto()


class ToolError(Exception):
    """Exception raised when a tool execution fails."""
    pass


class AgentError(Exception):
    """Exception raised when agent processing fails."""
    pass


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ConversationTurn:
    """Single turn in conversation history."""
    timestamp: datetime = field(default_factory=datetime.now)
    user_message: str = ""
    agent_response: str = ""
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for agent tools.
    
    All tools must inherit from this class and implement the execute method.
    Tools are registered with the agent and can be called during processing.
    
    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema describing expected parameters.
    """
    
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure subclasses define required class attributes."""
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise ValueError(f"Tool {cls.__name__} must define a 'name' attribute")
        if not cls.description:
            raise ValueError(f"Tool {cls.__name__} must define a 'description' attribute")
    
    @abstractmethod
    async def execute(self, **params: Any) -> ToolResult:
        """Execute the tool with given parameters.
        
        Args:
            **params: Parameters validated against self.parameters schema.
            
        Returns:
            ToolResult containing execution status and output.
        """
        pass
    
    def __repr__(self) -> str:
        return f"Tool({self.name})"


class Agent:
    """Core AI agent with state management and tool execution.
    
    The Agent class provides a framework for building conversational AI agents
    that can use tools, maintain state, and interact with the LoLLMs backend.
    
    Attributes:
        agent_id: Unique identifier for this agent instance.
        name: Display name of the agent.
        state: Current operational state (idle, processing, error).
        tools: Dictionary of registered tools by name.
        memory: Conversation history and working memory.
        lollms_client: Client for LoLLMs API communication.
        
    Example:
        >>> agent = Agent(lollms_client=client)
        >>> agent.register_tool(MyTool())
        >>> response = await agent.run("Hello, what tools do you have?")
    """
    
    def __init__(
        self,
        lollms_client: LollmsClient,
        config: Optional[BotConfig] = None,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize the Agent.
        
        Args:
            lollms_client: Configured LollmsClient for LLM communication.
            config: Optional BotConfig for behavior settings.
            agent_id: Optional unique ID (generated if not provided).
            name: Optional display name (from config if not provided).
        """
        self.config: BotConfig = config or BotConfig()
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.name: str = name or self.config.name
        
        self._state: AgentState = AgentState.IDLE
        self._tools: Dict[str, Tool] = {}
        self._memory: Dict[str, Any] = {
            "conversation_history": [] as List[ConversationTurn],
            "working_memory": {} as Dict[str, Any],
            "context": {} as Dict[str, Any],
        }
        
        self.lollms_client: LollmsClient = lollms_client
        
        self._state_lock: asyncio.Lock = asyncio.Lock()
        self._tool_lock: asyncio.Lock = asyncio.Lock()
    
    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state
    
    @property
    def tools(self) -> Dict[str, Tool]:
        """Get dictionary of registered tools."""
        return self._tools.copy()
    
    @property
    def memory(self) -> Dict[str, Any]:
        """Get agent memory (read-only copy)."""
        return {
            "conversation_history": self._memory["conversation_history"].copy(),
            "working_memory": self._memory["working_memory"].copy(),
            "context": self._memory["context"].copy(),
        }
    
    async def register_tool(self, tool: Tool) -> None:
        """Register a tool for agent use.
        
        Args:
            tool: Tool instance to register.
            
        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        async with self._tool_lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' is already registered")
            self._tools[tool.name] = tool
    
    async def unregister_tool(self, tool_name: str) -> Optional[Tool]:
        """Remove a registered tool.
        
        Args:
            tool_name: Name of the tool to remove.
            
        Returns:
            The removed tool if found, None otherwise.
        """
        async with self._tool_lock:
            return self._tools.pop(tool_name, None)
    
    async def run(self, message: str) -> str:
        """Process a user message and generate a response.
        
        This is the main entry point for agent interaction. It handles
        planning, tool execution, and response generation.
        
        Args:
            message: User input message to process.
            
        Returns:
            Generated response string.
            
        Raises:
            AgentError: If processing fails in an unrecoverable way.
        """
        async with self._state_lock:
            if self._state == AgentState.PROCESSING:
                return "I'm currently processing another request. Please wait."
            self._state = AgentState.PROCESSING
        
        turn = ConversationTurn(user_message=message)
        tools_used: List[str] = []
        
        try:
            # Planning phase: determine what tools are needed
            tools_needed = await self._plan(message)
            
            # Execution phase: run required tools
            tool_results: List[ToolResult] = []
            for tool_name, params in tools_needed:
                result = await self._execute(tool_name, params)
                tool_results.append(result)
                if result.success:
                    tools_used.append(tool_name)
            
            # Reflection phase: process results and build context
            reflection = await self._reflect(tool_results)
            
            # Generate final response using LLM
            context = self._build_context(message, tools_needed, tool_results, reflection)
            response = await self._generate_response(context)
            
            # Update memory
            turn.agent_response = response
            turn.tools_used = tools_used
            turn.metadata = {
                "tools_needed": tools_needed,
                "reflection": reflection,
                "state": self._state.name,
            }
            
            await self._add_to_history(turn)
            
            return response
            
        except Exception as exc:
            async with self._state_lock:
                self._state = AgentState.ERROR
            raise AgentError(f"Agent processing failed: {exc}") from exc
        
        finally:
            async with self._state_lock:
                if self._state != AgentState.ERROR:
                    self._state = AgentState.IDLE
    
    async def _plan(self, message: str) -> List[tuple[str, Dict[str, Any]]]:
        """Plan which tools are needed to process the message.
        
        Uses the LLM to analyze the message and determine required tools
        and their parameters.
        
        Args:
            message: User message to analyze.
            
        Returns:
            List of (tool_name, parameters) tuples.
        """
        if not self._tools:
            return []
        
        # Build tool descriptions for the LLM
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self._tools.items()
        ])
        
        planning_prompt = f"""You are a planning assistant. Analyze the user message and determine which tools are needed.

Available tools:
{tool_descriptions}

User message: {message}

Respond in this exact format:
TOOLS_NEEDED: <comma-separated list of tool names, or NONE if no tools needed>
PARAMETERS: <JSON object mapping tool names to their parameters>

If no tools are needed, respond with:
TOOLS_NEEDED: NONE
PARAMETERS: {{}}
"""
        
        try:
            plan_response = await self.lollms_client.generate(
                planning_prompt,
                n_predict=200,
                temperature=0.3,
            )
            
            tools_needed: List[tuple[str, Dict[str, Any]]] = []
            
            # Parse the planning response
            for line in plan_response.split("\n"):
                if line.startswith("TOOLS_NEEDED:"):
                    tools_line = line.replace("TOOLS_NEEDED:", "").strip()
                    if tools_line == "NONE" or not tools_line:
                        return []
                    tool_names = [t.strip() for t in tools_line.split(",")]
                    
                elif line.startswith("PARAMETERS:"):
                    params_text = line.replace("PARAMETERS:", "").strip()
                    import json
                    try:
                        params = json.loads(params_text)
                    except json.JSONDecodeError:
                        params = {}
                    
                    # Build result list
                    for tool_name in tool_names:
                        if tool_name in self._tools:
                            tool_params = params.get(tool_name, {})
                            tools_needed.append((tool_name, tool_params))
            
            return tools_needed
            
        except Exception:
            # If planning fails, return empty list (fallback to direct response)
            return []
    
    async def _execute(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a specific tool with given parameters.
        
        Args:
            tool_name: Name of the registered tool to execute.
            params: Parameters to pass to the tool.
            
        Returns:
            ToolResult from execution.
            
        Raises:
            ToolError: If tool is not found or execution fails.
        """
        async with self._tool_lock:
            tool = self._tools.get(tool_name)
        
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found",
            )
        
        import time
        start_time = time.time()
        
        try:
            result = await tool.execute(**params)
            return result
            
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=str(exc),
                execution_time=time.time() - start_time,
            )
    
    async def _reflect(self, results: List[ToolResult]) -> str:
        """Reflect on tool execution results to inform response generation.
        
        Args:
            results: List of results from executed tools.
            
        Returns:
            Reflection summary string.
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        reflection_parts: List[str] = []
        
        if successful:
            reflection_parts.append(f"Successfully executed {len(successful)} tool(s).")
        
        if failed:
            reflection_parts.append(f"Failed to execute {len(failed)} tool(s):")
            for result in failed:
                reflection_parts.append(f"  - Error: {result.error}")
        
        if not results:
            reflection_parts.append("No tools were executed for this request.")
        
        return " ".join(reflection_parts)
    
    def _build_context(
        self,
        message: str,
        tools_needed: List[tuple[str, Dict[str, Any]]],
        tool_results: List[ToolResult],
        reflection: str,
    ) -> str:
        """Build context for LLM response generation.
        
        Args:
            message: Original user message.
            tools_needed: Tools that were planned.
            tool_results: Results from tool executions.
            reflection: Reflection summary.
            
        Returns:
            Formatted context string.
        """
        # Add conversation history context
        history_context = ""
        recent_history = self._memory["conversation_history"][-3:]
        if recent_history:
            history_parts = []
            for turn in recent_history:
                history_parts.append(f"User: {turn.user_message}")
                history_parts.append(f"Assistant: {turn.agent_response}")
            history_context = "\n".join(history_parts)
        
        # Build tool results context
        results_context = ""
        for (tool_name, _), result in zip(tools_needed, tool_results):
            status = "success" if result.success else "failure"
            results_context += f"\n- {tool_name}: {status}"
            if result.success:
                results_context += f", output: {result.output}"
            else:
                results_context += f", error: {result.error}"
        
        prompt = f"""You are {self.name}, an AI assistant.

Conversation history:
{history_context}

Current user message: {message}

Tool execution results:{results_context or " No tools executed."}

Reflection: {reflection}

Provide a helpful, natural response to the user based on the available information."""
        
        return prompt
    
    async def _generate_response(self, context: str) -> str:
        """Generate final response using the LLM.
        
        Args:
            context: Full context prompt for generation.
            
        Returns:
            Generated response string.
        """
        response = await self.lollms_client.generate(
            context,
            n_predict=self.config.max_history * 50,
            temperature=0.7,
        )
        return response.strip()
    
    async def _add_to_history(self, turn: ConversationTurn) -> None:
        """Add a conversation turn to history, maintaining size limit.
        
        Args:
            turn: ConversationTurn to add.
        """
        self._memory["conversation_history"].append(turn)
        
        # Trim to max history size
        max_history = self.config.max_history
        while len(self._memory["conversation_history"]) > max_history:
            self._memory["conversation_history"].pop(0)
    
    async def clear_memory(self) -> None:
        """Clear all conversation history and working memory."""
        self._memory["conversation_history"].clear()
        self._memory["working_memory"].clear()
    
    async def set_context(self, key: str, value: Any) -> None:
        """Set a value in the persistent context memory.
        
        Args:
            key: Context key.
            value: Value to store.
        """
        self._memory["context"][key] = value
    
    async def get_context(self, key: str) -> Any:
        """Get a value from persistent context memory.
        
        Args:
            key: Context key.
            
        Returns:
            Stored value or None if not found.
        """
        return self._memory["context"].get(key)
    
    def __repr__(self) -> str:
        return f"Agent({self.agent_id}, {self.name}, {self._state.name})"