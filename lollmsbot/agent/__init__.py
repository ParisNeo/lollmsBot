"""
Agent module for LollmsBot - RLM Memory Edition

Updated to use RLM-compliant memory manager with double-memory structure:
- External Memory Store (EMS): Compressed, chunked long-term storage
- REPL Context Buffer (RCB): Working memory with loadable handles
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import field
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable

from lollmsbot.config import BotConfig
from lollmsbot.lollms_client import LollmsClient, build_lollms_client
from lollmsbot.guardian import get_guardian, Guardian


# Import from modular structure
from lollmsbot.agent.config import (
    AgentState,
    PermissionLevel,
    Tool,
    ToolResult,
    SkillResult,
    UserPermissions,
    ConversationTurn,
    ToolEventCallback,
    ToolError,
    AgentError,
)

# Import RLM Memory package (new modular structure)
from lollmsbot.agent.rlm import (
    RLMMemoryManager,
    MemoryChunk,
    MemoryChunkType,
    RCBEntry,
    PromptInjectionSanitizer,
)

from lollmsbot.agent.identity import IdentityDetector
from lollmsbot.agent.llm import PromptBuilder
from lollmsbot.agent.tools import ToolParser, FileGenerator
from lollmsbot.agent.logging import AgentLogger

# Import environment detection
from lollmsbot.agent.environment import EnvironmentDetector, detect_environment, EnvironmentInfo


class Agent:
    """
    Core AI agent with RLM-compliant memory system.
    
    Uses the Recursive Language Model memory architecture:
    - EMS (External Memory Store): SQLite-backed compressed storage in ~/.lollmsbot/rlm_memory.db
    - RCB (REPL Context Buffer): Working memory with [[MEMORY:...]] handles visible to LLM
    
    The LLM sees a REPL-style interface where it can conceptually "load" memories.
    The implementation handles actual retrieval and context management.
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
        memory_db_path: Optional[Any] = None,  # Path or None for default
    ) -> None:
        # Core configuration
        self.config: BotConfig = config or BotConfig()
        self.agent_id: str = agent_id or self._generate_id()
        self.name: str = name or self.config.name
        
        # State management
        self._state: AgentState = AgentState.IDLE
        self._state_lock: asyncio.Lock = asyncio.Lock()
        
        # Guardian integration
        self._guardian_enabled = enable_guardian
        self._guardian: Optional[Guardian] = None
        if enable_guardian:
            self._guardian = get_guardian()
            if self._guardian.is_quarantined:
                self._state = AgentState.QUARANTINED
        
        # RLM MEMORY SYSTEM: New double-memory architecture
        self._memory: Optional[RLMMemoryManager] = None
        self._memory_lock: asyncio.Lock = asyncio.Lock()
        self._memory_db_path = memory_db_path
        
        # Legacy compatibility: Identity detection still useful
        self._identity_detector = IdentityDetector()
        self._prompt_builder = PromptBuilder(agent_name=self.name)
        self._logger = AgentLogger(
            agent_name=self.name,
            agent_id=self.agent_id,
            verbose=verbose_logging,
        )
        
        # Environment detection
        self._environment_info: Optional[EnvironmentInfo] = None
        self._environment_detector = EnvironmentDetector()
        
        # Tools management
        self._tools: Dict[str, Tool] = {}
        self._tool_lock: asyncio.Lock = asyncio.Lock()
        self._tool_parser: Optional[ToolParser] = None
        self._file_generator: Optional[FileGenerator] = None
        
        # Skills (lazy initialization)
        self._skills_enabled = enable_skills
        self._skill_registry: Any = None
        self._skill_executor: Any = None
        self._skill_learner: Any = None
        self._skill_lock: asyncio.Lock = asyncio.Lock()
        
        # LoLLMS client (lazy initialization)
        self._lollms_client: Optional[LollmsClient] = None
        self._lollms_client_initialized = False
        
        # Permissions
        self._user_permissions: Dict[str, UserPermissions] = {}
        self._default_permissions = UserPermissions(level=default_permissions)
        self._permission_lock: asyncio.Lock = asyncio.Lock()
        
        # Callbacks
        self._file_delivery_callback: Optional[Callable[[str, List[Dict[str, Any]]], Awaitable[bool]]] = None
        self._tool_event_callback: Optional[ToolEventCallback] = None
        self._skill_event_callback: Optional[Callable[[str, str, Dict[str, Any]], Awaitable[None]]] = None
        self._memory_event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
        
        # Soul (lazy initialization)
        self._soul: Any = None
        self._soul_initialized = False
        
        # Async initialization flag
        self._initialized = False
        
        # Dev mode flag for enhanced logging
        self._dev_mode = verbose_logging
        
        if self._state == AgentState.QUARANTINED:
            self._logger.log_critical("🚨 Agent initialized in QUARANTINED state")
    
    async def initialize(self, gateway_mode: str = "unknown", host_bindings: Optional[List[str]] = None) -> None:
        """Async initialization - must be called before using agent."""
        if self._initialized:
            return
        
        # Detect and store environment info FIRST (before memory init)
        # This ensures environment facts are available for self-knowledge seeding
        self._environment_info = detect_environment(gateway_mode, host_bindings)
        self._logger.log(
            f"🌍 Environment: {self._environment_detector.get_summary()}",
            "cyan", "🖥️"
        )
        
        # Initialize RLM Memory System FIRST (before tools that need it)
        await self._ensure_memory()
        
        # Store environment facts as self-knowledge
        await self._store_environment_knowledge()
        
        self._initialized = True
        self._logger.log(f"✅ Agent {self.name} initialized with environment awareness", "green", "🚀")
    
    def _generate_id(self) -> str:
        """Generate a unique agent ID."""
        import uuid
        return str(uuid.uuid4())
    
    async def _ensure_memory(self) -> RLMMemoryManager:
        """Lazy initialization of RLM Memory Manager."""
        if self._memory is None:
            self._logger.log("Initializing RLM Memory System...", "magenta", "🧠")
            
            self._memory = RLMMemoryManager(
                db_path=self._memory_db_path,
                agent_name=self.name,
                version="0.2.0",
                heartbeat_interval=30.0,
            )
            
            # Set up memory event callbacks
            self._memory.on_memory_load(self._on_memory_loaded)
            self._memory.on_injection_detected(self._on_memory_injection_detected)
            
            # Initialize DB and seed self-knowledge
            await self._memory.initialize()
            
            # Log memory stats
            stats = await self._memory.get_stats()
            self._logger.log(
                f"✅ RLM Memory ready: {stats.get('active_chunks', 0)} chunks in EMS, "
                f"{stats.get('rcb_entries', 0)}/{stats.get('rcb_capacity', 10)} RCB entries",
                "green", "🧠"
            )
        
        return self._memory
    
    async def _store_environment_knowledge(self) -> None:
        """Store detected environment information as self-knowledge."""
        if not self._environment_info or not self._memory:
            return
        
        self._logger.log("Storing environment knowledge...", "cyan", "🌍")
        
        # Convert environment info to facts
        facts = self._environment_info.to_facts()
        
        for fact_id, content, importance in facts:
            # Store in database as self-knowledge
            await self._memory._db.store_self_knowledge(
                knowledge_id=f"environment_{fact_id}",
                category="environment",
                content=content,
                importance=importance,
            )
            
            # Also store as chunk for consistency
            chunk_id = f"env_{fact_id}_{self.agent_id[:8]}"
            try:
                await self._memory.store_in_ems(
                    content=content,
                    chunk_type=MemoryChunkType.SELF_KNOWLEDGE,
                    importance=importance,
                    tags=["environment", "self_knowledge", "runtime", fact_id],
                    summary=f"Environment: {content[:100]}",
                    load_hints=["environment", "platform", "os", "runtime", fact_id],
                    source="environment_detection",
                )
            except Exception as e:
                self._logger.log(f"Failed to store environment fact {fact_id}: {e}", "yellow")
        
        self._logger.log(f"✅ Stored {len(facts)} environment facts", "green", "🌍")
    
    def _on_memory_loaded(self, chunk_id: str, chunk: MemoryChunk) -> Awaitable[None]:
        """Callback when memory is loaded from EMS to RCB."""
        self._logger.log(f"📥 Memory loaded: {chunk_id} ({chunk.chunk_type.name})", "cyan", "🧠")
        return asyncio.sleep(0)  # Dummy awaitable
    
    def _on_memory_injection_detected(self, event: Dict[str, Any]) -> Awaitable[None]:
        """Callback when prompt injection is detected in memory content."""
        self._logger.log(
            f"🛡️ Injection sanitized in memory from {event.get('source', 'unknown')}: "
            f"{len(event.get('detections', []))} patterns neutralized",
            "yellow", "🚨"
        )
        # Could also notify guardian here
        return asyncio.sleep(0)  # Dummy awaitable
    
    # ========== Soul Integration ==========
    
    def _ensure_soul(self) -> Optional[Any]:
        """Lazy initialization of Soul for personality and identity."""
        if not self._soul_initialized:
            self._logger.log("Initializing Soul...", "magenta", "🧬")
            try:
                from lollmsbot.soul import get_soul
                self._soul = get_soul()
                if self._soul and self._soul.name:
                    self.name = self._soul.name
                self._logger.log(
                    f"✅ Soul loaded: {self._soul.name if self._soul else 'default'}",
                    "green", "🧬"
                )
            except Exception as e:
                self._logger.log_error("Failed to initialize Soul", e)
                self._soul = None
            self._soul_initialized = True
        return self._soul
    
    # ========== LoLLMS Client ==========
    
    def _ensure_lollms_client(self) -> Optional[LollmsClient]:
        """Lazy initialization of LoLLMS client."""
        if not self._lollms_client_initialized:
            self._logger.log("Initializing LoLLMS client...", "cyan", "🔗")
            try:
                self._lollms_client = build_lollms_client()
                self._logger.log("✅ LoLLMS client connected", "green", "🔗")
            except Exception as e:
                self._logger.log_error("Failed to initialize LoLLMS client", e)
                self._lollms_client = None
            self._lollms_client_initialized = True
        return self._lollms_client
    
    # ========== Skills ==========
    
    def _ensure_skills_initialized(self) -> bool:
        """Lazy initialization of skills subsystems."""
        if not self._skills_enabled:
            return False
        
        if self._skill_registry is None:
            self._logger.log("Initializing skills subsystems...", "magenta", "📚")
            try:
                from lollmsbot.skills import get_skill_registry, SkillExecutor, SkillLearner
                
                self._skill_registry = get_skill_registry()
                self._skill_executor = SkillExecutor(self, self._skill_registry, self._guardian)
                self._skill_learner = SkillLearner(self._skill_registry, self._skill_executor)
                
                skill_count = len(self._skill_registry._skills) if self._skill_registry else 0
                self._logger.log(f"✅ Skills loaded: {skill_count} skills available", "green", "📚")
                return True
            except Exception as e:
                self._logger.log_error("Failed to initialize skills", e)
                self._skills_enabled = False
                return False
        
        return True
    
    # ========== Tool Management ==========
    
    async def register_tool(self, tool: Tool) -> None:
        """Register a tool with the agent."""
        async with self._tool_lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' already registered")
            
            # SPECIAL HANDLING: Wire RLM memory to HTTP tool
            if tool.name == "http" and hasattr(tool, 'set_rlm_memory'):
                tool.set_rlm_memory(self._memory)
                self._logger.log(f"🔗 Connected RLM memory to HTTP tool", "cyan", "🔗")
            
            self._tools[tool.name] = tool
            self._logger.log(
                f"🔧 Tool registered: {tool.name} (risk={tool.risk_level})",
                "purple", "➕"
            )
            
            # Update tool-dependent components
            self._tool_parser = ToolParser(self._tools)
            self._file_generator = FileGenerator(self._tools)
    
    @property
    def tools(self) -> Dict[str, Tool]:
        return dict(self._tools)
    
    # ========== Callbacks ==========
    
    def set_file_delivery_callback(
        self, 
        callback: Callable[[str, List[Dict[str, Any]]], Awaitable[bool]]
    ) -> None:
        self._file_delivery_callback = callback
    
    def set_tool_event_callback(self, callback: ToolEventCallback) -> None:
        self._tool_event_callback = callback
    
    def set_skill_event_callback(
        self,
        callback: Callable[[str, str, Dict[str, Any]], Awaitable[None]]
    ) -> None:
        self._skill_event_callback = callback
    
    def set_memory_event_callback(
        self,
        callback: Callable[[str, Dict[str, Any]], Awaitable[None]]
    ) -> None:
        self._memory_event_callback = callback
    
    # ========== Permissions ==========
    
    async def check_permission(self, user_id: str, required_level: PermissionLevel) -> bool:
        """Check if a user has the required permission level."""
        async with self._permission_lock:
            user_perms = self._user_permissions.get(user_id, self._default_permissions)
            has_permission = user_perms.level.value >= required_level.value
            
            if not has_permission:
                self._logger.log(
                    f"🚫 Permission denied for {user_id}: needs {required_level.name}, has {user_perms.level.name}",
                    "yellow", "🔒"
                )
            return has_permission
    
    async def set_user_permission(
        self,
        user_id: str,
        level: PermissionLevel,
        allowed_tools: Optional[Set[str]] = None,
        denied_tools: Optional[Set[str]] = None,
        allowed_skills: Optional[Set[str]] = None,
        denied_skills: Optional[Set[str]] = None,
    ) -> None:
        """Set permissions for a specific user."""
        async with self._permission_lock:
            self._user_permissions[user_id] = UserPermissions(
                level=level,
                allowed_tools=allowed_tools,
                allowed_skills=allowed_skills,
                denied_tools=denied_tools or set(),
                denied_skills=denied_skills or set(),
            )
            self._logger.log(f"🔐 Set permission for {user_id}: {level.name}", "cyan", "🔒")
    
    # ========== Main Chat Processing ==========
    
    async def chat(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process chat message - orchestrates all components including RLM memory."""
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        # Log command receipt
        self._logger.log_command_received(user_id, message, context, len(self._tools))
        
        # Check quarantine
        if self._state == AgentState.QUARANTINED:
            return self._quarantine_response()
        
        # Security screening
        security_result = await self._security_check(user_id, message)
        if security_result:
            return security_result
        
        # Permission check
        if not await self.check_permission(user_id, PermissionLevel.BASIC):
            return self._permission_denied_response()
        
        # Set processing state
        state_ok = await self._set_processing_state()
        if not state_ok:
            return self._busy_response()
        
        try:
            result = await self._process_message_rlm(user_id, message, context)
            return result
            
        except Exception as exc:
            # Enhanced dev mode logging with full traceback
            if self._dev_mode:
                tb_str = traceback.format_exc()
                self._logger.log_critical(f"🚨 DEV MODE FULL TRACEBACK:\n{tb_str}")
                # Also log to console for immediate visibility
                print(f"\n{'='*60}")
                print("DEV MODE ERROR TRACEBACK:")
                print(tb_str)
                print(f"{'='*60}\n")
            
            self._logger.log_error("Error in chat processing", exc)
            await self._set_error_state()
            return self._error_response(str(exc))
        
        finally:
            await self._return_to_idle()
    
    # ========== Response Helpers ==========
    
    def _quarantine_response(self) -> Dict[str, Any]:
        return {
            "success": False,
            "response": "Security Alert: System is in protective quarantine.",
            "error": "System quarantined by Guardian",
            "tools_used": [], "skills_used": [], "files_to_send": [],
        }
    
    def _permission_denied_response(self) -> Dict[str, Any]:
        return {
            "success": False, "response": "", "error": "Access denied.",
            "tools_used": [], "skills_used": [], "files_to_send": [],
            "permission_denied": True,
        }
    
    def _busy_response(self) -> Dict[str, Any]:
        return {
            "success": False, "response": "Processing another request.", "error": "Agent busy",
            "tools_used": [], "skills_used": [], "files_to_send": [],
        }
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": f"Processing error: {error}",
            "tools_used": [], "skills_used": [], "files_to_send": [],
        }
    
    # ========== Processing Steps ==========
    
    async def _security_check(
        self,
        user_id: str,
        message: str,
    ) -> Optional[Dict[str, Any]]:
        """Perform security screening. Returns response dict if blocked, None if safe."""
        if not self._guardian:
            return None
        
        self._logger.log("🔍 Running security screening...", "yellow", "🛡️")
        is_safe, event = self._guardian.check_input(message, source=f"user:{user_id}")
        self._logger.log_security_check(is_safe, event)
        
        if not is_safe:
            return {
                "success": False,
                "response": "Message blocked by security screening.",
                "error": f"Security blocked: {event.description if event else 'policy violation'}",
                "security_blocked": True,
                "tools_used": [], "skills_used": [], "files_to_send": [],
            }
        return None
    
    async def _set_processing_state(self) -> bool:
        """Set state to PROCESSING. Returns False if busy."""
        async with self._state_lock:
            if self._state == AgentState.PROCESSING:
                self._logger.log("⚠️ Agent busy - rejecting concurrent request", "yellow", "⏳")
                return False
            old_state = self._state
            self._state = AgentState.PROCESSING
            self._logger.log_state_change(old_state.name, AgentState.PROCESSING.name, "processing new message")
            return True
    
    async def _set_error_state(self) -> None:
        """Set state to ERROR."""
        async with self._state_lock:
            old_state = self._state
            self._state = AgentState.ERROR
            self._logger.log_state_change(old_state.name, AgentState.ERROR.name, "exception occurred")
    
    async def _return_to_idle(self) -> None:
        """Return state to IDLE from PROCESSING."""
        async with self._state_lock:
            if self._state != AgentState.ERROR:
                old_state = self._state
                self._state = AgentState.IDLE
                self._logger.log_state_change(old_state.name, AgentState.IDLE.name, "processing complete")
    
    async def _process_message_rlm(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Main message processing with RLM memory system."""
        tools_used: List[str] = []
        skills_used: List[str] = []
        files_to_send: List[Dict[str, Any]] = []
        
        # Get RLM memory manager
        memory = await self._ensure_memory()
        
        # Identity detection for important information
        identity_detection = self._identity_detector.detect(message)
        if identity_detection.categories and identity_detection.extracted_facts:
            self._logger.log(
                f"🔍 Identity detected: {', '.join(identity_detection.categories)}",
                "gold", "🧠"
            )
            # LOG: About to store identity information
            self._logger.log(
                f"💾 STORING TO RLM EMS: Identity info from {user_id} - "
                f"categories: {identity_detection.categories}, "
                f"facts: {list(identity_detection.extracted_facts.keys())}, "
                f"importance: {identity_detection.importance_boost}",
                "cyan", "📝"
            )
            # Store in EMS with high importance
            try:
                # Build a human-readable summary with actual values, not just keys
                facts_summary = []
                for key, value in identity_detection.extracted_facts.items():
                    if value:  # Only include non-empty values
                        facts_summary.append(f"{key}='{value}'")
                
                summary_text = ", ".join(facts_summary) if facts_summary else "identity information recorded"
                
                chunk_id = await memory.store_in_ems(
                    content=json.dumps(identity_detection.extracted_facts),
                    chunk_type=MemoryChunkType.FACT,
                    importance=identity_detection.importance_boost,
                    tags=identity_detection.categories + ["identity", "auto_detected", f"user_{user_id}"],
                    summary=f"Identity info from {user_id}: {summary_text}",
                    load_hints=list(identity_detection.extracted_facts.keys()) + list(identity_detection.extracted_facts.values())[:5],
                    source=f"identity_detection:{user_id}",
                )
                # LOG: Successfully stored
                self._logger.log(
                    f"✅ RLM EMS STORAGE SUCCESS: chunk_id={chunk_id}, "
                    f"importance={identity_detection.importance_boost}, "
                    f"tags={identity_detection.categories + ['identity', 'auto_detected']}",
                    "green", "🧠"
                )
                # Also notify via memory event callback if set
                if self._memory_event_callback:
                    await self._memory_event_callback("identity_stored", {
                        "chunk_id": chunk_id,
                        "user_id": user_id,
                        "categories": identity_detection.categories,
                        "importance": identity_detection.importance_boost,
                        "facts_keys": list(identity_detection.extracted_facts.keys()),
                    })
            except Exception as e:
                # LOG: Failed to store
                self._logger.log(
                    f"❌ RLM EMS STORAGE FAILED: {str(e)}",
                    "red", "🧠"
                )
                # Continue processing even if storage fails
        
        # Check for informational query (no tools needed)
        if self._identity_detector.is_informational_query(message):
            return await self._handle_informational_query(memory, user_id)
        
        # Check for file generation request
        is_file_request = self._file_generator and self._file_generator.is_file_request(message)
        
        # Build RLM-compliant system prompt with memory REPL interface
        soul = self._ensure_soul()
        system_prompt = await self._build_rlm_system_prompt(
            memory=memory,
            tools=self._tools,
            context=context,
            soul=soul,
            is_file_request=is_file_request,
            user_id=user_id,
        )
        
        # Get LLM response - with full RLM context
        client = self._ensure_lollms_client()
        response, extracted_tools, extracted_files = await self._get_llm_response(
            client, system_prompt, user_id, message, is_file_request, memory
        )
        
        tools_used.extend(extracted_tools)
        files_to_send.extend(extracted_files)
        
        # LOG: Storing conversation in EMS
        self._logger.log(
            f"💾 STORING TO RLM EMS: Conversation turn from {user_id}, "
            f"tools_used: {tools_used}, importance: {2.0 if identity_detection.categories else 1.0}",
            "cyan", "📝"
        )
        
        # Store conversation in EMS (long-term memory)
        try:
            conv_chunk_id = await memory.store_conversation_turn(
                user_id=user_id,
                user_message=message,
                agent_response=response,
                tools_used=tools_used,
                importance=2.0 if identity_detection.categories else 1.0,
            )
            self._logger.log(
                f"✅ RLM EMS CONVERSATION STORED: chunk_id={conv_chunk_id}",
                "green", "🧠"
            )
        except Exception as e:
            self._logger.log(
                f"❌ RLM EMS CONVERSATION STORAGE FAILED: {str(e)}",
                "red", "🧠"
            )
        
        # Deliver files
        await self._deliver_files(user_id, files_to_send)
        
        # Build final response
        final_response = self._build_final_response(response, files_to_send)
        
        # Log completion
        self._logger.log_response_sent(user_id, len(final_response), tools_used)
        
        return {
            "success": True,
            "response": final_response,
            "tools_used": tools_used,
            "skills_used": skills_used,
            "files_to_send": files_to_send,
            "memory_stored": True,
        }
    
    async def _build_rlm_system_prompt(
        self,
        memory: RLMMemoryManager,
        tools: Dict[str, Tool],
        context: Optional[Dict[str, Any]],
        soul: Optional[Any],
        is_file_request: bool,
        user_id: str,
    ) -> str:
        """Build system prompt with RLM memory REPL interface."""
        
        # Start with Soul's identity if available
        if soul:
            base_prompt = soul.generate_system_prompt(context)
        else:
            base_prompt = f"You are {self.name}, an AI assistant with RLM (Recursive Language Model) memory."
        
        # Add RLM memory interface explanation - with web content specifics
        rlm_explanation = """
## Your Memory System (RLM Architecture)

You have a **double-memory structure** following MIT CSAIL's Recursive Language Model research:

1. **External Memory Store (EMS)**: All your long-term memories are stored here as compressed, 
   sanitized chunks with importance-weighted retention. This includes:
   - Conversation history
   - Important facts about users
   - Web content you have fetched
   - Self-knowledge about your own systems

2. **REPL Context Buffer (RCB)**: What you see below - your working memory with loadable handles.

### How to Use Your Memory

The RCB shows memory handles like: [[MEMORY:abc123|{"type": "SELF_KNOWLEDGE", "summary": "..."}]]

To access full content of a memory, reference its handle in your reasoning. The system 
automatically retrieves the full content when you need it.

### WEB CONTENT - Special Handling

When you use the HTTP tool to fetch a URL:
- The full content is stored as a WEB_CONTENT chunk in EMS
- You receive a memory handle like [[MEMORY:web_abc123]]
- The chunk is automatically loaded into your RCB
- YOU MUST read and process this actual content - never hallucinate

CRITICAL: When a user asks you to summarize web content:
1. Use the HTTP tool to fetch it
2. The content becomes available via memory handle
3. ACTUALLY READ the content through the handle
4. Provide a real summary based on what was fetched
5. NEVER make up a generic description
"""
        
        # Get formatted RCB (working memory) from memory manager
        rcb_content = memory.format_rcb_for_prompt(max_chars=8000)  # Increased for web content
        
        # Add tool documentation with HTTP-specific instructions
        tool_list = self._prompt_builder._format_tool_list(tools)
        
        # HTTP-specific instructions - emphasizing RLM integration
        http_instructions = ""
        if "http" in tools:
            http_instructions = """
### HTTP TOOL - RLM-Integrated Usage

When you need to fetch content from a URL:
1. Call [[TOOL:http|{"method": "get", "url": "THE_URL"}]]
2. The tool stores content in RLM memory and returns a memory handle
3. The handle is automatically loaded into your RCB below
4. YOU MUST reference this handle to access the content
5. Provide accurate responses based on the ACTUAL fetched content

The HTTP tool does NOT return raw text - it returns a memory handle.
The content is in your RLM system. Use it via the memory handles in your RCB.
"""
        
        # Add environment context if available
        env_context = ""
        if self._environment_info:
            env = self._environment_info
            env_context = f"""
## Your Runtime Environment

You are running on:
- **Platform**: {env.os_system} {env.os_release} ({env.os_machine})
- **Python**: {env.python_version} ({env.python_implementation})
- **Virtual Environment**: {"Yes - " + env.virtualenv_path if env.in_virtualenv else "No - system Python"}
- **Container**: {"Docker container" if env.in_docker else "WSL" if env.in_wsl else "Bare metal/VM"}
- **Working Directory**: {env.working_directory}
- **Data Storage**: {env.lollmsbot_data_dir}
- **Current Channel/Mode**: {env.gateway_mode}
"""
            if env.host_bindings:
                env_context += f"- **Network Access**: {', '.join(env.host_bindings)}\n"
        
        # Search for user-specific identity memories and load them
        user_memories = []
        try:
            # FIXED: Search with just user_id tag, not "user_id identity" - the tag is stored as f"user_{user_id}" not "user_{user_id} identity"
            # Use user_id directly to match the tag pattern
            identity_results = await memory.search_ems(
                query=user_id,  # FIXED: Changed from f"user_{user_id} identity" to just user_id
                chunk_types=[MemoryChunkType.FACT],
                min_importance=2.0,
                limit=5,
            )
            for chunk, score in identity_results:
                # Only process identity-related chunks
                tags = chunk.tags or []
                if "identity" not in tags and "creator_identity" not in tags:
                    continue
                    
                content = chunk.decompress_content()
                try:
                    facts = json.loads(content)
                    # Format facts nicely for display
                    fact_lines = []
                    for k, v in facts.items():
                        if v:  # Only show non-empty values
                            fact_lines.append(f"{k}: {v}")
                    if fact_lines:
                        user_memories.append(f"User identity - {', '.join(fact_lines)}")
                except:
                    # If not valid JSON, show summary
                    user_memories.append(f"User memory: {chunk.summary or content[:100]}")
            
            # Also search for any memories tagged with this user more broadly
            user_conversation_results = await memory.search_ems(
                query=user_id,
                chunk_types=[MemoryChunkType.CONVERSATION, MemoryChunkType.FACT],
                min_importance=1.0,
                limit=3,
            )
            for chunk, score in user_conversation_results:
                if chunk.chunk_type == MemoryChunkType.CONVERSATION:
                    # Don't load full conversation content, just note it exists
                    user_memories.append(f"Previous conversation on {chunk.created_at.strftime('%Y-%m-%d') if hasattr(chunk, 'created_at') and chunk.created_at else 'earlier'}")
        except Exception as e:
            self._logger.log(f"Error loading user memories: {e}", "yellow")
        
        user_context = ""
        if user_memories:
            user_context = """
## Information About This User

"""
            for mem in user_memories:
                user_context += f"- {mem}\n"
        
        # Combine all parts
        parts = [
            base_prompt,
            rlm_explanation,
            env_context,
            user_context,
            "",
            "=" * 60,
            "YOUR CURRENT WORKING MEMORY (RCB)",
            "=" * 60,
            rcb_content,
            "",
            "=" * 60,
            "AVAILABLE TOOLS",
            "=" * 60,
            tool_list,
            http_instructions,
        ]
        
        if is_file_request:
            parts.extend([
                "",
                "FILE GENERATION MODE: Use filesystem tool to create files. Do not output code directly.",
            ])
        
        if context:
            channel = context.get("channel", "unknown")
            parts.append(f"\nCurrent channel: {channel}")
        
        return "\n".join(parts)
    
    async def _handle_informational_query(
        self,
        memory: RLMMemoryManager,
        user_id: str,
    ) -> Dict[str, Any]:
        """Handle simple informational query about capabilities."""
        self._logger.log("ℹ️ Detected informational query - using memory + tools list", "cyan", "📝")
        
        # Get self-knowledge from memory
        self_knowledge = await memory.search_ems(
            query="self knowledge purpose architecture",
            chunk_types=[MemoryChunkType.SELF_KNOWLEDGE],
            min_importance=5.0,
            limit=3,
        )
        
        # Get environment knowledge
        env_knowledge = await memory.search_ems(
            query="environment platform os runtime",
            chunk_types=[MemoryChunkType.SELF_KNOWLEDGE],
            min_importance=5.0,
            limit=3,
        )
        
        # Build response from self-knowledge and tools
        knowledge_parts = []
        for chunk, relevance in self_knowledge:
            content = chunk.decompress_content()
            # Extract key sentences
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
            knowledge_parts.extend(sentences[:2])
        
        # Add environment info
        env_parts = []
        for chunk, relevance in env_knowledge:
            content = chunk.decompress_content()
            env_parts.append(content)
        
        # Get memory stats for transparency
        stats = await memory.get_stats()
        
        # Build environment summary
        env_summary = ""
        if self._environment_info:
            env = self._environment_info
            env_summary = (
                f"I'm running on {env.os_system} {env.os_release} "
                f"with Python {env.python_version}"
            )
            if env.in_docker:
                env_summary += " inside a Docker container"
            elif env.in_wsl:
                env_summary += " in WSL"
            if env.in_virtualenv:
                env_summary += f" (virtualenv: {env.virtualenv_path})"
            env_summary += "."
        
        tool_list = "\n".join([
            f"- **{name}**: {tool.description[:80]}"
            for name, tool in self._tools.items()
        ])
        
        response = (
            f"I am {self.name}, an AI assistant with RLM (Recursive Language Model) memory.\n\n"
            f"**My Environment**: {env_summary}\n\n"
            f"**Memory System**: {stats.get('active_chunks', 0)} long-term memory chunks, "
            f"{stats.get('rcb_entries', 0)}/{stats.get('rcb_capacity', 10)} RCB entries. "
            f"I use recursive loading to manage unlimited context beyond my native window.\n\n"
            f"**Available Tools**:\n{tool_list}\n\n"
            f"Just ask me to use any of these, or tell me about something you'd like me to remember!"
        )
        
        self._logger.log_response_sent(user_id, len(response), [])
        
        return {
            "success": True,
            "response": response,
            "tools_used": [],
            "skills_used": [],
            "files_to_send": [],
        }
    
    async def _get_llm_response(
        self,
        client: Optional[LollmsClient],
        system_prompt: str,
        user_id: str,
        message: str,
        is_file_request: bool,
        memory: RLMMemoryManager,
    ) -> tuple[str, List[str], List[Dict[str, Any]]]:
        """Get response from LLM, with RLM memory-aware processing."""
        if not client:
            # Fallback: try direct tool execution
            return await self._try_direct_execution(message, user_id)
        
        # Build conversation context with recent EMS conversations loaded into RCB
        # Search for recent conversations with this user
        recent_convs = await memory.search_ems(
            query=user_id,
            chunk_types=[MemoryChunkType.CONVERSATION],
            limit=3,
        )
        
        # Load top conversations into RCB temporarily
        for chunk, _ in recent_convs:
            await memory.load_from_ems(chunk.chunk_id, add_to_rcb=True)
        
        # Also search for and load any high-importance facts about this user
        # FIXED: Use user_id directly to match tags properly
        user_facts = await memory.search_ems(
            query=user_id,
            chunk_types=[MemoryChunkType.FACT],
            min_importance=2.0,
            limit=3,
        )
        for chunk, _ in user_facts:
            # Only load identity-related facts
            tags = chunk.tags or []
            if "identity" in tags or "creator_identity" in tags:
                await memory.load_from_ems(chunk.chunk_id, add_to_rcb=True)
        
        # Refresh system prompt with updated RCB (now includes loaded memories)
        # We need to rebuild the prompt to include the updated RCB
        # Actually, let's rebuild to get fresh RCB
        soul = self._ensure_soul()
        fresh_prompt = await self._build_rlm_system_prompt(
            memory=memory,
            tools=self._tools,
            context={"channel": "chat"},  # Simplified context
            soul=soul,
            is_file_request=is_file_request,
            user_id=user_id,
        )
        
        # Build final prompt with conversation
        prompt = f"{fresh_prompt}\n\nUser: {message}\nAssistant:"
        
        self._logger.log_llm_call(len(prompt), fresh_prompt[:500])
        
        # Call LLM
        llm_response = client.generate_text(
            prompt=prompt,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        
        has_tools = "[[TOOL:" in llm_response or "<function_calls>" in llm_response
        self._logger.log_llm_response(len(llm_response), has_tools)
        
        # Parse and execute tools
        if self._tool_parser:
            result = await self._tool_parser.parse_and_execute(llm_response, user_id, None)
            
            final_response, tools_used, files_generated = result
            
            # Strip any [[MEMORY:...]] references from the response - they shouldn't be visible to user
            final_response = self._strip_memory_handles(final_response)
            
            # NO MORE SECOND PASS - RLM handles this naturally!
            # The HTTP tool stores content in RLM, and the RCB shows the handle.
            # If the LLM didn't properly use the content, that's a prompt engineering issue,
            # not a workflow issue. The RLM architecture is designed to make content
            # available via the RCB in the next turn if needed.
            
            return final_response, tools_used, files_generated
        
        # Strip memory handles from raw response too
        clean_response = self._strip_memory_handles(llm_response.strip())
        return clean_response, [], []
    
    def _strip_memory_handles(self, text: str) -> str:
        """Remove [[MEMORY:...]] references from response text."""
        import re
        # Pattern matches [[MEMORY:chunk_id|{metadata}]] or [[MEMORY:chunk_id]]
        pattern = r'\[\[MEMORY:[^\]]+\]\]'
        return re.sub(pattern, '', text).strip()
    
    async def _try_direct_execution(
        self,
        message: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, List[str], List[Dict[str, Any]]]:
        """Try direct tool execution without LLM."""
        self._logger.log("⚠️ No LLM client - attempting direct tool execution", "yellow", "🔧")
        
        if not self._file_generator:
            return (
                f"I received: '{message}'. However, I'm not connected to a language model backend.",
                [],
                []
            )
        
        response = await self._file_generator.try_direct_execution(message, user_id, context)
        
        if response:
            tools_used = []
            files = []
            if self._file_generator.last_tool_result:
                tools_used = [self._file_generator.last_tool_name or "unknown"]
                files = self._file_generator.last_tool_result.files_to_send or []
            return response, tools_used, files
        
        return (
            f"I received: '{message}'. However, I'm not connected to a language model backend.",
            [],
            []
        )
    
    async def _deliver_files(self, user_id: str, files: List[Dict[str, Any]]) -> None:
        """Trigger file delivery callback if files were generated."""
        if not files or not self._file_delivery_callback:
            return
        
        self._logger.log_file_generation(len(files), [f.get("filename", "unnamed") for f in files])
        self._logger.log(f"📦 Delivering {len(files)} file(s) to user {user_id}", "green")
        
        success = await self._file_delivery_callback(user_id, files)
        if not success:
            self._logger.log("⚠️ File delivery callback reported failure", "yellow", "❌")
    
    def _build_final_response(
        self,
        response: str,
        files_to_send: List[Dict[str, Any]],
    ) -> str:
        """Build final response with file information if needed."""
        if not files_to_send:
            return response
        
        # Check if response already mentions files
        file_keywords = ["created", "saved", "generated", "built", "file"]
        has_file_mention = any(kw in response.lower() for kw in file_keywords)
        
        if has_file_mention:
            return response
        
        # Add file mention
        file_names = [f.get("filename", "unnamed") for f in files_to_send]
        return f"{response}\n\n📁 I've created: {', '.join(file_names)}"
    
    # ========== Properties ==========
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    @property
    def guardian(self) -> Optional[Guardian]:
        return self._guardian
    
    @property
    def skill_registry(self) -> Any:
        self._ensure_skills_initialized()
        return self._skill_registry
    
    @property
    def environment_info(self) -> Optional[EnvironmentInfo]:
        """Get detected environment information."""
        return self._environment_info
    
    # ========== Skill Methods ==========
    
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
        
        # Get skills
        if search_query:
            skills = self._skill_registry.search(search_query)
            skills = [s for s, _ in skills]
        elif category:
            skills = self._skill_registry.list_skills(category=category)
        else:
            skills = list(self._skill_registry._skills.values())
        
        # Filter by permissions
        if perms.allowed_skills is not None:
            skills = [s for s in skills if s.name in perms.allowed_skills]
        skills = [s for s in skills if s.name not in perms.denied_skills]
        
        # Format results
        available_tools = set(self._tools.keys())
        available_skills = set(self._skill_registry._skills.keys())
        
        return [
            {
                "name": s.name,
                "version": s.metadata.version,
                "description": s.metadata.description,
                "complexity": s.metadata.complexity.name,
                "tags": s.metadata.tags,
                "confidence": s.metadata.confidence_score,
                "can_execute": s.check_dependencies(available_tools, available_skills)[0],
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
        """Execute a skill with full permission checking."""
        if not self._ensure_skills_initialized():
            return SkillResult(
                success=False, result=None, skill_name=skill_name, skill_version="unknown",
                execution_id="", error="Skills not initialized",
            )
        
        perms = self._user_permissions.get(user_id, self._default_permissions)
        
        # Permission checks
        self._logger.log(f"🔒 Checking permissions for skill '{skill_name}'", "yellow", "🛡️")
        
        if perms.level.value < PermissionLevel.SKILLS.value:
            return self._skill_permission_denied(skill_name, "lacks SKILLS level")
        
        if perms.denied_skills and skill_name in perms.denied_skills:
            return self._skill_permission_denied(skill_name, "in denylist")
        
        if perms.allowed_skills is not None and skill_name not in perms.allowed_skills:
            return self._skill_permission_denied(skill_name, "not in allowlist")
        
        # Notify start
        if self._skill_event_callback:
            await self._skill_event_callback("skill_start", skill_name, {"inputs": list(inputs.keys())})
        
        # Execute
        self._logger.log(f"🎯 Executing skill '{skill_name}'...", "magenta", "📚")
        
        from datetime import datetime
        start = datetime.now()
        result = await self._skill_executor.execute(skill_name, inputs, context)
        duration = (datetime.now() - start).total_seconds()
        
        # Build result
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
        
        # Log completion
        self._logger.log_skill_execution(skill_name, skill_result.success)
        
        # Store skill execution in RLM memory
        if self._initialized and self._memory:
            await self._memory.store_in_ems(
                content=json.dumps({
                    "skill": skill_name,
                    "inputs": inputs,
                    "success": skill_result.success,
                    "duration": skill_result.duration_seconds,
                }),
                chunk_type=MemoryChunkType.SKILL_EXECUTION,
                importance=3.0 if skill_result.success else 5.0,  # Higher importance if failed (to avoid repeating)
                tags=["skill", skill_name, "execution"],
                summary=f"Skill '{skill_name}' execution: {'success' if skill_result.success else 'failed'}",
                source=f"skill_execution:{user_id}",
            )
        
        # Notify completion
        if self._skill_event_callback:
            event_type = "skill_complete" if skill_result.success else "skill_error"
            await self._skill_event_callback(
                event_type,
                skill_name,
                {"duration": skill_result.duration_seconds, "steps": len(skill_result.steps_taken)},
            )
        
        return skill_result
    
    def _skill_permission_denied(self, skill_name: str, reason: str) -> SkillResult:
        """Build permission denied skill result."""
        self._logger.log(f"🚫 Skill '{skill_name}' {reason}", "red", "❌")
        return SkillResult(
            success=False,
            result=None,
            skill_name=skill_name,
            skill_version="unknown",
            execution_id="",
            error=f"Permission denied: {reason}",
        )
    
    # ========== Memory Maintenance ==========
    
    async def run_memory_maintenance(self) -> Dict[str, Any]:
        """Run RLM memory maintenance (forgetting curve, compression)."""
        if not self._initialized or not self._memory:
            return {"error": "Memory not initialized"}
        
        self._logger.log("🧠 Running RLM memory maintenance...", "cyan", "💓")
        
        # Apply forgetting curve
        forgetting_result = await self._memory.apply_forgetting_curve()
        
        # Get updated stats
        stats = await self._memory.get_stats()
        
        self._logger.log(
            f"✅ Memory maintenance: {forgetting_result.get('archived_count', 0)} archived, "
            f"{stats.get('active_chunks', 0)} total chunks",
            "green", "💓"
        )
        
        return {
            "forgetting_applied": forgetting_result,
            "stats": stats,
        }
    
    async def close(self) -> None:
        """Cleanup and close resources."""
        if self._memory:
            await self._memory.close()
            self._memory = None
