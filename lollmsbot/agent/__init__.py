"""
Agent module for LollmsBot - RLM Memory Edition with Semantic Anchors

Updated to use RLM-compliant memory manager with double-memory structure:
- External Memory Store (EMS): Compressed, chunked long-term storage  
- REPL Context Buffer (RCB): Working memory with loadable handles

CRITICAL: Web content fetched via HTTP is now properly loaded into context
with full text available via semantic anchors that describe memory contents.
"""

from __future__ import annotations

import asyncio
import json
import traceback
import re
from dataclasses import field, dataclass
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


@dataclass
class MemoryAnchor:
    """Internal representation - never shown to users."""
    chunk_id: str
    anchor_type: str  # "web_content", "user_fact", "conversation", "self_knowledge", "file"
    title: str
    summary: str
    tags: List[str]
    content_preview: str
    full_content_available: bool = True
    importance: float = 1.0


class Agent:
    """
    Core AI agent with RLM memory.
    
    Memory architecture (internal, never exposed to users):
    - EMS: Long-term compressed storage
    - RCB: Working memory
    
    The LLM receives loaded content directly in context and responds naturally.
    No memory IDs, no meta-commentary about memory systems.
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
        memory_db_path: Optional[Any] = None,
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
        
        # RLM MEMORY SYSTEM
        self._memory: Optional[RLMMemoryManager] = None
        self._memory_lock: asyncio.Lock = asyncio.Lock()
        self._memory_db_path = memory_db_path
        
        self._identity_detector = IdentityDetector()
        self._prompt_builder = PromptBuilder(agent_name=self.name)
        self._logger = AgentLogger(
            agent_name=self.name,
            agent_id=self.agent_id,
            verbose=verbose_logging,
        )
        
        self._environment_info: Optional[EnvironmentInfo] = None
        self._environment_detector = EnvironmentDetector()
        
        # Tools
        self._tools: Dict[str, Tool] = {}
        self._tool_lock: asyncio.Lock = asyncio.Lock()
        self._tool_parser: Optional[ToolParser] = None
        self._file_generator: Optional[FileGenerator] = None
        
        # Skills
        self._skills_enabled = enable_skills
        self._skill_registry: Any = None
        self._skill_executor: Any = None
        self._skill_learner: Any = None
        self._skill_lock: asyncio.Lock = asyncio.Lock()
        
        # LoLLMS client
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
        
        # Soul
        self._soul: Any = None
        self._soul_initialized = False
        
        self._initialized = False
        self._dev_mode = verbose_logging
        
        # Memory cache - internal only
        self._anchor_cache: Dict[str, str] = {}
        self._loaded_anchors: Set[str] = set()
        
        # Fast conversation cache
        self._recent_conversations: Dict[str, List[ConversationTurn]] = {}
        self._max_recent_history: int = 10
        
        self._debug_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        
        if self._state == AgentState.QUARANTINED:
            self._logger.log_critical("🚨 Agent initialized in QUARANTINED state")
    
    def set_debug_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._debug_callback = callback
    
    def _debug(self, event_type: str, data: Dict[str, Any]) -> None:
        if self._debug_callback:
            try:
                self._debug_callback(event_type, data)
            except Exception:
                pass
    
    async def initialize(self, gateway_mode: str = "unknown", host_bindings: Optional[List[str]] = None) -> None:
        if self._initialized:
            return
        
        self._environment_info = detect_environment(gateway_mode, host_bindings)
        self._logger.log(
            f"🌍 Environment: {self._environment_detector.get_summary()}",
            "cyan", "🖥️"
        )
        
        await self._ensure_memory()
        await self._store_environment_knowledge()
        
        self._initialized = True
        self._logger.log(f"✅ Agent {self.name} initialized with environment awareness", "green", "🚀")
    
    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())
    
    async def _ensure_memory(self) -> RLMMemoryManager:
        if self._memory is None:
            self._logger.log("Initializing RLM Memory System...", "magenta", "🧠")
            
            self._memory = RLMMemoryManager(
                db_path=self._memory_db_path,
                agent_name=self.name,
                version="0.2.0",
                heartbeat_interval=30.0,
            )
            
            self._memory.on_memory_load(self._on_memory_loaded)
            self._memory.on_injection_detected(self._on_memory_injection_detected)
            
            await self._memory.initialize()
            
            stats = await self._memory.get_stats()
            self._logger.log(
                f"✅ RLM Memory ready: {stats.get('active_chunks', 0)} chunks in EMS, "
                f"{stats.get('rcb_entries', 0)}/{stats.get('rcb_capacity', 10)} RCB entries",
                "green", "🧠"
            )
        
        return self._memory
    
    async def _store_environment_knowledge(self) -> None:
        if not self._environment_info or not self._memory:
            return
        
        self._logger.log("Storing environment knowledge...", "cyan", "🌍")
        
        facts = self._environment_info.to_facts()
        
        for fact_id, content, importance in facts:
            await self._memory._db.store_self_knowledge(
                knowledge_id=f"environment_{fact_id}",
                category="environment",
                content=content,
                importance=importance,
            )
            
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
        self._logger.log(f"📥 Memory loaded: {chunk_id} ({chunk.chunk_type.name})", "cyan", "🧠")
        return asyncio.sleep(0)
    
    def _on_memory_injection_detected(self, event: Dict[str, Any]) -> Awaitable[None]:
        self._logger.log(
            f"🛡️ Injection sanitized in memory from {event.get('source', 'unknown')}: "
            f"{len(event.get('detections', []))} patterns neutralized",
            "yellow", "🚨"
        )
        return asyncio.sleep(0)
    
    async def _search_and_load_memories(
        self,
        memory: RLMMemoryManager,
        user_id: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories and return loaded content.
        PRIORITIZES web_content chunks with substantive content.
        """
        loaded_content = []
        seen_chunks = set()
        
        # Extract key terms from query for better searching
        key_terms = self._extract_key_terms(query)
        self._logger.log(f"🔍 Searching for: '{query}' (terms: {key_terms})", "cyan")
        
        # CRITICAL: Search WITHOUT type restrictions first to get ALL relevant content
        # The database search searches in summary, tags, load_hints AND content preview
        all_results = await memory.search_ems(
            query=query,
            limit=20,
        )
        
        self._logger.log(f"🔍 Found {len(all_results)} total chunks", "cyan")
        
        # Also search with key terms
        if len(key_terms) > 0:
            key_term_query = " ".join(key_terms)
            key_results = await memory.search_ems(
                query=key_term_query,
                limit=15,
            )
            # Merge and deduplicate
            for chunk, score in key_results:
                found = False
                for existing_chunk, existing_score in all_results:
                    if existing_chunk.chunk_id == chunk.chunk_id:
                        found = True
                        break
                if not found:
                    all_results.append((chunk, score))
        
        self._logger.log(f"🔍 Total after key term search: {len(all_results)} chunks", "cyan")
        
        # Sort by score descending
        all_results.sort(key=lambda x: -x[1])
        
        # Process results - prioritize web content
        web_content_loaded = 0
        other_content_loaded = 0
        
        for chunk, score in all_results:
            if chunk.chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk.chunk_id)
            
            try:
                content = chunk.decompress_content()
                
                # Skip if too short
                if len(content) < 100:
                    continue
                
                # Prioritize web content
                if chunk.chunk_type == MemoryChunkType.WEB_CONTENT and web_content_loaded < 3:
                    # For web content, we want substantial excerpts
                    # But limit to avoid overwhelming the context
                    excerpt = content[:15000] if len(content) > 15000 else content
                    
                    loaded_content.append({
                        "chunk_id": chunk.chunk_id,
                        "type": "web_content",
                        "title": chunk.summary or "Web content",
                        "content": excerpt,
                        "full_length": len(content),
                        "score": score,
                    })
                    
                    self._anchor_cache[chunk.chunk_id] = content
                    self._loaded_anchors.add(chunk.chunk_id)
                    web_content_loaded += 1
                    
                    self._logger.log(
                        f"📥 Loaded web content: {chunk.chunk_id[:20]}... ({len(content)} chars, score={score:.1f})",
                        "green"
                    )
                    
                elif chunk.chunk_type == MemoryChunkType.FACT and other_content_loaded < 2:
                    loaded_content.append({
                        "chunk_id": chunk.chunk_id,
                        "type": "fact",
                        "title": "User information",
                        "content": content[:2000],
                        "full_length": len(content),
                        "score": score,
                    })
                    self._anchor_cache[chunk.chunk_id] = content
                    other_content_loaded += 1
                    
                elif chunk.chunk_type == MemoryChunkType.CONVERSATION and other_content_loaded < 2:
                    # Parse conversation
                    try:
                        conv_data = json.loads(content)
                        user_msg = conv_data.get("user_message", "")
                        agent_resp = conv_data.get("agent_response", "")
                        
                        if len(user_msg) > 20:
                            loaded_content.append({
                                "chunk_id": chunk.chunk_id,
                                "type": "conversation",
                                "title": "Previous conversation",
                                "content": f"User: {user_msg}\nAssistant: {agent_resp[:1000]}",
                                "full_length": len(content),
                                "score": score,
                            })
                            other_content_loaded += 1
                    except:
                        pass
                
                # Stop once we have enough content
                if web_content_loaded >= 2 and other_content_loaded >= 1:
                    break
                    
            except Exception as e:
                self._logger.log(f"Failed to load {chunk.chunk_id}: {e}", "yellow")
        
        self._logger.log(
            f"✅ Loaded {web_content_loaded} web content + {other_content_loaded} other = {len(loaded_content)} total",
            "green" if web_content_loaded > 0 else "yellow"
        )
        
        return loaded_content
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract meaningful key terms from query for searching."""
        stop_words = {"do", "you", "remember", "my", "the", "a", "an", "is", "are", "was", "were",
                     "i", "me", "my", "mine", "have", "has", "had", "this", "that", "these", "those",
                     "can", "could", "would", "will", "about", "tell", "what", "who", "where", "when",
                     "how", "why", "and", "or", "but", "if", "then", "than", "so", "very", "just",
                     "now", "here", "there", "with", "from", "for", "to", "of", "in", "on", "at"}
        
        words = re.findall(r'\b\w+\b', query.lower())
        
        key_terms = []
        for word in words:
            if word not in stop_words and (len(word) >= 3 or word.isdigit()):
                key_terms.append(word)
        
        # Add variations
        expanded_terms = list(key_terms)
        for term in key_terms:
            if " " in query and term in query.lower():
                expanded_terms.append(term.replace(" ", "-"))
                expanded_terms.append(term.replace(" ", ""))
        
        return expanded_terms
    
    def _ensure_soul(self) -> Optional[Any]:
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
    
    def _ensure_lollms_client(self) -> Optional[LollmsClient]:
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
    
    def _ensure_skills_initialized(self) -> bool:
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
    
    async def register_tool(self, tool: Tool) -> None:
        async with self._tool_lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool '{tool.name}' already registered")
            
            if tool.name == "http" and hasattr(tool, 'set_rlm_memory'):
                tool.set_rlm_memory(self._memory)
                self._logger.log(f"🔗 Connected RLM memory to HTTP tool", "cyan", "🔗")
            
            self._tools[tool.name] = tool
            self._logger.log(
                f"🔧 Tool registered: {tool.name} (risk={tool.risk_level})",
                "purple", "➕"
            )
            
            self._tool_parser = ToolParser(self._tools)
            self._file_generator = FileGenerator(self._tools)
    
    @property
    def tools(self) -> Dict[str, Tool]:
        return dict(self._tools)
    
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
    
    async def check_permission(self, user_id: str, required_level: PermissionLevel) -> bool:
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
        async with self._permission_lock:
            self._user_permissions[user_id] = UserPermissions(
                level=level,
                allowed_tools=allowed_tools,
                allowed_skills=allowed_skills,
                denied_tools=denied_tools or set(),
                denied_skills=denied_skills or set(),
            )
            self._logger.log(f"🔐 Set permission for {user_id}: {level.name}", "cyan", "🔒")
    
    async def chat(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            await self.initialize()
        
        self._logger.log_command_received(user_id, message, context, len(self._tools))
        
        if self._state == AgentState.QUARANTINED:
            return self._quarantine_response()
        
        security_result = await self._security_check(user_id, message)
        if security_result:
            return security_result
        
        if not await self.check_permission(user_id, PermissionLevel.BASIC):
            return self._permission_denied_response()
        
        state_ok = await self._set_processing_state()
        if not state_ok:
            return self._busy_response()
        
        try:
            result = await self._process_message_rlm(user_id, message, context)
            return result
            
        except Exception as exc:
            if self._dev_mode:
                tb_str = traceback.format_exc()
                self._logger.log_critical(f"🚨 DEV MODE FULL TRACEBACK:\n{tb_str}")
                print(f"\n{'='*60}")
                print("DEV MODE ERROR TRACEBACK:")
                print(tb_str)
                print(f"{'='*60}\n")
            
            self._logger.log_error("Error in chat processing", exc)
            await self._set_error_state()
            return self._error_response(str(exc))
        
        finally:
            await self._return_to_idle()
    
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
    
    async def _security_check(
        self,
        user_id: str,
        message: str,
    ) -> Optional[Dict[str, Any]]:
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
        async with self._state_lock:
            if self._state == AgentState.PROCESSING:
                self._logger.log("⚠️ Agent busy - rejecting concurrent request", "yellow", "⏳")
                return False
            old_state = self._state
            self._state = AgentState.PROCESSING
            self._logger.log_state_change(old_state.name, AgentState.PROCESSING.name, "processing new message")
            return True
    
    async def _set_error_state(self) -> None:
        async with self._state_lock:
            old_state = self._state
            self._state = AgentState.ERROR
            self._logger.log_state_change(old_state.name, AgentState.ERROR.name, "exception occurred")
    
    async def _return_to_idle(self) -> None:
        async with self._state_lock:
            if self._state != AgentState.ERROR:
                old_state = self._state
                self._state = AgentState.IDLE
                self._logger.log_state_change(old_state.name, AgentState.IDLE.name, "processing complete")
    
    def _get_conversation_history(self, user_id: str) -> List[ConversationTurn]:
        return self._recent_conversations.get(user_id, [])[-self._max_recent_history:]
    
    def _format_history_for_prompt(self, history: List[ConversationTurn]) -> str:
        if not history:
            return ""
        
        lines = ["", "=" * 60, "CONVERSATION HISTORY", "=" * 60, ""]
        
        for turn in history[-5:]:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.agent_response}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _extract_urls_and_facts(self, message: str) -> List[Dict[str, Any]]:
        """Extract URLs and important facts from user messages for memory storage."""
        extracted = []
        
        # URL pattern - matches http/https URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, message)
        
        for url in urls:
            # Clean up common trailing punctuation
            url = url.rstrip('.,;:!?)')
            extracted.append({
                "type": "url",
                "content": url,
                "context": message[:200],  # Surrounding context
            })
        
        # Also look for domain-like patterns that might be URLs without protocol
        domain_pattern = r'\b([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        domains = re.findall(domain_pattern, message)
        # domains is list of tuples due to groups, flatten it
        domains = [d[0] if isinstance(d, tuple) else d for d in domains]
        
        # Filter out common false positives
        false_positives = {'com', 'org', 'net', 'io', 'ai', 'co', 'app', 'dev'}
        for domain in domains:
            # Skip if just TLD or common suffix without domain
            parts = domain.lower().split('.')
            if len(parts) < 2:
                continue
            # Check if it looks like a real domain (has substance before TLD)
            if parts[0] not in false_positives and len(parts[0]) > 1:
                # Check it's not already captured as full URL
                full_url = f"https://{domain}"
                if not any(full_url.startswith(u) or u.startswith(full_url) for u in urls):
                    extracted.append({
                        "type": "url",
                        "content": full_url,
                        "context": f"Domain mentioned: {domain}",
                    })
        
        return extracted
    
    async def _store_extracted_facts(self, user_id: str, message: str, memory: RLMMemoryManager) -> None:
        """Extract and store important facts, URLs, and references from user message."""
        extracted = self._extract_urls_and_facts(message)
        
        for item in extracted:
            if item["type"] == "url":
                url = item["content"]
                context = item["context"]
                
                # Store as high-importance fact about user's content
                try:
                    # Create a descriptive entry
                    content = f"User shared URL: {url}\nContext: {context[:300]}"
                    
                    await memory.store_in_ems(
                        content=content,
                        chunk_type=MemoryChunkType.FACT,
                        importance=8.0,  # High importance for user-shared URLs
                        tags=["url", "user_shared", "reference", "important"],
                        summary=f"URL from user: {url[:80]}",
                        load_hints=[url, "url", "link", "website", "homo", "zombius", "novel", "book"],
                        source=f"extracted_url:{user_id}",
                    )
                    self._logger.log(f"🔗 Stored URL in memory: {url[:60]}...", "cyan", "📝")
                except Exception as e:
                    self._logger.log(f"Failed to store URL: {e}", "yellow")
    
    async def _process_message_rlm(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tools_used: List[str] = []
        skills_used: List[str] = []
        files_to_send: List[Dict[str, Any]] = []
        
        memory = await self._ensure_memory()
        conversation_history = self._get_conversation_history(user_id)
        
        # Debug info
        try:
            rcb_entries_full = await memory._db.get_rcb_entries(limit=50)
        except Exception as e:
            rcb_entries_full = []
        
        self._debug("memory_before", {
            "user_id": user_id,
            "rcb_count": len(rcb_entries_full) if isinstance(rcb_entries_full, list) else 0,
            "anchor_cache_size": len(self._anchor_cache),
            "conversation_history_turns": len(conversation_history),
        })
        
        # CRITICAL NEW STEP: Extract and store URLs/facts from message BEFORE processing
        # This ensures URLs are stored with high importance for future recall
        await self._store_extracted_facts(user_id, message, memory)
        
        # Identity detection
        identity_detection = self._identity_detector.detect(message)
        if identity_detection.categories and identity_detection.extracted_facts:
            self._logger.log(
                f"🔍 Identity detected: {', '.join(identity_detection.categories)}",
                "gold", "🧠"
            )
            try:
                facts_summary = []
                for key, value in identity_detection.extracted_facts.items():
                    if value:
                        facts_summary.append(f"{key}={value}")
                
                summary_text = ", ".join(facts_summary) if facts_summary else "identity recorded"
                
                await memory.store_in_ems(
                    content=json.dumps(identity_detection.extracted_facts),
                    chunk_type=MemoryChunkType.FACT,
                    importance=identity_detection.importance_boost,
                    tags=identity_detection.categories + ["identity", f"user_{user_id}"],
                    summary=f"Identity: {summary_text}",
                    load_hints=list(identity_detection.extracted_facts.keys()),
                    source=f"identity:{user_id}",
                )
            except Exception as e:
                self._logger.log(f"Failed to store identity: {e}", "yellow")
        
        # Check for info query
        if self._identity_detector.is_informational_query(message):
            return await self._handle_informational_query(memory, user_id)
        
        is_file_request = self._file_generator and self._file_generator.is_file_request(message)
        
        # CRITICAL: Search and load relevant memories
        loaded_memories = await self._search_and_load_memories(memory, user_id, message)
        
        # Build prompt with loaded content
        soul = self._ensure_soul()
        system_prompt = self._build_system_prompt(
            soul=soul,
            context=context,
            conversation_history=conversation_history,
            loaded_memories=loaded_memories,
            is_file_request=is_file_request,
        )
        
        # CRITICAL DEBUG: Log the actual memory content being sent
        if loaded_memories:
            for i, mem in enumerate(loaded_memories[:2]):
                content_preview = mem.get("content", "")[:500]
                self._logger.log(
                    f"📋 Memory {i+1} in prompt: {mem.get('type')} - {mem.get('title', 'untitled')[:50]}... "
                    f"(content: {len(mem.get('content', ''))} chars, preview: {content_preview[:100]}...)",
                    "cyan"
                )
        
        # Get LLM response
        client = self._ensure_lollms_client()
        response, extracted_tools, extracted_files, raw_llm_response, tool_results = await self._get_llm_response_with_tools(
            client, system_prompt, user_id, message, is_file_request, memory, conversation_history
        )
        
        # FULL raw response logging
        self._debug("llm_raw_response", {
            "user_id": user_id,
            "full_raw_response": raw_llm_response,  # FULL content
            "response_length": len(response),
            "tools_used": len(extracted_tools),
        })
        
        tools_used.extend(extracted_tools)
        files_to_send.extend(extracted_files)
        
        # Store conversation
        new_turn = ConversationTurn(
            user_message=message,
            agent_response=response,
            tools_used=tools_used,
            skills_used=skills_used,
            importance_score=2.0 if identity_detection.categories else 1.0,
        )
        
        if user_id not in self._recent_conversations:
            self._recent_conversations[user_id] = []
        self._recent_conversations[user_id].append(new_turn)
        if len(self._recent_conversations[user_id]) > self._max_recent_history:
            self._recent_conversations[user_id] = self._recent_conversations[user_id][-self._max_recent_history:]
        
        try:
            await memory.store_conversation_turn(
                user_id=user_id,
                user_message=message,
                agent_response=response,
                tools_used=tools_used,
                importance=2.0 if identity_detection.categories else 1.0,
            )
        except Exception as e:
            self._logger.log(f"Failed to store conversation: {e}", "yellow")
        
        await self._deliver_files(user_id, files_to_send)
        
        final_response = self._build_final_response(response, files_to_send)
        self._logger.log_response_sent(user_id, len(final_response), tools_used)
        
        return {
            "success": True,
            "response": final_response,
            "tools_used": tools_used,
            "skills_used": skills_used,
            "files_to_send": files_to_send,
        }
    
    def _build_system_prompt(
        self,
        soul: Optional[Any],
        context: Optional[Dict[str, Any]],
        conversation_history: List[ConversationTurn],
        loaded_memories: List[Dict[str, Any]],
        is_file_request: bool,
    ) -> str:
        """Build system prompt with loaded memories integrated naturally."""
        
        # Start with soul identity
        if soul:
            base_prompt = soul.generate_system_prompt(context)
        else:
            base_prompt = f"You are {self.name}, a helpful AI assistant."
        
        parts = [base_prompt]
        
        # Add conversation history
        if conversation_history:
            parts.append(self._format_history_for_prompt(conversation_history))
        
        # CRITICAL: Add loaded memories with FULL CONTENT - this is the key fix
        if loaded_memories:
            # Sort by type: web_content first, then others
            web_memories = [m for m in loaded_memories if m.get("type") == "web_content"]
            other_memories = [m for m in loaded_memories if m.get("type") != "web_content"]
            sorted_memories = web_memories + other_memories
            
            memory_parts = []
            
            # Add a very clear header
            memory_parts.append("")
            memory_parts.append("=" * 80)
            memory_parts.append("BELOW IS THE FULL CONTENT OF RELEVANT MEMORIES FROM LONG-TERM STORAGE")
            memory_parts.append("=" * 80)
            memory_parts.append("")
            memory_parts.append("⚠️  IMPORTANT: Read this content carefully and use it to answer the user's question.")
            memory_parts.append("⚠️  The user is asking about something you've discussed before - it's RIGHT BELOW.")
            memory_parts.append("⚠️  DO NOT say you don't remember or don't have access - you DO have it below!")
            memory_parts.append("")
            
            for i, mem in enumerate(sorted_memories[:3], 1):  # Top 3 max
                mem_type = mem.get("type", "unknown")
                title = mem.get("title", "Memory")
                content = mem.get("content", "")
                
                memory_parts.append(f"{'='*80}")
                memory_parts.append(f"MEMORY {i}: {title}")
                memory_parts.append(f"Type: {mem_type} | Full length: {mem.get('full_length', len(content))} chars")
                memory_parts.append(f"{'='*80}")
                memory_parts.append("CONTENT:")
                memory_parts.append(content)  # FULL content, no truncation here
                memory_parts.append("")
                memory_parts.append(f"--- END MEMORY {i} ---")
                memory_parts.append("")
            
            memory_parts.append("=" * 80)
            memory_parts.append("END OF LOADED MEMORIES")
            memory_parts.append("=" * 80)
            memory_parts.append("")
            memory_parts.append("INSTRUCTIONS:")
            memory_parts.append("1. Use the information ABOVE to answer the user's question.")
            memory_parts.append("2. DO NOT mention 'memory', 'stored', 'database', 'loaded', or technical terms.")
            memory_parts.append("3. DO NOT say 'According to my memory...' - just answer naturally.")
            memory_parts.append("4. If the user asks about a novel/story above, DESCRIBE IT from the content.")
            memory_parts.append("5. Speak as if you personally remember this, not as if you're reading a file.")
            
            parts.append("\n".join(memory_parts))
        else:
            # No memories found
            parts.append("")
            parts.append("=" * 80)
            parts.append("NO RELEVANT MEMORIES FOUND")
            parts.append("=" * 80)
            parts.append("")
            parts.append("I don't have specific information about this topic in my memory.")
            parts.append("I'll answer based on my general knowledge or ask for clarification.")
        
        # Add tool instructions
        parts.append("")
        parts.append("=" * 60)
        parts.append("AVAILABLE TOOLS")
        parts.append("=" * 60)
        parts.append(self._prompt_builder._build_strict_tool_instructions(self._tools))
        parts.append("")
        
        if is_file_request:
            parts.append("FILE GENERATION: Use filesystem tool to create files when requested.")
        
        # Final reminder
        parts.append("")
        parts.append("FINAL REMINDER: Answer based on the MEMORY CONTENT shown above if available.")
        parts.append("Speak naturally - DO NOT reference memory IDs or technical storage terms.")
        
        if context:
            channel = context.get("channel", "unknown")
            parts.append(f"Channel: {channel}")
        
        final_prompt = "\n".join(parts)
        
        # Log prompt length for debugging
        self._logger.log(f"📝 System prompt length: {len(final_prompt)} characters", "cyan")
        
        return final_prompt
    
    async def _handle_informational_query(
        self,
        memory: RLMMemoryManager,
        user_id: str,
    ) -> Dict[str, Any]:
        stats = await memory.get_stats()
        
        response = (
            f"I can help you with various tasks using my available tools:\n\n"
            f"**Tools:** File operations, HTTP requests, calendar management\n\n"
            f"I maintain memory of our conversations and can recall information "
            f"you've shared with me previously. Just ask me about something "
            f"we've discussed before.\n\n"
            f"What would you like to do?"
        )
        
        return {
            "success": True,
            "response": response,
            "tools_used": [],
            "skills_used": [],
            "files_to_send": [],
        }
    
    async def _get_llm_response_with_tools(
        self,
        client: Optional[LollmsClient],
        system_prompt: str,
        user_id: str,
        message: str,
        is_file_request: bool,
        memory: RLMMemoryManager,
        conversation_history: Optional[List[ConversationTurn]] = None,
        max_iterations: int = 5,
    ) -> tuple[str, List[str], List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        if not client:
            response = await self._try_direct_execution(message, user_id)
            return response[0], response[1], response[2], "[no LLM]", []
        
        tools_used: List[str] = []
        files_generated: List[Dict[str, Any]] = []
        all_tool_results: List[Dict[str, Any]] = []
        last_raw_response = ""
        
        base_prompt = system_prompt
        
        for iteration in range(max_iterations):
            if iteration == 0:
                current_prompt = f"{base_prompt}\n\nUser: {message}\nAssistant:"
            else:
                tool_results_text = self._format_tool_results_for_llm(all_tool_results[-1])
                current_prompt = f"{base_prompt}\n\nUser: {message}\n\n{tool_results_text}\nAssistant:"
            
            self._logger.log_llm_call(len(current_prompt), base_prompt[:500])
            
            llm_response = client.generate_text(
                prompt=current_prompt,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            
            last_raw_response = llm_response
            
            has_tool_calls = "<tool>" in llm_response or "[[TOOL:" in llm_response
            
            if not has_tool_calls:
                clean_response = self._strip_memory_handles(llm_response.strip())
                return clean_response, tools_used, files_generated, last_raw_response, all_tool_results
            
            if self._tool_parser:
                if self._tool_event_callback:
                    await self._tool_event_callback("planning_start", "unknown", {})
                
                result = await self._tool_parser.parse_and_execute(llm_response, user_id, None)
                parsed_response, iteration_tools, iteration_files = result
                
                if not iteration_tools:
                    clean_response = self._strip_memory_handles(llm_response.strip())
                    return clean_response, tools_used, files_generated, last_raw_response, all_tool_results
                
                tools_used.extend(iteration_tools)
                files_generated.extend(iteration_files)
                
                tool_result = {
                    "iteration": iteration,
                    "tools_used": iteration_tools,
                    "files_generated": iteration_files,
                    "response_preview": parsed_response[:500],
                }
                all_tool_results.append(tool_result)
                continue
            else:
                clean_response = self._strip_memory_handles(llm_response.strip())
                return clean_response, tools_used, files_generated, last_raw_response, all_tool_results
        
        self._logger.log("⚠️ Max tool iterations reached", "yellow", "⏳")
        final_response = self._strip_memory_handles(last_raw_response.strip())
        return final_response, tools_used, files_generated, last_raw_response, all_tool_results
    
    def _format_tool_results_for_llm(self, tool_result: Dict[str, Any]) -> str:
        lines = [
            "",
            "TOOL EXECUTION RESULTS:",
            f"Tools: {', '.join(tool_result['tools_used'])}",
        ]
        
        if tool_result.get('files_generated'):
            lines.append(f"Files created: {len(tool_result['files_generated'])}")
        
        if tool_result.get('response_preview'):
            lines.append(f"Result: {tool_result['response_preview']}")
        
        lines.append("Provide your response based on these results.")
        
        return "\n".join(lines)
    
    def _strip_memory_handles(self, text: str) -> str:
        pattern = r'\[\[MEMORY:[^\]]+\]\]'
        return re.sub(pattern, '', text).strip()
    
    async def _try_direct_execution(
        self,
        message: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, List[str], List[Dict[str, Any]]]:
        self._logger.log("⚠️ No LLM client - attempting direct execution", "yellow", "🔧")
        
        if not self._file_generator:
            return (
                f"I'm not connected to a language model backend. Please check your configuration.",
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
            f"I'm not connected to a language model backend.",
            [],
            []
        )
    
    async def _deliver_files(self, user_id: str, files: List[Dict[str, Any]]) -> None:
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
        if not files_to_send:
            return response
        
        file_keywords = ["created", "saved", "generated", "built", "file"]
        has_file_mention = any(kw in response.lower() for kw in file_keywords)
        
        if has_file_mention:
            return response
        
        file_names = [f.get("filename", "unnamed") for f in files_to_send]
        return f"{response}\n\n📁 I've created: {', '.join(file_names)}"
    
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
        return self._environment_info
    
    async def list_available_skills(
        self,
        user_id: str,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
        
        available_tools = set(self._tools.keys()) if hasattr(self, 'tools') else set()
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
        if not self._ensure_skills_initialized():
            return SkillResult(
                success=False, result=None, skill_name=skill_name, skill_version="unknown",
                execution_id="", error="Skills not initialized",
            )
        
        perms = self._user_permissions.get(user_id, self._default_permissions)
        
        self._logger.log(f"🔒 Checking permissions for skill '{skill_name}'", "yellow", "🛡️")
        
        if perms.level.value < PermissionLevel.SKILLS.value:
            return self._skill_permission_denied(skill_name, "lacks SKILLS level")
        
        if perms.denied_skills and skill_name in perms.denied_skills:
            return self._skill_permission_denied(skill_name, "in denylist")
        
        if perms.allowed_skills is not None and skill_name not in perms.allowed_skills:
            return self._skill_permission_denied(skill_name, "not in allowlist")
        
        if self._skill_event_callback:
            await self._skill_event_callback("skill_start", skill_name, {"inputs": list(inputs.keys())})
        
        self._logger.log(f"🎯 Executing skill '{skill_name}'...", "magenta", "📚")
        
        from datetime import datetime
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
        
        self._logger.log_skill_execution(skill_name, skill_result.success)
        
        if self._initialized and self._memory:
            await self._memory.store_in_ems(
                content=json.dumps({
                    "skill": skill_name,
                    "inputs": inputs,
                    "success": skill_result.success,
                    "duration": skill_result.duration_seconds,
                }),
                chunk_type=MemoryChunkType.SKILL_EXECUTION,
                importance=3.0 if skill_result.success else 5.0,
                tags=["skill", skill_name, "execution"],
                summary=f"Skill '{skill_name}': {'success' if skill_result.success else 'failed'}",
                source=f"skill:{user_id}",
            )
        
        if self._skill_event_callback:
            event_type = "skill_complete" if skill_result.success else "skill_error"
            await self._skill_event_callback(
                event_type,
                skill_name,
                {"duration": skill_result.duration_seconds, "steps": len(skill_result.steps_taken)},
            )
        
        return skill_result
    
    def _skill_permission_denied(self, skill_name: str, reason: str) -> SkillResult:
        self._logger.log(f"🚫 Skill '{skill_name}' {reason}", "red", "❌")
        return SkillResult(
            success=False,
            result=None,
            skill_name=skill_name,
            skill_version="unknown",
            execution_id="",
            error=f"Permission denied: {reason}",
        )
    
    async def run_memory_maintenance(self) -> Dict[str, Any]:
        if not self._initialized or not self._memory:
            return {"error": "Memory not initialized"}
        
        self._logger.log("🧠 Running RLM memory maintenance...", "cyan", "💓")
        
        forgetting_result = await self._memory.apply_forgetting_curve()
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
        if self._memory:
            await self._memory.close()
            self._memory = None
