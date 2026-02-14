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
        Multi-step memory retrieval with progressive refinement.
        
        Performs multiple "inner turns" of memory searching:
        1. Initial broad search for context
        2. Follow-up targeted searches based on initial findings
        3. Deep loading of high-relevance content
        4. Cross-referencing related memories
        
        Returns rich memory context for the LLM to reason about.
        """
        loaded_content: List[Dict[str, Any]] = []
        seen_chunks: Set[str] = set()
        search_iterations: int = 0
        max_iterations: int = 3  # Allow multiple search passes
        
        # Start with the original query
        current_queries: List[str] = [query]
        extracted_entities: Set[str] = set()
        
        self._logger.log(f"🔍 Starting multi-step memory retrieval for: '{query}'", "cyan")
        
        while search_iterations < max_iterations and current_queries:
            search_iterations += 1
            self._logger.log(f"🔍 Memory iteration {search_iterations}: {len(current_queries)} queries", "cyan")
            
            iteration_results: List[tuple[Any, float]] = []
            
            # Search with all current queries
            for q in current_queries:
                # Extract key terms for this query
                key_terms = self._extract_key_terms(q)
                
                # Primary search
                results = await memory.search_ems(query=q, limit=15)
                iteration_results.extend(results)
                
                # Secondary search with key terms
                if key_terms:
                    key_query = " ".join(key_terms[:5])  # Top 5 terms
                    key_results = await memory.search_ems(query=key_query, limit=10)
                    iteration_results.extend(key_results)
                    
                    # Extract entities for follow-up searches
                    for term in key_terms:
                        if len(term) > 3 and not term.isdigit():
                            extracted_entities.add(term)
            
            # Sort and deduplicate results
            iteration_results.sort(key=lambda x: -x[1])
            unique_results: List[tuple[Any, float]] = []
            for chunk, score in iteration_results:
                if chunk.chunk_id not in seen_chunks:
                    seen_chunks.add(chunk.chunk_id)
                    unique_results.append((chunk, score))
            
            self._logger.log(f"🔍 Iteration {search_iterations}: {len(unique_results)} unique results", "cyan")
            
            # Determine which chunks to load deeply
            # Load top results and anything scoring above threshold
            load_candidates = [
                (chunk, score) for chunk, score in unique_results[:10]  # Top 10
                if score > 3.0 or chunk.memory_importance > 5.0  # Or high importance
            ]
            
            # Load content from candidates
            new_entities_from_content: Set[str] = set()
            
            for chunk, score in load_candidates:
                try:
                    content = chunk.decompress_content()
                    
                    # Skip very short content
                    if len(content) < 50:
                        continue
                    
                    # Check if we already have similar content
                    is_duplicate = False
                    for existing in loaded_content:
                        # Simple similarity check - avoid near-duplicates
                        if self._content_similarity(content, existing["content"]) > 0.85:
                            is_duplicate = True
                            break
                    
                    if is_duplicate:
                        continue
                    
                    # Extract additional entities from content for next iteration
                    content_entities = self._extract_entities_from_text(content)
                    new_entities_from_content.update(content_entities)
                    
                    # Determine how much to load based on content type and score
                    load_size = self._determine_load_size(chunk, score)
                    
                    # Build rich content object
                    content_obj = {
                        "chunk_id": chunk.chunk_id,
                        "type": chunk.chunk_type.name.lower(),
                        "title": chunk.summary or f"Memory from {chunk.source or 'unknown'}",
                        "content": content[:load_size] if len(content) > load_size else content,
                        "full_length": len(content),
                        "score": score,
                        "importance": chunk.memory_importance,
                        "tags": chunk.tags or [],
                        "source": chunk.source,
                        "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                    }
                    
                    # Add full content to anchor cache for potential follow-up
                    self._anchor_cache[chunk.chunk_id] = content
                    
                    # Load into RCB so it's immediately available
                    await memory.load_from_ems(chunk.chunk_id, add_to_rcb=True)
                    self._loaded_anchors.add(chunk.chunk_id)
                    
                    loaded_content.append(content_obj)
                    
                    self._logger.log(
                        f"📥 Loaded: {chunk.chunk_id[:16]}... ({content_obj['type']}, "
                        f"score={score:.1f}, imp={chunk.memory_importance:.1f}, {len(content)} chars)",
                        "green"
                    )
                    
                except Exception as e:
                    self._logger.log(f"Failed to load {chunk.chunk_id}: {e}", "yellow")
            
            # Prepare next iteration queries
            if search_iterations < max_iterations:
                # Generate follow-up queries from new entities
                next_queries: List[str] = []
                
                # Combine extracted entities in various ways
                entity_list = list(new_entities_from_content - extracted_entities)
                extracted_entities.update(new_entities_from_content)
                
                if len(entity_list) >= 2:
                    # Pairwise combinations for deeper search
                    for i in range(min(3, len(entity_list))):
                        for j in range(i+1, min(5, len(entity_list))):
                            combined = f"{entity_list[i]} {entity_list[j]}"
                            if len(combined) < 100:
                                next_queries.append(combined)
                
                # Add individual high-value entities
                for entity in entity_list[:3]:
                    if len(entity) > 4:
                        next_queries.append(entity)
                
                # Limit and deduplicate next queries
                current_queries = list(set(next_queries))[:5]
                
                if not current_queries:
                    self._logger.log("🔍 No new entities to explore, stopping early", "cyan")
                    break
            else:
                current_queries = []
        
        # Final enrichment: find related memories for loaded content
        if loaded_content:
            await self._enrich_with_related_memories(memory, loaded_content, seen_chunks)
        
        self._logger.log(
            f"✅ Memory retrieval complete: {len(loaded_content)} chunks after {search_iterations} iterations",
            "green" if loaded_content else "yellow"
        )
        
        # Log what's available for debugging
        content_summary = ", ".join([
            f"{c['type']}:{c['chunk_id'][:8]}" for c in loaded_content[:5]
        ])
        self._logger.log(f"📋 Available: {content_summary}", "cyan")
        
        return loaded_content
    
    def _content_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for deduplication."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_entities_from_text(self, text: str) -> Set[str]:
        """Extract potential entities (names, key terms) from text."""
        entities: Set[str] = set()
        
        # Look for quoted phrases
        quoted = re.findall(r'"([^"]{3,50})"', text)
        entities.update(quoted)
        
        # Capitalized phrases (potential names/titles)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z\s]{2,30}[a-z]\b', text)
        entities.update([c.strip() for c in capitalized if len(c.strip()) > 3])
        
        # Technical terms with special chars
        technical = re.findall(r'\b\w+[-_]\w+\b', text)
        entities.update([t for t in technical if len(t) > 4])
        
        # URLs/domains
        urls = re.findall(r'https?://[^\s]+', text)
        for url in urls:
            # Extract domain as entity
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.netloc:
                    entities.add(parsed.netloc)
            except:
                pass
        
        return entities
    
    def _determine_load_size(self, chunk: Any, score: float) -> int:
        """Determine how much content to load based on relevance and type."""
        base_size = 5000
        
        # Higher score = more content
        if score > 8.0:
            base_size = 20000  # Very relevant - load lots
        elif score > 5.0:
            base_size = 15000  # Quite relevant
        elif score > 3.0:
            base_size = 10000  # Moderately relevant
        
        # Web content gets more for creative works
        from lollmsbot.agent.rlm.models import MemoryChunkType
        if chunk.chunk_type == MemoryChunkType.WEB_CONTENT:
            # Check if it looks like a creative work
            summary_lower = (chunk.summary or "").lower()
            if any(term in summary_lower for term in ["novel", "book", "story", "fiction", "creative"]):
                base_size = min(base_size * 2, 50000)  # Up to 50K for creative works
        
        # High importance memories get full load
        if chunk.memory_importance > 8.0:
            base_size = max(base_size, 30000)
        
        return base_size
    
    async def _enrich_with_related_memories(
        self,
        memory: RLMMemoryManager,
        loaded_content: List[Dict[str, Any]],
        seen_chunks: Set[str],
    ) -> None:
        """Find memories related to already-loaded content for cross-referencing."""
        # Extract key terms from loaded content
        all_text = " ".join([c["content"][:500] for c in loaded_content])
        key_terms = self._extract_key_terms(all_text)
        
        if len(key_terms) < 2:
            return
        
        # Search for related content with combined terms
        related_query = " ".join(key_terms[:4])
        
        try:
            related = await memory.search_ems(
                query=related_query,
                limit=8,
            )
            
            added = 0
            for chunk, score in related:
                if chunk.chunk_id in seen_chunks:
                    continue
                
                # Only add if strongly related
                if score < 4.0:
                    continue
                
                try:
                    content = chunk.decompress_content()
                    if len(content) < 100:
                        continue
                    
                    # Add as "related" reference
                    loaded_content.append({
                        "chunk_id": chunk.chunk_id,
                        "type": "related",
                        "title": f"Related: {chunk.summary or 'memory'}"[:50],
                        "content": content[:3000],  # Smaller excerpt for related
                        "full_length": len(content),
                        "score": score,
                        "relation_note": f"Related to main query (score: {score:.1f})",
                    })
                    
                    seen_chunks.add(chunk.chunk_id)
                    added += 1
                    
                    if added >= 2:  # Limit related memories
                        break
                        
                except Exception:
                    continue
            
            if added > 0:
                self._logger.log(f"📎 Added {added} related memories for cross-reference", "cyan")
                
        except Exception as e:
            self._logger.log(f"Related memory enrichment failed: {e}", "yellow")
    
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
        
        # Get memory structure for LLM introspection
        memory_structure = ""
        if self._memory:
            try:
                memory_structure = await self._memory.get_memory_structure_for_prompt(max_length=3000)
            except Exception as e:
                self._logger.log(f"Failed to generate memory structure: {e}", "yellow")
        
        # Build prompt with loaded content
        soul = self._ensure_soul()
        system_prompt = self._build_system_prompt(
            soul=soul,
            context=context,
            conversation_history=conversation_history,
            loaded_memories=loaded_memories,
            is_file_request=is_file_request,
            memory_structure=memory_structure,
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
        memory_structure: str = "",
    ) -> str:
        """Build system prompt with multi-step memory reasoning instructions."""
        
        # Start with soul identity
        if soul:
            base_prompt = soul.generate_system_prompt(context)
        else:
            base_prompt = f"You are {self.name}, a helpful AI assistant with persistent memory."
        
        parts = [base_prompt]
        
        # Add memory structure awareness (NEW)
        if memory_structure:
            parts.append(memory_structure)
            parts.append("")  # Separator
        
        # Add conversation history
        if conversation_history:
            parts.append(self._format_history_for_prompt(conversation_history))
        
        # CRITICAL: Multi-step memory reasoning section
        if loaded_memories:
            # Sort memories by relevance and type
            sorted_memories = self._prioritize_memories(loaded_memories)
            
            memory_parts = []
            
            # Section header
            memory_parts.append("")
            memory_parts.append("=" * 80)
            memory_parts.append("YOUR MEMORY - MULTI-STEP REASONING REQUIRED")
            memory_parts.append("=" * 80)
            memory_parts.append("")
            memory_parts.append("You have retrieved the following information from your long-term memory.")
            memory_parts.append("BEFORE answering, you MUST engage in explicit reasoning about what you found.")
            memory_parts.append("")
            
            # Step 1: What was retrieved
            memory_parts.append("STEP 1 - MEMORY INVENTORY:")
            memory_parts.append("Review each memory and assess its relevance to the user's question.")
            memory_parts.append("")
            
            for i, mem in enumerate(sorted_memories[:5], 1):  # Top 5 memories
                mem_type = mem.get("type", "unknown")
                title = mem.get("title", "Memory")
                content = mem.get("content", "")[:10000]  # Limit individual chunks but show substantial content
                score = mem.get("score", 0)
                importance = mem.get("importance", 1.0)
                
                memory_parts.append(f"{'='*60}")
                memory_parts.append(f"[{i}] {title}")
                memory_parts.append(f"Relevance: {score:.1f}/10 | Importance: {importance:.1f}/10 | Type: {mem_type}")
                
                # Show tags if available
                tags = mem.get("tags", [])
                if tags:
                    memory_parts.append(f"Tags: {', '.join(tags[:5])}")
                
                # Show source if relevant
                source = mem.get("source", "")
                if source and "http" in source:
                    memory_parts.append(f"Source: {source[:100]}")
                
                memory_parts.append(f"{'='*60}")
                memory_parts.append(content)
                memory_parts.append("")
            
            # Step 2: Instructions for inner reasoning
            memory_parts.append("")
            memory_parts.append("STEP 2 - INNER REASONING (Think Before You Answer):")
            memory_parts.append("Before responding, you MUST ask yourself:")
            memory_parts.append("")
            memory_parts.append("Q1: What specific information in the memories above answers the user's question?")
            memory_parts.append("Q2: Is there conflicting information? If so, which source is more reliable?")
            memory_parts.append("Q3: What gaps remain? Do I need to ask clarifying questions?")
            memory_parts.append("Q4: How does this relate to our previous conversations?")
            memory_parts.append("Q5: Is there a specific quote or detail I should reference?")
            memory_parts.append("")
            memory_parts.append("Write your reasoning process (this won't be shown to the user):")
            memory_parts.append("<thinking>")
            memory_parts.append("[Your analysis of the memories and how they relate to the question]")
            memory_parts.append("</thinking>")
            memory_parts.append("")
            
            # Step 3: Answer synthesis instructions
            memory_parts.append("STEP 3 - SYNTHESIZE YOUR ANSWER:")
            memory_parts.append("Based on your reasoning above, now compose your response.")
            memory_parts.append("IMPORTANT RULES:")
            memory_parts.append("- DO NOT mention 'memory', 'database', 'retrieved', or 'stored'")
            memory_parts.append("- DO NOT say 'According to my memory...' - just share what you know")
            memory_parts.append("- Reference specific details from the content above naturally")
            memory_parts.append("- If you found relevant information, use it confidently")
            memory_parts.append("- If memories don't answer the question, say so honestly")
            memory_parts.append("")
            
            # Handle related memories specially
            related = [m for m in sorted_memories if m.get("type") == "related"]
            if related:
                memory_parts.append("ADDITIONAL CONTEXT (Related Memories):")
                for mem in related[:2]:
                    memory_parts.append(f"- {mem.get('title', 'Related memory')}: {mem.get('content', '')[:200]}...")
                memory_parts.append("")
            
            parts.append("\n".join(memory_parts))
        else:
            # No memories found
            parts.append("")
            parts.append("=" * 80)
            parts.append("MEMORY SEARCH RESULTS")
            parts.append("=" * 80)
            parts.append("")
            parts.append("I searched my memory but found no directly relevant information.")
            parts.append("<thinking>")
            parts.append("No memories match this query. I should answer based on:")
            parts.append("- The current conversation context")
            parts.append("- My general knowledge")
            parts.append("- Asking for clarification if needed")
            parts.append("</thinking>")
        
        # Add tool instructions
        parts.append("")
        parts.append("=" * 60)
        parts.append("AVAILABLE TOOLS")
        parts.append("=" * 60)
        parts.append(self._prompt_builder._build_strict_tool_instructions(self._tools))
        parts.append("")
        
        if is_file_request:
            parts.append("FILE GENERATION: Use filesystem tool to create files when requested.")
        
        # Final synthesis instruction
        parts.append("")
        parts.append("=" * 60)
        parts.append("RESPONSE PROTOCOL")
        parts.append("=" * 60)
        parts.append("1. First, analyze your memories in <thinking> tags (not shown to user)")
        parts.append("2. Then provide your actual response")
        parts.append("3. If you found relevant information, use it naturally")
        parts.append("4. Never reveal technical details about your memory system")
        
        if context:
            channel = context.get("channel", "unknown")
            parts.append(f"\nChannel: {channel}")
            
            doc_contexts = context.get("document_contexts")
            if doc_contexts:
                parts.append("")
                parts.append("=" * 60)
                parts.append("DOCUMENT CONTEXTS")
                parts.append("=" * 60)
                for i, doc_ctx in enumerate(doc_contexts, 1):
                    parts.append(f"\n--- Document Context {i} ---\n")
                    parts.append(doc_ctx)
        
        final_prompt = "\n".join(parts)
        self._logger.log(f"📝 System prompt: {len(final_prompt)} chars, {len(loaded_memories)} memories", "cyan")
        
        return final_prompt
    
    def _prioritize_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort memories by relevance, importance, and recency."""
        def score_memory(mem: Dict[str, Any]) -> float:
            base_score = mem.get("score", 0)
            importance = mem.get("importance", 1.0)
            
            # Boost certain types
            mem_type = mem.get("type", "")
            type_boost = 1.0
            if mem_type == "web_content":
                type_boost = 1.2  # Web content often has detailed info
            elif mem_type == "fact":
                type_boost = 1.3  # Explicit facts are important
            elif mem_type == "related":
                type_boost = 0.8  # Related is slightly less direct
            
            return base_score * importance * type_boost
        
        return sorted(memories, key=score_memory, reverse=True)
    
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


# Re-export for convenience
from lollmsbot.agent.integrated_document_agent import IntegratedDocumentAgent

__all__ = [
    "Agent",
    "IntegratedDocumentAgent",
    "AgentState",
    "PermissionLevel",
    "Tool",
    "ToolResult",
    "SkillResult",
    "UserPermissions",
    "ConversationTurn",
    "ToolEventCallback",
    "ToolError",
    "AgentError",
    "MemoryAnchor",
]
