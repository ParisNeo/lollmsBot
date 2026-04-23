"""
Agent module for LollmsBot - RLM Memory Edition with Semantic Anchors

Updated to use RLM-compliant memory manager with double-memory structure:
- External Memory Store (EMS): Compressed, chunked long-term storage  
- REPL Context Buffer (RCB): Working memory with loadable handles

CRITICAL: Web content fetched via HTTP is now properly loaded into context
with full text available via semantic anchors that describe memory contents.
"""

from __future__ import annotations
from pathlib import Path
import asyncio
import json
import traceback
import re
import time
from datetime import datetime
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable, AsyncGenerator

from lollmsbot.lollms_client import build_lollms_client
from lollmsbot.config import BotConfig, LollmsSettings
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
from lollmsbot.agent.project_memory import (
    ProjectMemoryManager,
    Project,
    MemorySegment,
    ProjectStatus,
    get_project_memory_manager,
)

# Import RLM Memory package (new modular structure)
from lollms_client import LollmsClient, LollmsDiscussion, LollmsDataManager
from lollms_client.lollms_types import MSG_TYPE
from lollms_client.lollms_discussion import ArtefactType

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
        
        # DATA AND DISCUSSION MANAGEMENT
        self._db_manager: Optional[LollmsDataManager] = None
        self._discussion: Optional[LollmsDiscussion] = None
        self._memory_db_path = memory_db_path or (Path.home() / ".lollmsbot" / "discussion_vault.db")

        # RLM MEMORY SYSTEM (Bridged to Discussion)
        self._memory: Optional[RLMMemoryManager] = None
        self._memory_lock: asyncio.Lock = asyncio.Lock()
        
        # SIMPLIFIED_AGENT_INTEGRATION INTEGRATION
        self._simplified_agent: Optional[Any] = None
        
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
        self._logger.log(f"🌍 Environment: {self._environment_detector.get_summary()}", "cyan", "🖥️")

        # 1. Initialize Data Manager and Discussion
        db_path = f"sqlite:///{self._memory_db_path}"
        self._db_manager = LollmsDataManager(db_path)

        client = self._ensure_lollms_client()

        # Pull context size from LollmsSettings, not BotConfig
        lollms_settings = LollmsSettings.from_env()
        ctx_size = lollms_settings.context_size or 8192

        self._discussion = LollmsDiscussion.create_new(
            lollms_client=client,
            db_manager=self._db_manager,
            autosave=True,
            system_prompt=self._prompt_builder.build_system_prompt(tools={}, soul=self._ensure_soul()),
            max_context_size=ctx_size
        )

        # 2. Bridge legacy RLM memory
        await self._ensure_memory()
        await self._store_environment_knowledge()
        
        # Register project memory tool if available
        try:
            from lollmsbot.tools.project_memory import ProjectMemoryTool
            if self._project_memory:
                project_tool = ProjectMemoryTool()
                project_tool.set_project_memory(self._project_memory)
                await self.register_tool(project_tool)
                self._logger.log("✅ Project memory tool registered", "green", "🔧")
        except Exception as e:
            self._logger.log(f"Could not register project memory tool: {e}", "yellow")
            import traceback
            self._logger.log(f"Traceback: {traceback.format_exc()}", "dim")

        # Register search tools
        try:
            from lollmsbot.tools.search import get_search_tools, SearchManager

            # Get search config from wizard config if available
            search_config = None
            try:
                config_path = Path.home() / ".lollmsbot" / "config.json"
                if config_path.exists():
                    import json
                    wizard_data = json.loads(config_path.read_text())
                    search_config = wizard_data.get("search", {})
            except Exception as e:
                self._logger.log(f"Could not load search config: {e}", "dim")

            search_manager = SearchManager(search_config)
            search_tools = get_search_tools(search_config)
            
            self._logger.log(f"🔍 Found {len(search_tools)} search tool(s) to register", "cyan")

            registered_count = 0
            for tool in search_tools:
                try:
                    # Connect search manager
                    if hasattr(tool, 'set_search_manager'):
                        tool.set_search_manager(search_manager)
                        self._logger.log(f"  🔗 Connected search manager to {tool.name}", "dim")
                    
                    # Connect agent for memory storage
                    tool._agent = self
                    
                    # Register the tool
                    await self.register_tool(tool)
                    registered_count += 1
                    self._logger.log(f"  ✅ Registered: {tool.name}", "green")
                except Exception as tool_e:
                    self._logger.log(f"  ❌ Failed to register {tool.name}: {tool_e}", "yellow")

            status = search_manager.get_status()
            available = [k for k, v in status.items() if v]
            self._logger.log(f"✅ Search tools registered ({registered_count}/{len(search_tools)} tools, {len(available)} providers: {', '.join(available) if available else 'none'})", "green", "🔍")

        except Exception as e:
            self._logger.log(f"Search tools not available: {e}", "yellow")
            import traceback
            self._logger.log(f"Traceback: {traceback.format_exc()}", "dim")
        
        # NEW: Swarm Bridge Registration
        from lollmsbot.tools.swarm import SwarmBridgeTool
        await self.register_tool(SwarmBridgeTool())
        self._logger.log("🐝 Swarm Bridge active. Inter-agent communication enabled.", "cyan", "🌐")

        # Initialize simplified_agant integration
        try:
            from lollmsbot.agent.simplified_agant_integration import SimplifiedAgantIntegration
            self._simplified_agent = SimplifiedAgantIntegration(self)
            await self._simplified_agent.initialize()

            # Register simplified_agant tools
            await self._register_simplified_agent_integration_tools()

            self._logger.log("✅ SimplifiedAgant integration initialized", "green", "🚀")
        except Exception as e:
            self._logger.log(f"SimplifiedAgant integration not available: {e}", "yellow")
            self._simplified_agent = None
        
        self._initialized = True
        self._logger.log(f"✅ Agent {self.name} initialized with project-based memory", "green", "🚀")
    
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
            
            # Initialize project memory manager
            self._project_memory = get_project_memory_manager(self._memory)
            await self._project_memory.initialize()
            self._logger.log("✅ Project Memory Manager initialized", "green", "📁")
        
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
        
        Loads memories up to 60% of configured context size to leave
        room for system prompt, conversation history, and response generation.
        
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
        
        # Get context size from config and calculate safe loading limit
        from lollmsbot.config import LollmsSettings
        settings = LollmsSettings.from_env()
        context_size = settings.context_size or 4096
        max_memory_chars = int(context_size * 4 * 0.6)  # ~4 chars/token, 60% limit
        
        self._logger.log(
            f"🧠 Memory budget: {max_memory_chars:,} chars (~{context_size:,} tokens × 4 × 60%)",
            "cyan"
        )
        
        # Track memory usage
        total_loaded_chars: int = 0
        
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
                # Check memory budget before loading
                if total_loaded_chars >= max_memory_chars:
                    self._logger.log(
                        f"⏹️ Memory budget reached: {total_loaded_chars:,}/{max_memory_chars:,} chars",
                        "yellow"
                    )
                    break
                
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
                    
                    # Respect memory budget
                    remaining_budget = max_memory_chars - total_loaded_chars
                    if load_size > remaining_budget:
                        load_size = remaining_budget
                        self._logger.log(
                            f"📉 Truncated load to fit budget: {load_size:,} chars",
                            "dim"
                        )
                    
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
                    
                    # Track memory usage
                    content_chars = len(content_obj["content"])
                    total_loaded_chars += content_chars
                    
                    # Add full content to anchor cache for potential follow-up
                    self._anchor_cache[chunk.chunk_id] = content
                    
                    # Load into RCB so it's immediately available
                    await memory.load_from_ems(chunk.chunk_id, add_to_rcb=True)
                    self._loaded_anchors.add(chunk.chunk_id)
                    
                    loaded_content.append(content_obj)
                    
                    self._logger.log(
                        f"📥 Loaded: {chunk.chunk_id[:16]}... ({content_obj['type']}, "
                        f"score={score:.1f}, imp={chunk.memory_importance:.1f}, {content_chars:,} chars, "
                        f"budget: {total_loaded_chars:,}/{max_memory_chars:,})",
                        "green"
                    )
                    
                    # Check if we've hit budget after this load
                    if total_loaded_chars >= max_memory_chars:
                        self._logger.log(
                            f"⏹️ Memory budget exhausted: {total_loaded_chars:,}/{max_memory_chars:,} chars",
                            "yellow"
                        )
                        break
                    
                except Exception as e:
                    self._logger.log(f"Failed to load {chunk.chunk_id}: {e}", "yellow")
            
            # Prepare next iteration queries (only if we have budget)
            if search_iterations < max_iterations and total_loaded_chars < max_memory_chars:
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
        
        # Final enrichment: find related memories for loaded content (respect budget)
        if loaded_content and total_loaded_chars < max_memory_chars * 0.9:  # Only if we have 10% headroom
            await self._enrich_with_related_memories(memory, loaded_content, seen_chunks, max_memory_chars - total_loaded_chars)
        
        # Calculate usage percentage
        usage_pct = (total_loaded_chars / max_memory_chars * 100) if max_memory_chars > 0 else 0
        
        self._logger.log(
            f"✅ Memory retrieval complete: {len(loaded_content)} chunks, "
            f"{total_loaded_chars:,}/{max_memory_chars:,} chars ({usage_pct:.1f}%) "
            f"after {search_iterations} iterations",
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
        remaining_budget: int = 0,
    ) -> None:
        """Find memories related to already-loaded content for cross-referencing."""
        if remaining_budget <= 0:
            return
        
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
            used_chars = 0
            
            for chunk, score in related:
                if chunk.chunk_id in seen_chunks:
                    continue
                
                # Check remaining budget
                if used_chars >= remaining_budget:
                    break
                
                # Only add if strongly related
                if score < 4.0:
                    continue
                
                try:
                    content = chunk.decompress_content()
                    if len(content) < 100:
                        continue
                    
                    # Respect budget for related memories
                    excerpt_size = min(3000, remaining_budget - used_chars)
                    if excerpt_size <= 0:
                        break
                    
                    # Add as "related" reference
                    loaded_content.append({
                        "chunk_id": chunk.chunk_id,
                        "type": "related",
                        "title": f"Related: {chunk.summary or 'memory'}"[:50],
                        "content": content[:excerpt_size],
                        "full_length": len(content),
                        "score": score,
                        "relation_note": f"Related to main query (score: {score:.1f})",
                    })
                    
                    used_chars += len(content[:excerpt_size])
                    seen_chunks.add(chunk.chunk_id)
                    added += 1
                    
                    if added >= 2:  # Limit related memories
                        break
                        
                except Exception:
                    continue
            
            if added > 0:
                self._logger.log(
                    f"📎 Added {added} related memories ({used_chars:,} chars) for cross-reference",
                    "cyan"
                )
                
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
                self._logger.log(f"Tool '{tool.name}' already registered, skipping", "yellow", "⚠️")
                return  # Skip instead of raising
            
            if tool.name == "http" and hasattr(tool, 'set_rlm_memory'):
                tool.set_rlm_memory(self._memory)
                self._logger.log(f"🔗 Connected RLM memory to HTTP tool", "cyan", "🔗")
            
            # Special handling for project_memory tool
            if tool.name == "project_memory" and self._project_memory:
                if hasattr(tool, 'set_project_memory'):
                    tool.set_project_memory(self._project_memory)
                    self._logger.log(f"🔗 Connected project memory manager to project_memory tool", "cyan", "🔗")
            
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
        """Synchronous chat wrapper around streaming core."""
        full_response = ""
        tools_used = []
        files_to_send = []

        async for chunk in self.chat_stream(user_id, message, context):
            if chunk["type"] == "text":
                full_response += chunk["content"]
            elif chunk["type"] == "tool_complete":
                tools_used.append(chunk["tool"])
                if chunk.get("files"):
                    files_to_send.extend(chunk["files"])
            elif chunk["type"] == "error":
                return self._error_response(chunk["content"])

        return {
            "success": True,
            "response": full_response,
            "tools_used": tools_used,
            "files_to_send": files_to_send
        }

    async def chat_stream(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming chat core using lollms_discussion orchestration."""
        async with self._state_lock:
            if not self._initialized:
                await self.initialize()

            if self._state == AgentState.QUARANTINED:
                yield {"type": "error", "content": "System quarantined"}
                return

            await self._set_processing_state()
            try:
                # 1. Memory retrieval
                memory = await self._ensure_memory()
                loaded_memories = await self._search_and_load_memories(memory, user_id, message)

                # 2. Update discussion layers with RLM context and current tools
                self._discussion.personality_data_zone = "\n".join([m.get("content", "") for m in loaded_memories])

                # Re-build system prompt to include current tools and soul
                current_system_prompt = self._prompt_builder.build_system_prompt(
                    tools=self._tools, 
                    soul=self._ensure_soul()
                )
                self._discussion.system_prompt = current_system_prompt

                # 3. Map tools to discussion format
                discussion_tools = self._get_discussion_tools()

                # 4. Execution loop via LollmsDiscussion
                loop = asyncio.get_event_loop()
                event_queue = asyncio.Queue()

                def discussion_callback(text: str, msg_type: MSG_TYPE, meta: dict):
                    loop.call_soon_threadsafe(event_queue.put_nowait, (text, msg_type, meta))
                    return True

                # Run discussion.chat in a thread to keep gateway responsive
                gen_task = asyncio.create_task(asyncio.to_thread(
                    self._discussion.chat,
                    user_message=message,
                    tools=discussion_tools,
                    streaming_callback=discussion_callback,
                    enable_repl_tools=True,
                    enable_inline_widgets=True,
                    enable_notes=True,
                    enable_silent_artefact_explanation=True
                ))

                yield {"type": "status", "content": "Connected to Discussion Engine..."}

                while True:
                    try:
                        # Check if background task crashed
                        if gen_task.done() and gen_task.exception():
                            raise gen_task.exception()

                        # Wait for an event with a timeout
                        text, msg_type, meta = await asyncio.wait_for(event_queue.get(), timeout=0.2)

                        # Map discussion types to UI chunks
                        if msg_type == MSG_TYPE.MSG_TYPE_CHUNK:
                            # Handle processing announcements from the unified protocol
                            if meta.get("type") == "processing_open":
                                yield {"type": "tool_start", "content": f"Brain: {meta.get('title') or meta.get('processing_type')}..."}
                            elif not text and meta.get("type") == "artefact_update":
                                yield {"type": "tool_start", "content": f"Building {meta['content']['title']}..."}
                            else:
                                # Normal text stream
                                yield {"type": "text", "content": text}

                        elif msg_type == MSG_TYPE.MSG_TYPE_THOUGHT_CHUNK:
                            # Forward thinking blocks (chain of thought)
                            yield {"type": "text", "content": text}

                        elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_CALL:
                            yield {"type": "tool_start", "content": f"Executing tool: {meta.get('tool')}..."}

                        elif msg_type == MSG_TYPE.MSG_TYPE_TOOL_OUTPUT:
                            yield {"type": "tool_complete", "tool": meta.get("tool"), "files": []}

                        elif msg_type == MSG_TYPE.MSG_TYPE_ERROR:
                            yield {"type": "error", "content": text}

                        elif msg_type == MSG_TYPE.MSG_TYPE_EXCEPTION:
                            yield {"type": "error", "content": f"Lollms Exception: {text}"}

                    except asyncio.TimeoutError:
                        # Ensure the queue is fully drained before breaking, even if task is done
                        if gen_task.done() and event_queue.empty():
                            # Check for background task exceptions
                            try:
                                gen_task.result()
                            except Exception as e:
                                yield {"type": "error", "content": f"Generation failed: {str(e)}"}
                            break
                        continue

            except Exception as e:
                self._logger.log_error("Discussion error", e)
                yield {"type": "error", "content": str(e)}
            finally:
                await self._return_to_idle()

    def _get_discussion_tools(self) -> Dict[str, Any]:
        """Convert Tool objects into LollmsDiscussion-compatible dictionary."""
        disc_tools = {}
        for name, tool in self._tools.items():
            # Extract parameters for discussion schema
            params = []
            props = tool.parameters.get("properties", {})
            req = tool.parameters.get("required", [])
            for p_name, p_val in props.items():
                params.append({
                    "name": p_name,
                    "type": p_val.get("type", "str"),
                    "optional": p_name not in req,
                    "description": p_val.get("description", "")
                })

            disc_tools[name] = {
                "name": name,
                "description": tool.description,
                "parameters": params,
                "callable": tool.execute
            }
        return disc_tools
    
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
    
    async def _get_conversation_history(self, user_id: str) -> List[ConversationTurn]:
        """Get conversation history from both in-memory cache and persistent storage."""
        # Start with in-memory recent conversations
        recent = self._recent_conversations.get(user_id, [])[-self._max_recent_history:]
        
        # If we have memory manager, also search EMS for older conversations
        if self._memory and len(recent) < self._max_recent_history:
            try:
                # Search for conversation chunks from this user
                results = await self._memory.search_ems(
                    query=f"user_{user_id} conversation",
                    chunk_types=[MemoryChunkType.CONVERSATION],
                    limit=self._max_recent_history,
                )
                
                # Convert chunks back to ConversationTurn objects
                ems_turns = []
                for chunk, score in results:
                    try:
                        content = chunk.decompress_content()
                        data = json.loads(content)
                        
                        turn = ConversationTurn(
                            user_message=data.get("user_message", ""),
                            agent_response=data.get("agent_response", ""),
                            tools_used=data.get("tools_used", []),
                            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                            importance_score=chunk.memory_importance,
                        )
                        ems_turns.append(turn)
                    except Exception:
                        continue
                
                # Merge and sort by timestamp
                all_turns = recent + ems_turns
                all_turns.sort(key=lambda t: t.timestamp if hasattr(t, 'timestamp') else datetime.now())
                return all_turns[-self._max_recent_history:]
                
            except Exception as e:
                self._logger.log(f"Failed to load conversation history from EMS: {e}", "yellow")
        
        return recent
    
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
        thinking_content: Optional[str] = None
        
        memory = await self._ensure_memory()
        conversation_history = await self._get_conversation_history(user_id)
        
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
        
        # Check for info query (basic tools inquiry)
        if self._identity_detector.is_informational_query(message):
            # But still load self-knowledge if it's about the system
            if not is_self_knowledge_query:
                return await self._handle_informational_query(memory, user_id)
            # Otherwise continue to load self-knowledge below
        
        is_file_request = self._file_generator and self._file_generator.is_file_request(message)
        
        # CRITICAL: Search and load relevant memories
        loaded_memories = await self._search_and_load_memories(memory, user_id, message)
        
        # Get memory structure for LLM introspection
        memory_structure = ""
        if self._memory:
            try:
                memory_structure = await self._memory.get_memory_structure_for_prompt(max_length=2000)
            except Exception as e:
                self._logger.log(f"Failed to generate memory structure: {e}", "yellow")
        
        # Get project memory structure
        project_structure = ""
        if self._project_memory:
            try:
                project_structure = self._project_memory.format_memory_structure_for_prompt(max_length=2000)
            except Exception as e:
                self._logger.log(f"Failed to generate project structure: {e}", "yellow")
        
        # Build prompt with loaded content
        soul = self._ensure_soul()
        system_prompt = self._build_system_prompt(
            soul=soul,
            context=context,
            conversation_history=conversation_history,
            loaded_memories=loaded_memories,
            is_file_request=is_file_request,
            memory_structure=memory_structure,
            project_structure=project_structure,
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
        response, extracted_tools, extracted_files, raw_llm_response, tool_results, thinking_content = await self._get_llm_response_with_tools(
            client, system_prompt, user_id, message, is_file_request, memory, conversation_history
        )
        
        # FULL raw response logging
        self._debug("llm_raw_response", {
            "user_id": user_id,
            "full_raw_response": raw_llm_response,  # FULL content
            "response_length": len(response),
            "tools_used": len(extracted_tools),
            "has_thinking": bool(thinking_content),
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

        # NEW: Autonomous Skill Compounding
        if len(tools_used) >= 2 and result.get("success"):
            self._logger.log("🧠 Complex success detected. Compounding into a reusable Skill...", "magenta", "📚")
            asyncio.create_task(self._skill_learner.learn_from_description(
                name=f"auto_skill_{int(time.time())}",
                description=f"Automated skill generated from successful task: {message[:50]}",
                example_inputs=[{"message": message}],
                expected_outputs=[{"response": response}]
            ))

        final_response = self._build_final_response(response, files_to_send)
        self._logger.log_response_sent(user_id, len(final_response), tools_used)
        
        result = {
            "success": True,
            "response": final_response,
            "tools_used": tools_used,
            "skills_used": skills_used,
            "files_to_send": files_to_send,
        }
        
        # Include thinking content for CLI display (not for channels)
        if thinking_content:
            result["thinking_content"] = thinking_content
            
        return result
    
    def _build_system_prompt(
        self,
        soul: Optional[Any],
        context: Optional[Dict[str, Any]],
        conversation_history: List[ConversationTurn],
        loaded_memories: List[Dict[str, Any]],
        is_file_request: bool,
        memory_structure: str = "",
        project_structure: str = "",
    ) -> str:
        """Build system prompt with multi-step memory reasoning instructions."""
        
        # Start with soul identity
        if soul:
            base_prompt = soul.generate_system_prompt(context)
        else:
            base_prompt = f"You are {self.name}, a helpful AI assistant with persistent memory."
        
        parts = [base_prompt]
        
        # Add project memory structure FIRST (most important for context)
        if project_structure:
            parts.append(project_structure)
            parts.append("")  # Separator
        
        # Add memory structure awareness
        if memory_structure:
            parts.append(memory_structure)
            parts.append("")  # Separator
        
        # Add conversation history
        if conversation_history:
            parts.append(self._format_history_for_prompt(conversation_history))
        
        # TOOL SELECTION GUIDANCE - CRITICAL for search triggering
        parts.append("")
        parts.append("=" * 60)
        parts.append("🔍 WHEN TO USE SEARCH vs INTERNAL KNOWLEDGE")
        parts.append("=" * 60)
        parts.append("")
        parts.append("You have access to internet search tools. USE THEM when:")
        parts.append("  • User asks about current events, news, or recent developments")
        parts.append("  • Question involves facts that change over time (prices, weather, scores)")
        parts.append("  • User asks 'what is', 'who is', 'latest', 'current', 'today'")
        parts.append("  • You need real-time or recent information")
        parts.append("  • The answer requires data from after your training cutoff")
        parts.append("  • ANY question about time-sensitive topics (sports, politics, tech, etc.)")
        parts.append("")
        parts.append("DO NOT guess or hallucinate facts. If unsure, ALWAYS search first.")
        parts.append("The current timestamp is shown above - use it to judge time-sensitivity.")
        parts.append("")
        parts.append("To search: <tool>quick_search</tool> or <tool>internet_search</tool>")
        parts.append("  <query>your search terms</query>")
        parts.append("")
        
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
        
        # Project memory specific guidance
        parts.append("")
        parts.append("PROJECT MEMORY GUIDANCE:")
        parts.append("- When user says 'create project [name]': use create_project")
        parts.append("- When user says 'open [name] project' or 'open project [name]': use find_or_open_project")
        parts.append("  (This will load existing if found, or create new if not found)")
        parts.append("- When user says 'switch to [name]' or 'load [name]': use load_project with specific ID")
        parts.append("- To see all projects: use list_projects")
        parts.append("")
        
        if is_file_request:
            parts.append("ARTIFACT GENERATION: You have a live visual side-pane.")
            parts.append("When creating games, websites, or visual apps, use the 'filesystem' tool to create '.html' files.")
            parts.append("The user will see your work RENDERED LIVE in the side-pane immediately.")
        
        # Final synthesis instruction
        parts.append("")
        parts.append("=" * 60)
        parts.append("HERMES-LEVEL RESPONSE PROTOCOL")
        parts.append("=" * 60)
        parts.append("1. INTERNAL MONOLOGUE: You MUST start every response with <thought> tags.")
        parts.append("   - Decompose the request.")
        parts.append("   - Identify missing data.")
        parts.append("   - Plan tool execution.")
        parts.append("2. TOOL CALLS: Use the XML format if tools are needed.")
        parts.append("3. REASONING: If tools fail, explain why in <thought> before retrying.")
        parts.append("4. OUTPUT: Provide the final answer ONLY after the internal monologue.")
        
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
    ) -> tuple[str, List[str], List[Dict[str, Any]], str, List[Dict[str, Any]], Optional[str]]:
        if not client:
            response = await self._try_direct_execution(message, user_id)
            return response[0], response[1], response[2], "[no LLM]", [], None
        
        tools_used: List[str] = []
        files_generated: List[Dict[str, Any]] = []
        all_tool_results: List[Dict[str, Any]] = []
        last_raw_response = ""
        thinking_content: Optional[str] = None
        
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
            
            # Extract thinking blocks and clean response
            thinking_blocks = client.extract_thinking_blocks(llm_response)
            if thinking_blocks:
                thinking_content = "\n\n".join(thinking_blocks)
                self._logger.log(f"🧠 Extracted {len(thinking_blocks)} thinking block(s)", "cyan", "💭")
            
            # Remove thinking blocks for processing and final output
            cleaned_llm_response = client.remove_thinking_blocks(llm_response)
            
            # Log what we're working with for debugging
            self._logger.log(f"📄 Cleaned response length: {len(cleaned_llm_response)} chars", "dim")
            
            has_tool_calls = "<tool>" in cleaned_llm_response or "[[TOOL:" in cleaned_llm_response
            
            if not has_tool_calls:
                clean_response = self._strip_memory_handles(cleaned_llm_response.strip())
                # Ensure we have actual content, not just empty
                if not clean_response and thinking_content:
                    # If no clean response but we have thinking, extract conclusion from thinking
                    clean_response = self._extract_conclusion_from_thinking(thinking_content)
                return clean_response, tools_used, files_generated, last_raw_response, all_tool_results, thinking_content
            
            if self._tool_parser:
                if self._tool_event_callback:
                    await self._tool_event_callback("planning_start", "unknown", {})
                
                result = await self._tool_parser.parse_and_execute(cleaned_llm_response, user_id, None)
                parsed_response, iteration_tools, iteration_files = result
                
                # Log tool execution results
                self._logger.log(f"🔧 Tool execution: {len(iteration_tools)} tools, response length {len(parsed_response)}", "purple")
                
                if not iteration_tools:
                    # No tools executed - use the parsed response (which may contain error messages)
                    clean_response = self._strip_memory_handles(parsed_response.strip())
                    if not clean_response:
                        clean_response = self._strip_memory_handles(cleaned_llm_response.strip())
                    return clean_response, tools_used, files_generated, last_raw_response, all_tool_results, thinking_content
                
                tools_used.extend(iteration_tools)
                files_generated.extend(iteration_files)
                
                tool_result = {
                    "iteration": iteration,
                    "tools_used": iteration_tools,
                    "files_generated": iteration_files,
                    "response_preview": parsed_response[:500],
                }
                all_tool_results.append(tool_result)
                
                # If we got a meaningful parsed response with tool results, use it as final
                # This handles the case where tool execution produces the final output
                if parsed_response.strip() and iteration == 0:
                    # Check if this looks like a final response (not just tool indicators)
                    if len(parsed_response) > 50 or "project" in parsed_response.lower():
                        clean_response = self._strip_memory_handles(parsed_response.strip())
                        return clean_response, tools_used, files_generated, last_raw_response, all_tool_results, thinking_content
                
                continue
            else:
                clean_response = self._strip_memory_handles(cleaned_llm_response.strip())
                return clean_response, tools_used, files_generated, last_raw_response, all_tool_results, thinking_content
        
        self._logger.log("⚠️ Max tool iterations reached", "yellow", "⏳")
        final_response = self._strip_memory_handles(client.remove_thinking_blocks(last_raw_response).strip())
        # Ensure we have content
        if not final_response and thinking_content:
            final_response = self._extract_conclusion_from_thinking(thinking_content)
        return final_response, tools_used, files_generated, last_raw_response, all_tool_results, thinking_content
    
    def _extract_conclusion_from_thinking(self, thinking_content: str) -> str:
        """Extract a conclusion or summary from thinking content when no explicit response is given."""
        # Look for common patterns in thinking that indicate the conclusion
        lines = thinking_content.strip().split('\n')
        
        # Find the last substantial line that looks like a conclusion
        for line in reversed(lines):
            line = line.strip()
            # Skip empty lines, headers, and meta-commentary
            if not line or line.startswith('│') or line.startswith('─') or line.startswith('='):
                continue
            if 'need to' in line.lower() or 'should' in line.lower() or 'will' in line.lower():
                # Clean up the line
                conclusion = line.replace('I need to', "I'll").replace('I should', "I'll")
                conclusion = conclusion.replace('I will', "I'll")
                # Remove leading bullet points or markers
                conclusion = re.sub(r'^[•\-\*]\s*', '', conclusion)
                return conclusion
        
        # Fallback: return a generic message with the last few lines
        substantial_lines = [l.strip() for l in lines if len(l.strip()) > 20]
        if substantial_lines:
            return substantial_lines[-1]
        
        return "I'm processing your request. Let me know if you need anything else."
    
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
    
    async def _register_simplified_agent_integration_tools(self) -> None:
        """Register tools for SimplifiedAgant features."""
        if not self._simplified_agent:
            return
        
        # CRM tools
        try:
            from lollmsbot.tools.crm_tools import CRMQueryTool, MeetingPrepTool
            await self.register_tool(CRMQueryTool(self._simplified_agent.crm))
            await self.register_tool(MeetingPrepTool(self._simplified_agent.crm))
            self._logger.log("✅ CRM tools registered", "green", "🔧")
        except Exception as e:
            self._logger.log(f"CRM tools not available: {e}", "dim")
        
        # Knowledge base tools
        try:
            from lollmsbot.tools.knowledge_tools import KnowledgeQueryTool, IngestContentTool
            await self.register_tool(KnowledgeQueryTool(self._simplified_agent.knowledge_base))
            await self.register_tool(IngestContentTool(self._simplified_agent.knowledge_base))
            self._logger.log("✅ Knowledge base tools registered", "green", "🔧")
        except Exception as e:
            self._logger.log(f"Knowledge tools not available: {e}", "dim")
        
        # Task management tools
        try:
            from lollmsbot.tools.task_tools import CreateTaskTool, GetTasksTool
            await self.register_tool(CreateTaskTool(self._simplified_agent.task_manager))
            await self.register_tool(GetTasksTool(self._simplified_agent.task_manager))
            self._logger.log("✅ Task tools registered", "green", "🔧")
        except Exception as e:
            self._logger.log(f"Task tools not available: {e}", "dim")
        
        # YouTube analytics tools
        try:
            from lollmsbot.tools.youtube_tools import YouTubeReportTool
            await self.register_tool(YouTubeReportTool(self._simplified_agent.youtube_analytics))
            self._logger.log("✅ YouTube tools registered", "green", "🔧")
        except Exception as e:
            self._logger.log(f"YouTube tools not available: {e}", "dim")
        
        # Business analysis tools
        try:
            from lollmsbot.tools.business_tools import BusinessReportTool
            await self.register_tool(BusinessReportTool(self._simplified_agent.business_council))
            self._logger.log("✅ Business analysis tools registered", "green", "🔧")
        except Exception as e:
            self._logger.log(f"Business tools not available: {e}", "dim")
    
    async def close(self) -> None:
        if self._memory:
            await self._memory.close()
            self._memory = None


# Re-export for convenience
from lollmsbot.agent.integrated_document_agent import IntegratedDocumentAgent

# simplified_agent-style exports
from lollmsbot.agent.simplified_agant_style import (
    SimplifiedAgantStyle,
    MinimalToolSet,
    CodeExtension,
    SessionBranch,
)
from lollmsbot.agent.simplified_agant_integration import (
    SimplifiedAgantTool,
    integrate_simplified_agant,
)

__all__ = [
    "Agent",
    "IntegratedDocumentAgent",
    "ProjectMemoryManager",
    "Project",
    "MemorySegment",
    "ProjectStatus",
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
    # simplified_agent exports
    "SimplifiedAgantStyle",
    "MinimalToolSet",
    "CodeExtension",
    "SessionBranch",
    "SimplifiedAgantTool",
    "integrate_simplified_agant",
]
