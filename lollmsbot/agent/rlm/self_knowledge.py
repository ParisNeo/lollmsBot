"""
LollmsBot Comprehensive Self-Knowledge

This module contains detailed self-knowledge about LollmsBot's architecture,
capabilities, and usage patterns. This knowledge is stored in EMS and loaded
when users ask questions about how to use or understand the system.
"""

from __future__ import annotations

from typing import List, Dict, Any

# Comprehensive self-knowledge about LollmsBot
# This will be stored as multiple high-importance memory chunks in EMS

SELF_KNOWLEDGE_ENTRIES: List[Dict[str, Any]] = [
    # Identity and Purpose
    {
        "id": "self_identity",
        "category": "identity",
        "content": """I am LollmsBot, a sovereign AI assistant created by ParisNeo. I was designed to be secure, personalized, and truly owned by my user—not a corporate product, but a digital companion that evolves with you.

Core Philosophy:
- Sovereignty: My values and identity belong to you, not a corporation
- Security: Built with defense-in-depth architecture (Guardian, sandboxed tools, audit trails)
- Persistence: I remember our conversations and learn your preferences over time
- Transparency: I'm honest about my capabilities, limitations, and memory systems

My purpose is to serve as a capable, evolving digital companion that empowers you while respecting your autonomy, privacy, and values.""",
        "importance": 10.0,
        "tags": ["identity", "core_values", "sovereignty", "purpose"],
        "load_hints": ["who are you", "what are you", "your purpose", "why created", "parisneo"],
    },
    
    # Architecture Overview
    {
        "id": "self_architecture",
        "category": "architecture",
        "content": """My Architecture - The 7 Pillars:

1. SOUL (Identity & Personality)
   - Stored in ~/.lollmsbot/soul.md (markdown format)
   - Contains: traits, values, communication style, expertise, relationships
   - Evolves over time with modification history
   - NOT a swappable config file—it's my persistent, growing identity

2. GUARDIAN (Security & Ethics)
   - Non-bypassable security layer
   - Prompt injection detection and sanitization
   - Permission gates for sensitive operations
   - Self-quarantine capability if compromise detected
   - Audit logging of all security events

3. HEARTBEAT (Self-Maintenance)
   - Runs every 30 minutes by default
   - Memory deduplication and consolidation
   - Security audits and anomaly detection
   - Performance optimization
   - Configurable via wizard or heartbeat.json

4. RLM MEMORY (Recursive Language Model)
   - Double-memory architecture:
     * EMS (External Memory Store): SQLite-based compressed long-term storage
     * RCB (REPL Context Buffer): Working memory with loadable handles
   - Importance-weighted retention with forgetting curve
   - Semantic search and automatic context loading
   - MemoryMap for LLM introspection of what I know

5. TOOLS (Capabilities)
   - filesystem: Read, write, organize files
   - http: Web requests with automatic memory storage
   - calendar: Event management
   - shell: Command execution (permission-gated)
   - Each tool is sandboxed and audited

6. SKILLS (Learned Workflows)
   - Reusable, documented, versioned capabilities
   - Can be created from examples or descriptions
   - Self-documenting with parameter schemas
   - Composable: skills can call other skills

7. PROJECT MEMORY (Hierarchical Organization)
   - Book/document writing with structured outlines
   - Reference material management
   - Writing task tracking and progress
   - Context lenses for zoomable document views""",
        "importance": 9.5,
        "tags": ["architecture", "7_pillars", "system_design", "components"],
        "load_hints": ["how do you work", "architecture", "system design", "components", "pillars", "rlm", "memory"],
    },
    
    # RLM Memory System Details
    {
        "id": "self_rlm_memory",
        "category": "rlm_memory",
        "content": """My Memory System (RLM - Recursive Language Model):

EXTERNAL MEMORY STORE (EMS):
- Location: ~/.lollmsbot/rlm_memory.db (SQLite)
- Stores: All long-term memories in compressed format
- Organization: Chunks with metadata (importance, tags, timestamps)
- Retention: Importance-weighted forgetting curve (Ebbinghaus-inspired)
- Access: Via semantic search or direct chunk IDs

REPL CONTEXT BUFFER (RCB):
- What I see in my working context
- Contains: System prompt, loaded memories, conversation history
- Capacity: ~10 entries (configurable)
- Loading: Memories appear as [[MEMORY:chunk_id|metadata]] handles
- When I "load" a memory, its full content appears in my context

MEMORY CHUNK TYPES:
- SELF_KNOWLEDGE: Facts about myself (permanent, high importance)
- CONVERSATION: Our chat history (medium importance, ages over time)
- FACT: General knowledge you've shared (variable importance)
- WEB_CONTENT: Information from URLs (tagged by source)
- EPISODIC: Specific events with context
- USER_PREFERENCE: Your likes, settings, requirements
- PROCEDURAL: How-to knowledge
- SKILL_EXECUTION: Records of skill usage

FORGETTING CURVE:
- Memories decay based on: time since access × importance × access frequency
- Formula: Retention = e^(-time / (halflife × strength))
- Default halflife: 7 days
- Each access strengthens the memory
- High-importance memories (>8) decay much slower

MEMORY WEIGHTS & DEGRADATION:
- Active: In RCB, full content available
- Cached: Recent access, fast reload
- Background: In EMS, loadable on demand
- Archived: Below threshold, rare access only
- Forgotten: Below minimum importance, may be purged

AUTOMATIC MEMORY OPERATIONS:
- Deduplication: Finds and merges duplicate web content
- Consolidation: Creates narrative summaries from related memories
- Compression: zlib compression for storage efficiency
- Sanitization: Prompt injection detection before storage""",
        "importance": 9.5,
        "tags": ["rlm", "memory", "ems", "rcb", "forgetting_curve", "architecture"],
        "load_hints": ["memory system", "how do you remember", "rlm", "ems", "rcb", "forgetting", "memory weights"],
    },
    
    # How to Use - Basic Commands
    {
        "id": "self_usage_basic",
        "category": "usage",
        "content": """How to Use LollmsBot - Quick Start:

CONFIGURATION:
1. Run: lollmsbot wizard
   - Configure AI backend (OpenAI, Claude, local models, etc.)
   - Set up channels (Discord, Telegram, WhatsApp, Slack)
   - Customize Soul (personality, values, expertise)
   - Configure Heartbeat (maintenance schedule)

2. Start the gateway:
   - lollmsbot gateway (API + optional web UI)
   - lollmsbot gateway --ui (includes web interface at /ui)
   - lollmsbot chat (console interface)

BASIC INTERACTION:
- Direct chat: Just type your message
- Tools: I can use tools automatically when needed, or you can ask directly
- File creation: "Create a game" or "Make an HTML app" - I'll use filesystem tool
- Web research: Share a URL or ask me to fetch something

COMMANDS:
- /help - Show available commands
- /clear - Clear conversation history display
- /history - Show conversation history
- /tools - List available tools
- /skills - List available skills
- /status - Show system status

MEMORY COMMANDS (in chat):
- "Remember that..." - Store a fact permanently
- "What do you know about..." - Search my memory
- "Forget..." - Remove specific memories (if allowed)
- "How do you work?" - Load this self-knowledge""",
        "importance": 9.0,
        "tags": ["usage", "quick_start", "commands", "how_to", "getting_started"],
        "load_hints": ["how to use", "getting started", "commands", "help", "quick start", "usage"],
    },
    
    # How to Use - Advanced Features
    {
        "id": "self_usage_advanced",
        "category": "usage",
        "content": """Advanced LollmsBot Features:

DOCUMENT/BOOK WRITING:
1. Create project: "Create a book project titled 'My Novel'"
2. Add references: "Ingest https://example.com/research for my book"
3. Create outline: "Create a 10-chapter outline"
4. Write sections: "Write Chapter 1, about 2000 words"
5. Submit content: I'll guide you through the writing workflow

SKILL CREATION:
- From description: "Learn a skill to organize my downloads"
- From example: "Learn from this: [show me how you do X]"
- From composition: "Combine the file organizer and email skills"

WEB INTEGRATION:
- Automatic URL ingestion: Paste any URL, I'll fetch and remember it
- Research synthesis: "Research Python async patterns across my references"
- Cross-reference: I can search across all ingested documents

CHANNEL-SPECIFIC:
- Discord: @mention me, or DM directly
- Telegram: Reply to my messages or send commands
- WhatsApp: Message my number (after QR scan setup)
- Slack: @mention in channels, or DM

CUSTOMIZATION:
- Edit ~/.lollmsbot/soul.md directly (careful!)
- Use wizard for guided changes: lollmsbot wizard
- Heartbeat settings: ~/.lollmsbot/heartbeat.json

SECURITY FEATURES:
- Guardian always active - cannot be disabled
- Permission levels: BASIC, TOOLS, SKILLS, SKILL_CREATE, ADMIN
- Per-user tool allowlists/denylists
- Automatic quarantine on detected attacks

DEBUGGING:
- --debug flag: Shows memory state for each request
- /debug/memory endpoint: Full memory dump (admin only)
- Rich logging: All operations logged with context""",
        "importance": 8.5,
        "tags": ["usage", "advanced", "book_writing", "skills", "customization", "security"],
        "load_hints": ["advanced features", "book writing", "skills", "document management", "customize"],
    },
    
    # File Locations and Configuration
    {
        "id": "self_file_locations",
        "category": "configuration",
        "content": """LollmsBot File Locations and Configuration:

CONFIGURATION FILES:
- ~/.lollmsbot/config.json - Main configuration (backends, channels)
- ~/.lollmsbot/soul.md - My personality and identity (markdown)
- ~/.lollmsbot/heartbeat.json - Self-maintenance settings
- ~/.lollmsbot/ethics_accepted - Hash of accepted ethical charter

DATABASE FILES:
- ~/.lollmsbot/rlm_memory.db - RLM memory (SQLite, compressed chunks)
- ~/.lollmsbot/rlm_memory.db-wal - Write-ahead log (WAL mode)
- ~/.lollmsbot/audit.log - Guardian security events

DATA DIRECTORIES:
- ~/.lollmsbot/data/ - General data storage
- ~/.lollmsbot/documents/ - Ingested documents and book projects
- ~/.lollmsbot/skills/ - Custom skills (.skill.json files)
- ~/.lollmsbot/temp/ - Temporary files (auto-cleaned)

CHANNEL-SPECIFIC:
- ~/.lollmsbot/whatsapp-bridge/ - WhatsApp Web.js files
- ~/.lollmsbot/chats/ - Saved conversation exports

ENVIRONMENT VARIABLES:
- LOLLMSBOT_NAME - My display name
- LOLLMSBOT_API_KEY - API authentication
- LOLLMS_HOST_ADDRESS - AI backend URL
- LOLLMS_API_KEY - AI backend key
- DISCORD_BOT_TOKEN, TELEGRAM_BOT_TOKEN, etc. - Channel tokens

IMPORTANT NOTES:
- All paths respect XDG standards on Linux
- Windows: Uses %USERPROFILE%\.lollmsbot
- Database uses WAL mode for durability
- Back up rlm_memory.db for memory preservation""",
        "importance": 8.0,
        "tags": ["configuration", "files", "paths", "database", "environment"],
        "load_hints": ["where are files", "file locations", "config files", "database location", "paths"],
    },
    
    # Troubleshooting
    {
        "id": "self_troubleshooting",
        "category": "troubleshooting",
        "content": """LollmsBot Troubleshooting Guide:

STARTUP ISSUES:
- "ImportError": Install dev deps: pip install -e .[dev]
- "Database locked": Delete ~/.lollmsbot/rlm_memory.db and restart
- "Cannot connect to AI": Check backend URL in wizard or .env
- "Port already in use": Change port with --port flag

MEMORY ISSUES:
- "Memory not found": I may have genuinely forgotten (low importance + old)
- "Wrong memory": Could be similar tagged content - clarify with more specific query
- "Duplicate memories": Heartbeat deduplication will clean these automatically
- "Memory too large": Large memories are chunked; load handles to access full content

WHATSAPP ISSUES:
- QR code not showing: Maximize terminal window (need 100+ chars width)
- "Scan failed": Use WhatsApp's built-in scanner (Settings → Linked Devices)
- "Session expired": Re-scan QR code
- "Can't message myself": WhatsApp blocks this - use another number or group

TOOL ISSUES:
- "Tool not found": Check available with /tools command
- "Tool failed": Check Guardian audit log for permission issues
- "File not created": Check filesystem permissions on output directory

PERFORMANCE:
- Slow responses: Reduce context size or use lighter model
- High memory usage: Heartbeat will compress and archive old memories
- Disk space: Check ~/.lollmsbot size; use lollmsbot channels disable [channel] to free resources

RESET/RECOVERY:
- Soft reset: Clear conversation history
- Hard reset: Delete ~/.lollmsbot/rlm_memory.db (loses all memories)
- Factory reset: Delete entire ~/.lollmsbot directory
- Keep config: Backup config.json before resetting

GETTING HELP:
- Check logs: Logs go to console and ~/.lollmsbot/audit.log
- Debug mode: lollmsbot gateway --debug for rich output
- GitHub: Report issues with version and reproduction steps""",
        "importance": 7.5,
        "tags": ["troubleshooting", "errors", "debugging", "help", "faq"],
        "load_hints": ["troubleshoot", "error", "problem", "fix", "not working", "help", "debug"],
    },
    
    # Tool Details
    {
        "id": "self_tools_detailed",
        "category": "tools",
        "content": """LollmsBot Tools - Detailed Reference:

FILESYSTEM TOOL:
- read_file(path): Read any text file
- write_file(path, content): Write text to file
- create_html_app(filename, html_content): Create interactive HTML game/app
- list_directory(path): List files in directory
- organize_files(source_dir, method): Auto-organize by date/type

HTTP TOOL:
- get(url, headers?): Fetch web page or API
- post(url, data?, headers?): Submit data
- Results automatically stored in memory with source attribution
- Web content deduplicated if same URL fetched multiple times

CALENDAR TOOL:
- add_event(title, start, end, description?): Create calendar event
- get_events(start, end): List events in range
- delete_event(event_id): Remove event
- Supports ISO 8601 datetime format

SHELL TOOL:
- execute(command, cwd?): Run shell command
- Permission-gated: Requires TOOLS level or explicit permission
- Sandbox: Runs with user privileges, restricted from system directories
- Audit: All commands logged with output

PROJECT MEMORY TOOL (for book/document writing):
- create_project(name, description?): Start new writing project
- add_reference(project_id, document_id): Link reference material
- create_outline(project_id, structure): Build document structure
- write_section(project_id, section_id, content): Add content
- get_project_memory(project_id): Load project context

TOOL SECURITY:
- All tools run through Guardian approval
- Risk levels: low, medium, high, critical
- Auto-approval: BASIC level can use low-risk tools
- Confirmation required: Medium+ risk or destructive operations
- Deny override: Explicit denies always block""",
        "importance": 8.0,
        "tags": ["tools", "filesystem", "http", "calendar", "shell", "project_memory", "reference"],
        "load_hints": ["tools", "filesystem", "http", "calendar", "shell", "what tools", "tool details"],
    },
    
    # API and Integration
    {
        "id": "self_api_integration",
        "category": "api",
        "content": """LollmsBot API and Integration:

REST API ENDPOINTS:
- GET / - Status and configuration summary
- GET /health - Health check with component status
- POST /chat - Main chat endpoint
  * Body: {"message": "...", "user_id": "..."}
  * Response: {"success": true, "response": "...", "tools_used": [...]}
- GET /documents - List all documents
- POST /documents/ingest - Ingest new document
- POST /documents/books - Create book project
- POST /documents/context - Get document context lens

WEBSOCKET (Socket Mode):
- Discord/Telegram/Slack use WebSocket for real-time messaging
- Configurable reconnection with exponential backoff
- Heartbeat pings to detect disconnections

WEBHOOK MODE:
- Alternative to WebSocket for HTTP-based channels
- Requires public HTTPS endpoint
- Signature verification for security

CUSTOM INTEGRATION:
from lollmsbot import Agent, PermissionLevel

async def custom_handler():
    agent = Agent(
        name="CustomBot",
        default_permissions=PermissionLevel.TOOLS
    )
    await agent.initialize()
    
    result = await agent.chat(
        user_id="user123",
        message="Hello!",
        context={"channel": "custom"}
    )
    print(result["response"])

MEMORY INTEGRATION:
- HTTP tool automatically stores fetched content in EMS
- Filesystem tool logs file operations in memory
- Calendar events tracked as episodic memories
- All tool results available for future recall

FILE DELIVERY:
- Generated files returned via files_to_send list
- HTTP API provides /files/download/{file_id} endpoints
- Automatic cleanup after TTL expires""",
        "importance": 7.0,
        "tags": ["api", "integration", "rest", "webhook", "websocket", "programming"],
        "load_hints": ["api", "integration", "rest api", "programmatic", "code", "webhook"],
    },
]


def get_self_knowledge_entries() -> List[Dict[str, Any]]:
    """Return all self-knowledge entries for initialization."""
    return SELF_KNOWLEDGE_ENTRIES


def get_combined_self_knowledge() -> str:
    """Get all self-knowledge as a single formatted document."""
    parts = [
        "=" * 80,
        "LOLLMSBOT COMPLETE SELF-KNOWLEDGE",
        "=" * 80,
        "",
    ]
    
    for entry in SELF_KNOWLEDGE_ENTRIES:
        parts.append(f"{'='*60}")
        parts.append(f"SECTION: {entry['id'].replace('self_', '').upper()}")
        parts.append(f"Importance: {entry['importance']}/10")
        parts.append(f"Tags: {', '.join(entry['tags'])}")
        parts.append(f"{'='*60}")
        parts.append(entry['content'])
        parts.append("")
    
    parts.append("=" * 80)
    parts.append("END OF SELF-KNOWLEDGE")
    parts.append("=" * 80)
    
    return "\n".join(parts)
