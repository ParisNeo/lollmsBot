#!/usr/bin/env python
"""
lollmsBot Gateway - Central Agent Architecture with File Delivery and Debug Mode
"""
import argparse
import asyncio
import json
import os
import secrets
import hashlib
import hmac
import sys
import time
from typing import Any, Dict, List, Optional, Set
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.json import JSON as RichJSON
from rich.syntax import Syntax
from rich import box

from lollmsbot.config import BotConfig, LollmsSettings, GatewaySettings
from lollmsbot.agent import Agent, PermissionLevel
# Import tools for registration
from lollmsbot.tools.filesystem import FilesystemTool
from lollmsbot.tools.http import HttpTool
from lollmsbot.tools.calendar import CalendarTool
from lollmsbot.tools.shell import ShellTool
# Import document management
from lollmsbot.document_manager import create_document_manager, DocumentManager
from lollmsbot.writing_tools import get_writing_tools

console = Console()
app = FastAPI(title="lollmsBot API")

# Global debug flag
DEBUG_MODE = False

# UI instance (optional)
_ui_instance: Optional[Any] = None
_ui_enabled: bool = False

# HTTP API channel (optional)
_http_api: Optional[Any] = None

# Document manager (global)
_document_manager: Optional[DocumentManager] = None

# ========== SHARED AGENT INSTANCE ==========
_agent: Optional[Agent] = None
_agent_lock: asyncio.Lock = asyncio.Lock()

async def get_agent() -> Agent:
    """Get or create the shared Agent instance with tools registered."""
    global _agent, _document_manager
    if _agent is None:
        async with _agent_lock:
            if _agent is None:
                config = BotConfig.from_env()
                _agent = Agent(
                    config=config,
                    name="LollmsBot",
                    default_permissions=PermissionLevel.BASIC,
                )
                # Initialize async resources with environment detection
                # Determine gateway mode from configuration
                gateway_mode = "standalone"
                host_bindings = [f"{HOST}:{PORT}"]
                
                if DISCORD_TOKEN:
                    gateway_mode = "discord"
                elif TELEGRAM_TOKEN:
                    gateway_mode = "telegram"
                
                await _agent.initialize(
                    gateway_mode=gateway_mode,
                    host_bindings=host_bindings
                )
                
                # Initialize document manager if memory is available
                if _agent._memory:
                    _document_manager = await create_document_manager(
                        memory_manager=_agent._memory,
                    )
                    
                    # Register writing tools
                    writing_tools = get_writing_tools(_document_manager)
                    for tool in writing_tools:
                        try:
                            await _agent.register_tool(tool)
                        except ValueError:
                            pass  # Already registered
                    
                    console.print(f"[green]✅ Document management initialized with {len(writing_tools)} writing tools[/]")
                
                console.print(f"[green]✅ Agent initialized: {_agent.name}[/]")
                console.print(f"[dim]   Environment: {_agent._environment_detector.get_summary() if _agent.environment_info else 'unknown'}[/]")
    return _agent

# ========== SHARED LOLLMS CLIENT ==========
_lollms_client: Optional[Any] = None

def get_lollms_client():
    """Get or create shared LoLLMS client."""
    global _lollms_client
    if _lollms_client is None:
        try:
            from lollmsbot.lollms_client import build_lollms_client
            _lollms_client = build_lollms_client()
            console.print("[green]✅ LoLLMS client initialized[/]")
        except Exception as e:
            console.print(f"[yellow]⚠️  LoLLMS client unavailable: {e}[/]")
            _lollms_client = None
    return _lollms_client

# ========== CONFIGURATION ==========

def _load_wizard_config() -> Dict[str, Any]:
    """Load config from wizard's config.json if it exists."""
    wizard_path = Path.home() / ".lollmsbot" / "config.json"
    if wizard_path.exists():
        try:
            return json.loads(wizard_path.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {}

_WIZARD_CONFIG = _load_wizard_config()

def _get_config(service: str, key: str, env_name: str, default: Any = None) -> Any:
    """Get config value: wizard config > env var > default."""
    if service in _WIZARD_CONFIG and key in _WIZARD_CONFIG[service]:
        return _WIZARD_CONFIG[service][key]
    return os.getenv(env_name, default)

# Security settings
DEFAULT_HOST = "127.0.0.1"
HOST = _get_config("lollmsbot", "host", "LOLLMSBOT_HOST", DEFAULT_HOST)
PORT = int(_get_config("lollmsbot", "port", "LOLLMSBOT_PORT", "8800"))
API_KEY = _get_config("lollmsbot", "api_key", "LOLLMSBOT_API_KEY", None)

if HOST not in ("127.0.0.1", "localhost", "::1") and not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    console.print(f"[bold yellow]⚠️  Auto-generated API key: {API_KEY}[/]")

_security = HTTPBearer(auto_error=False)

# Channel tokens
DISCORD_TOKEN = _get_config("discord", "bot_token", "DISCORD_BOT_TOKEN", None)
DISCORD_ALLOWED_USERS = _get_config("discord", "allowed_users", "DISCORD_ALLOWED_USERS", None)
DISCORD_ALLOWED_GUILDS = _get_config("discord", "allowed_guilds", "DISCORD_ALLOWED_GUILDS", None)
DISCORD_BLOCKED_USERS = _get_config("discord", "blocked_users", "DISCORD_BLOCKED_USERS", None)
DISCORD_REQUIRE_MENTION_GUILD = _get_config("discord", "require_mention_guild", "DISCORD_REQUIRE_MENTION_GUILD", "true")
DISCORD_REQUIRE_MENTION_DM = _get_config("discord", "require_mention_dm", "DISCORD_REQUIRE_MENTION_DM", "false")
TELEGRAM_TOKEN = _get_config("telegram", "bot_token", "TELEGRAM_BOT_TOKEN", None)

_active_channels: Dict[str, Any] = {}
_channel_tasks: List[asyncio.Task] = []

# ========== SECURITY ==========

def _verify_api_key(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """Verify API key."""
    if API_KEY is None:
        return True
    if credentials is None:
        return False
    provided = credentials.credentials.encode('utf-8')
    expected = API_KEY.encode('utf-8')
    return hmac.compare_digest(provided, expected)

async def require_auth(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)):
    """Require authentication for external access."""
    client_host = request.client.host if request.client else "unknown"
    
    # Always allow localhost
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return
    
    if API_KEY is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="External access not permitted. Gateway is in local-only mode.",
        )
    
    if not _verify_api_key(credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ========== MODELS ==========

class Health(BaseModel):
    status: str = "ok"
    url: str = f"http://{HOST}:{PORT}"

class ChatReq(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"

class ChatResp(BaseModel):
    success: bool
    response: str
    error: Optional[str] = None
    tools_used: List[str] = []
    files_generated: int = 0
    file_downloads: List[Dict[str, Any]] = []

class PermissionReq(BaseModel):
    admin_user_id: str
    target_user_id: str
    level: str  # "NONE", "BASIC", "TOOLS", "ADMIN"
    allowed_tools: Optional[List[str]] = None
    denied_tools: Optional[List[str]] = None

# Document management models
class IngestDocumentReq(BaseModel):
    source_type: str  # "url", "text"
    source: str
    title: Optional[str] = None
    document_id: Optional[str] = None

class IngestDocumentResp(BaseModel):
    success: bool
    document_id: Optional[str] = None
    title: Optional[str] = None
    blocks: int = 0
    words_estimate: int = 0
    error: Optional[str] = None

class CreateBookReq(BaseModel):
    title: str
    author: Optional[str] = None
    description: Optional[str] = None
    references: Optional[List[str]] = None

class CreateBookResp(BaseModel):
    success: bool
    project_id: Optional[str] = None
    title: Optional[str] = None
    references_connected: int = 0
    error: Optional[str] = None

class DocumentContextReq(BaseModel):
    document_id: str
    focus: Optional[str] = None
    detail_level: Optional[str] = "summary"
    include_references: Optional[bool] = True

class DocumentContextResp(BaseModel):
    success: bool
    context_text: Optional[str] = None
    tokens_used: int = 0
    navigation_handles: Optional[List[str]] = None
    error: Optional[str] = None

# ========== CORS ==========

_cors_origins = ["http://localhost", "http://127.0.0.1"]
if HOST not in ("127.0.0.1", "localhost", "::1"):
    _cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ========== DEBUG DISPLAY FUNCTIONS ==========

def _display_debug_memory(agent: Agent, user_id: str) -> None:
    """Display rich debug information about agent memory state."""
    if not DEBUG_MODE:
        return
    
    console.print()
    console.print(Panel(
        "[bold yellow]🔍 DEBUG: Agent Memory State[/bold yellow]",
        border_style="yellow",
        box=box.DOUBLE
    ))
    
    # RLM Memory Stats
    if agent._memory:
        try:
            # Run async stats retrieval
            import asyncio
            stats = asyncio.get_event_loop().run_until_complete(agent._memory.get_stats())
            
            stats_table = Table(title="RLM Memory Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            for key, value in stats.items():
                stats_table.add_row(key, str(value))
            
            console.print(stats_table)
            
            # Show RCB entries
            try:
                rcb_entries = asyncio.get_event_loop().run_until_complete(agent._memory._db.get_rcb_entries(limit=10))
                if rcb_entries:
                    rcb_table = Table(title="REPL Context Buffer (RCB) - Working Memory")
                    rcb_table.add_column("Order", style="dim")
                    rcb_table.add_column("Type", style="cyan")
                    rcb_table.add_column("Content Preview", style="green")
                    rcb_table.add_column("Chunk ID", style="yellow")
                    
                    for entry in rcb_entries:
                        content_preview = entry.get("content", "")[:50] + "..." if len(entry.get("content", "")) > 50 else entry.get("content", "")
                        rcb_table.add_row(
                            str(entry.get("display_order", "?")),
                            entry.get("entry_type", "unknown"),
                            content_preview,
                            entry.get("chunk_id", "none")[:12] if entry.get("chunk_id") else "none"
                        )
                    
                    console.print(rcb_table)
            except Exception as e:
                console.print(f"[dim]Could not retrieve RCB entries: {e}[/]")
            
            # Show anchor cache
            if hasattr(agent, '_anchor_cache') and agent._anchor_cache:
                anchor_table = Table(title="Memory Anchor Cache (Full Content Available)")
                anchor_table.add_column("Chunk ID", style="cyan")
                anchor_table.add_column("Content Length", style="green")
                anchor_table.add_column("In RCB", style="yellow")
                
                for chunk_id, content in list(agent._anchor_cache.items())[:10]:
                    in_rcb = "✅" if chunk_id in agent._loaded_anchors else "❌"
                    anchor_table.add_row(chunk_id[:20], str(len(content)), in_rcb)
                
                console.print(anchor_table)
                
        except Exception as e:
            console.print(f"[red]Error retrieving memory stats: {e}[/]")
    
    # Conversation History
    if hasattr(agent, '_memory') and agent._memory:
        try:
            user_history = agent._memory.get_user_history(user_id)
            if user_history:
                history_table = Table(title=f"Conversation History for {user_id[:30]}...")
                history_table.add_column("Turn", style="dim")
                history_table.add_column("User Message", style="cyan", max_width=40)
                history_table.add_column("Tools Used", style="yellow")
                
                for i, turn in enumerate(user_history[-5:], 1):
                    msg_preview = turn.user_message[:40] + "..." if len(turn.user_message) > 40 else turn.user_message
                    tools = ", ".join(turn.tools_used) if turn.tools_used else "none"
                    history_table.add_row(str(i), msg_preview, tools)
                
                console.print(history_table)
        except Exception as e:
            console.print(f"[dim]Could not retrieve conversation history: {e}[/]")
    
    # Important Facts
    if hasattr(agent, '_memory') and agent._memory:
        try:
            facts = agent._memory.get_important_facts()
            if facts:
                facts_panel = Panel(
                    RichJSON.from_data(facts),
                    title="Important Facts Stored",
                    border_style="green"
                )
                console.print(facts_panel)
        except Exception as e:
            console.print(f"[dim]Could not retrieve important facts: {e}[/]")
    
    console.print(Panel(
        "[dim]End of debug memory display[/dim]",
        border_style="yellow"
    ))
    console.print()

def _display_debug_response(result: Dict[str, Any], user_id: str, message: str) -> None:
    """Display rich debug information about a chat response."""
    if not DEBUG_MODE:
        return
    
    console.print()
    console.print(Panel(
        f"[bold blue]🔍 DEBUG: Response Details for {user_id[:30]}...[/bold blue]",
        border_style="blue",
        box=box.DOUBLE
    ))
    
    # Request info
    req_table = Table(title="Request")
    req_table.add_column("Field", style="cyan")
    req_table.add_column("Value", style="green")
    req_table.add_row("User ID", user_id[:50])
    req_table.add_row("Message", message[:60] + "..." if len(message) > 60 else message)
    console.print(req_table)
    
    # Response info
    resp_table = Table(title="Response")
    resp_table.add_column("Field", style="cyan")
    resp_table.add_column("Value", style="green")
    resp_table.add_row("Success", "✅ Yes" if result.get("success") else "❌ No")
    resp_table.add_row("Response Length", str(len(result.get("response", ""))))
    resp_table.add_row("Tools Used", ", ".join(result.get("tools_used", [])) or "none")
    resp_table.add_row("Skills Used", ", ".join(result.get("skills_used", [])) or "none")
    resp_table.add_row("Files Generated", str(len(result.get("files_to_send", []))))
    if result.get("error"):
        resp_table.add_row("Error", f"[red]{result.get('error')[:100]}[/]")
    console.print(resp_table)
    
    # Raw response preview
    if result.get("response"):
        response_preview = result["response"][:500] + "..." if len(result["response"]) > 500 else result["response"]
        console.print(Panel(
            Syntax(response_preview, "markdown", theme="monokai", word_wrap=True),
            title="Response Preview",
            border_style="green"
        ))
    
    console.print(Panel(
        "[dim]End of debug response display[/dim]",
        border_style="blue"
    ))
    console.print()

# ========== ROUTES ==========

@app.get("/")
async def root():
    agent = await get_agent()
    lollms_ok = get_lollms_client() is not None
    
    # Check channels
    channels_status = {
        "discord": "enabled" if DISCORD_TOKEN else "disabled",
        "telegram": "enabled" if TELEGRAM_TOKEN else "disabled",
    }
    if "discord" in _active_channels:
        channels_status["discord"] = "active"
    if "telegram" in _active_channels:
        channels_status["telegram"] = "active"
    
    # Environment info
    env_info = {}
    if agent.environment_info:
        env = agent.environment_info
        env_info = {
            "platform": env.os_system,
            "release": env.os_release,
            "python": env.python_version,
            "container": "docker" if env.in_docker else "wsl" if env.in_wsl else "none",
            "virtualenv": env.in_virtualenv,
        }
    
    # Document management status
    doc_status = "available" if _document_manager else "unavailable"
    if _document_manager:
        docs = _document_manager.list_documents()
        doc_status = f"{len(docs)} documents indexed"
    
    response = {
        "api": f"http://{HOST}:{PORT}",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
        "agent": {
            "name": agent.name,
            "state": agent.state.name,
            "tools": list(agent.tools.keys()),
        },
        "lollms": {
            "connected": lollms_ok,
            "host": LollmsSettings.from_env().host_address,
        },
        "security": {
            "host": HOST,
            "local_only": HOST in ("127.0.0.1", "localhost", "::1"),
            "auth_required": API_KEY is not None,
        },
        "channels": channels_status,
        "features": {
            "file_delivery": True,
            "web_ui": _ui_enabled,
            "environment_awareness": True,
            "debug_mode": DEBUG_MODE,
            "document_management": _document_manager is not None,
        },
        "documents": {
            "status": doc_status,
            "endpoints": {
                "ingest": "/documents/ingest",
                "create_book": "/documents/books",
                "context": "/documents/context",
                "list": "/documents",
            } if _document_manager else None,
        },
        "environment": env_info,
    }
    
    # Add debug endpoints if in debug mode
    if DEBUG_MODE:
        response["debug_endpoints"] = {
            "memory": "/debug/memory",
            "stats": "/debug/stats",
        }
    
    return response

@app.get("/health", response_model=Health)
async def health():
    """Health check endpoint."""
    agent = await get_agent()
    lollms_client = get_lollms_client()
    lollms_ok = lollms_client is not None
    
    discord_status = "active" if "discord" in _active_channels else "disabled"
    telegram_status = "active" if "telegram" in _active_channels else "disabled"
    
    # Count pending files across channels
    pending_files = 0
    if _http_api:
        pending_files += len(_http_api._pending_files) if hasattr(_http_api, '_pending_files') else 0
    
    # Environment summary
    env_summary = "unknown"
    if agent.environment_info:
        env = agent.environment_info
        env_summary = f"{env.os_system} {env.os_release}, Python {env.python_version}"
        if env.in_docker:
            env_summary += " (Docker)"
        elif env.in_wsl:
            env_summary += " (WSL)"
    
    # Document status
    doc_count = 0
    if _document_manager:
        doc_count = len(_document_manager.list_documents())
    
    response = {
        "status": "ok",
        "url": f"http://{HOST}:{PORT}",
        "discord": discord_status,
        "telegram": telegram_status,
        "lollms": {
            "connected": lollms_ok,
            "host": LollmsSettings.from_env().host_address,
        },
        "agent": agent.state.name,
        "tools": list(agent.tools.keys()),
        "security": {
            "mode": "local" if HOST in ("127.0.0.1", "localhost", "::1") else "network",
            "auth_enabled": API_KEY is not None,
        },
        "features": {
            "pending_files": pending_files,
            "file_delivery_enabled": True,
            "environment_awareness": True,
            "debug_mode": DEBUG_MODE,
        },
        "documents": {
            "indexed": doc_count,
            "writing_tools": sum(1 for t in agent.tools.keys() if t in [
                "ingest_document", "create_book_project", "create_outline",
                "get_document_context", "write_section", "submit_written_content",
                "search_references", "get_writing_progress"
            ]) if hasattr(agent, 'tools') else 0,
        },
        "environment": env_summary,
    }
    
    return response

@app.post("/chat", response_model=ChatResp, dependencies=[Depends(require_auth)])
async def chat(req: ChatReq):
    """Process a chat message through the Agent with file delivery support."""
    agent = await get_agent()
    
    # Debug: Display memory before processing
    if DEBUG_MODE:
        _display_debug_memory(agent, req.user_id or "anonymous")
    
    result = await agent.chat(
        user_id=req.user_id or "anonymous",
        message=req.message,
        context={"channel": "gateway_http", "source": "api"},
    )
    
    # Debug: Display response details
    if DEBUG_MODE:
        _display_debug_response(result, req.user_id or "anonymous", req.message)
    
    # Build file download info if files were generated
    file_downloads = []
    files_generated = result.get("files_to_send", [])
    
    # If we have an HTTP API channel, it may have registered the files
    if _http_api and hasattr(_http_api, '_pending_files'):
        for file_info in files_generated:
            file_path = file_info.get("path")
            # Find matching registered file
            for file_id, delivery in _http_api._pending_files.items():
                if delivery.original_path == file_path:
                    file_downloads.append({
                        "filename": delivery.filename,
                        "download_url": f"/files/download/{file_id}",
                        "description": delivery.description,
                        "expires_in_seconds": int(_http_api._file_ttl_seconds - (time.time() - delivery.created_at)),
                    })
                    break
    
    # Also check if files can be served directly
    if not file_downloads and files_generated:
        # Create direct download URLs for known output directory
        for file_info in files_generated:
            file_path = file_info.get("path", "")
            filename = file_info.get("filename") or Path(file_path).name
            # Add basic file info even without HTTP API channel
            file_downloads.append({
                "filename": filename,
                "path": file_path,
                "description": file_info.get("description", "Generated file"),
                "note": "File saved to server filesystem, download via direct access if enabled",
            })
    
    return ChatResp(
        success=result.get("success", False),
        response=result.get("response", ""),
        error=result.get("error"),
        tools_used=result.get("tools_used", []),
        files_generated=len(files_generated),
        file_downloads=file_downloads,
    )

# ========== DOCUMENT MANAGEMENT ENDPOINTS ==========

@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    if not _document_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    return {
        "documents": _document_manager.list_documents(),
    }

@app.post("/documents/ingest", response_model=IngestDocumentResp)
async def ingest_document(req: IngestDocumentReq):
    """Ingest a document for use as reference material."""
    if not _document_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    try:
        if req.source_type == "url":
            doc_id = await _document_manager.ingest_webpage(
                url=req.source,
                title=req.title,
                document_id=req.document_id,
            )
        elif req.source_type == "text":
            doc_id = await _document_manager.ingest_text(
                content=req.source,
                title=req.title or "Untitled Document",
                document_id=req.document_id,
            )
        else:
            return IngestDocumentResp(
                success=False,
                error=f"Unsupported source_type: {req.source_type}",
            )
        
        # Get stats
        index = _document_manager._documents.get(doc_id)
        stats = index.get_statistics() if index else {}
        
        return IngestDocumentResp(
            success=True,
            document_id=doc_id,
            title=req.title or "Untitled Document",
            blocks=stats.get("total_blocks", 0),
            words_estimate=stats.get("total_word_estimate", 0),
        )
        
    except Exception as e:
        return IngestDocumentResp(
            success=False,
            error=str(e),
        )

@app.post("/documents/books", response_model=CreateBookResp)
async def create_book(req: CreateBookReq):
    """Create a new book writing project."""
    if not _document_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    try:
        project_id = await _document_manager.create_book_project(
            title=req.title,
            author=req.author,
            description=req.description,
            references=req.references or [],
        )
        
        return CreateBookResp(
            success=True,
            project_id=project_id,
            title=req.title,
            references_connected=len(req.references or []),
        )
        
    except Exception as e:
        return CreateBookResp(
            success=False,
            error=str(e),
        )

@app.post("/documents/context", response_model=DocumentContextResp)
async def get_document_context(req: DocumentContextReq):
    """Get hierarchical context for a document."""
    if not _document_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    try:
        # Map detail_level to budget
        budget_map = {
            "overview": "small",
            "summary": "medium",
            "detailed": "large",
            "full": "full",
        }
        budget = budget_map.get(req.detail_level, "medium")
        
        lens = await _document_manager.get_context_lens(
            document_id=req.document_id,
            focus_block_id=req.focus if req.focus else None,
            budget=budget,
            include_surroundings=req.include_references if req.include_references is not None else True,
        )
        
        return DocumentContextResp(
            success=True,
            context_text=lens.to_prompt_fragment(),
            tokens_used=lens.current_tokens,
            navigation_handles=lens.navigation_handles,
        )
        
    except Exception as e:
        return DocumentContextResp(
            success=False,
            error=str(e),
        )

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details."""
    if not _document_manager:
        raise HTTPException(status_code=503, detail="Document management not available")
    
    index = _document_manager._documents.get(document_id)
    if not index:
        raise HTTPException(status_code=404, detail="Document not found")
    
    root_id = _document_manager._document_roots.get(document_id)
    root = index.get_block(root_id) if root_id else None
    
    stats = index.get_statistics()
    
    # Get structure
    structure = []
    for block_type in ["CHAPTER", "SECTION"]:
        for bid in index._sequence.get(getattr(__import__('lollmsbot.document_manager', fromlist=['BlockType']).BlockType, block_type, None) or [], []):
            block = index.get_block(bid)
            if block:
                structure.append({
                    "id": bid,
                    "type": block_type.lower(),
                    "title": block.title,
                    "status": block.status,
                    "word_count": block.word_count,
                })
    
    return {
        "document_id": document_id,
        "title": root.title if root else "Unknown",
        "type": "book" if document_id.startswith("book_") else "reference",
        "statistics": stats,
        "structure": structure,
    }

@app.post("/admin/permission", dependencies=[Depends(require_auth)])
async def set_permission(req: PermissionReq):
    """Admin endpoint to set user permissions."""
    agent = await get_agent()
    
    # This would need to be implemented in the Agent class
    # For now, return a placeholder
    return {
        "success": False,
        "error": "Admin permission management not yet implemented in this version",
    }

# ========== DEBUG ENDPOINTS ==========

if DEBUG_MODE:
    @app.get("/debug/memory")
    async def debug_memory():
        """Debug endpoint to view agent memory state."""
        agent = await get_agent()
        
        memory_data = {}
        
        if agent._memory:
            try:
                stats = await agent._memory.get_stats()
                memory_data["rlm_stats"] = stats
                
                rcb_entries = await agent._memory._db.get_rcb_entries(limit=20)
                memory_data["rcb_entries"] = rcb_entries
                
                self_knowledge = await agent._memory._db.get_self_knowledge()
                memory_data["self_knowledge"] = self_knowledge
                
            except Exception as e:
                memory_data["error"] = str(e)
        
        if hasattr(agent, '_anchor_cache'):
            memory_data["anchor_cache"] = {
                chunk_id: len(content) 
                for chunk_id, content in agent._anchor_cache.items()
            }
        
        # Document data
        if _document_manager:
            memory_data["documents"] = _document_manager.list_documents()
        
        return memory_data
    
    @app.get("/debug/stats")
    async def debug_stats():
        """Debug endpoint to view detailed system stats."""
        agent = await get_agent()
        
        return {
            "agent_state": agent.state.name,
            "tools_registered": list(agent.tools.keys()),
            "memory_initialized": agent._memory is not None,
            "soul_initialized": agent._soul_initialized,
            "lollms_client_initialized": agent._lollms_client_initialized,
            "document_manager_initialized": _document_manager is not None,
        }

# ========== FILE DOWNLOAD ENDPOINTS ==========

@app.get("/files/download/{file_id}")
async def download_file(file_id: str):
    """Download a generated file by ID (proxies to HTTP API channel if available)."""
    if _http_api and hasattr(_http_api, '_pending_files'):
        if file_id in _http_api._pending_files:
            delivery = _http_api._pending_files[file_id]
            from fastapi.responses import FileResponse
            return FileResponse(
                path=delivery.original_path,
                filename=delivery.filename,
                media_type=delivery.content_type or "application/octet-stream",
            )
    
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/files/list")
async def list_files():
    """List pending files for download."""
    files = []
    if _http_api and hasattr(_http_api, '_pending_files'):
        for file_id, delivery in _http_api._pending_files.items():
            files.append({
                "file_id": file_id,
                "filename": delivery.filename,
                "description": delivery.description,
                "expires_in_seconds": int(_http_api._file_ttl_seconds - (time.time() - delivery.created_at)),
            })
    
    return {"files": files, "count": len(files)}

# ========== UI ENABLE ==========

def enable_ui(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Enable the web UI."""
    global _ui_enabled, _ui_instance
    
    try:
        from lollmsbot.ui.app import WebUI
        # Need async initialization, so just mark for lifespan to handle
        _ui_enabled = True
        console.print(f"[green]✅ Web UI will be mounted at /ui[/]")
        
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not enable UI: {e}[/]")
        import traceback
        traceback.print_exc()

# ========== HTTP API ENABLE ==========

def enable_http_api(host: str = "0.0.0.0", port: int = 8800) -> None:
    """Enable standalone HTTP API channel (for advanced file delivery)."""
    global _http_api
    
    # The main gateway already provides HTTP API, but this enables the full
    # HttpApiChannel with advanced file delivery if needed separately
    console.print(f"[dim]HTTP API available at main gateway endpoints[/]")

# ========== LIFESPAN ==========

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Show security info
    is_local = HOST in ("127.0.0.1", "localhost", "::1")
    
    if is_local:
        console.print(f"[bold green]🔒 SECURITY: Local-only mode[/]")
    else:
        console.print(f"[bold red]🌐 SECURITY: Public interface {HOST}[/]")
        if API_KEY:
            console.print(f"[bold green]🔐 API key authentication ENABLED[/]")
    
    # Show debug mode status
    if DEBUG_MODE:
        console.print(Panel(
            "[bold yellow]🐛 DEBUG MODE ENABLED[/bold yellow]\n\n"
            "• Memory state will be displayed for each request\n"
            "• Response details will be logged with rich formatting\n"
            "• Debug endpoints available at /debug/*\n"
            "• Performance impact: moderate",
            border_style="yellow",
            box=box.DOUBLE
        ))
    
    # Initialize shared Agent (this triggers async initialization with env detection)
    # This also initializes document manager
    agent = await get_agent()
    lollms_client = get_lollms_client()
    
    # Register tools (single source of truth for tool registration)
    async def register_tools():
        tools_registered = []
        tools_failed = []
        
        tools_to_register = [
            ("filesystem", FilesystemTool()),
            ("http", HttpTool()),
            ("calendar", CalendarTool()),
        ]
        
        # Shell tool - more dangerous, only register if explicitly enabled
        if os.getenv("LOLLMSBOT_ENABLE_SHELL", "").lower() in ("true", "1", "yes"):
            tools_to_register.append(("shell", ShellTool()))
        
        for name, tool in tools_to_register:
            try:
                # Check if already registered to avoid errors
                if name not in agent.tools:
                    await agent.register_tool(tool)
                    tools_registered.append(name)
                    console.print(f"[green]  • {tool.__class__.__name__} registered[/]")
                else:
                    console.print(f"[dim]  • {tool.__class__.__name__} already registered[/]")
            except Exception as e:
                tools_failed.append((name, str(e)))
                if "already registered" not in str(e).lower():
                    console.print(f"[yellow]  • {tool.__class__.__name__} failed: {e}[/]")
        
        # Log summary
        if tools_registered:
            console.print(f"[dim]  Registered: {', '.join(tools_registered)}[/]")
        if tools_failed and not all("already registered" in str(e).lower() for _, e in tools_failed):
            console.print(f"[yellow]  Some tools failed to register[/]")
    
    # Register tools during startup
    await register_tools()
    
    # Show environment info
    if agent.environment_info:
        env = agent.environment_info
        console.print(f"[dim]  Platform: {env.os_system} {env.os_release}[/]")
        if env.in_docker:
            console.print(f"[dim]  Container: Docker ({env.container_id[:12] if env.container_id else 'unknown'})[/]")
        elif env.in_wsl:
            console.print(f"[dim]  Container: WSL[/]")
        if env.in_virtualenv:
            console.print(f"[dim]  Virtualenv: {env.virtualenv_path}[/]")
    
    console.print(f"[green]🚀 Gateway starting on http://{HOST}:{PORT}[/]")
    console.print(f"[dim]  • Chat endpoint: POST /chat[/]")
    console.print(f"[dim]  • File downloads: GET /files/download/<file_id>[/]")
    console.print(f"[dim]  • Document ingest: POST /documents/ingest[/]")
    console.print(f"[dim]  • Book creation: POST /documents/books[/]")
    if DEBUG_MODE:
        console.print(f"[dim]  • Debug memory: GET /debug/memory[/]")
        console.print(f"[dim]  • Debug stats: GET /debug/stats[/]")
    
    # Auto-enable UI
    if os.getenv("LOLLMSBOT_ENABLE_UI", "").lower() in ("true", "1", "yes"):
        enable_ui()
    
    global _active_channels, _channel_tasks, _agent, _document_manager
    
    # Discord with full agent capabilities
    if DISCORD_TOKEN:
        try:
            from lollmsbot.channels.discord import DiscordChannel
            
            def parse_id_list(val: Optional[str]) -> Optional[Set[int]]:
                if not val:
                    return None
                try:
                    return set(int(x.strip()) for x in val.split(","))
                except ValueError:
                    return None
            
            allowed_users = parse_id_list(DISCORD_ALLOWED_USERS)
            allowed_guilds = parse_id_list(DISCORD_ALLOWED_GUILDS)
            blocked_users = parse_id_list(DISCORD_BLOCKED_USERS)
            
            require_mention_guild = DISCORD_REQUIRE_MENTION_GUILD.lower() in ("true", "1", "yes")
            require_mention_dm = DISCORD_REQUIRE_MENTION_DM.lower() in ("true", "1", "yes")
            
            channel = DiscordChannel(
                agent=agent,
                bot_token=DISCORD_TOKEN,
                allowed_users=allowed_users,
                allowed_guilds=allowed_guilds,
                blocked_users=blocked_users,
                require_mention_in_guild=require_mention_guild,
                require_mention_in_dm=require_mention_dm,
            )
            _active_channels["discord"] = channel
            
            task = asyncio.create_task(channel.start())
            _channel_tasks.append(task)
            
            async def wait_discord():
                ready = await channel.wait_for_ready(timeout=15.0)
                if ready:
                    console.print("[bold green]✅ Discord connected with FULL AGENT capabilities![/]")
                    console.print("[dim]   File delivery enabled: Users receive generated files via DM[/]")
                    if allowed_users:
                        console.print(f"[dim]   Allowed users: {len(allowed_users)}[/]")
                    if blocked_users:
                        console.print(f"[dim]   Blocked users: {len(blocked_users)}[/]")
                else:
                    console.print("[yellow]⚠️  Discord still connecting...[/]")
            
            asyncio.create_task(wait_discord())
            
        except Exception as e:
            console.print(f"[red]❌ Discord failed: {e}[/]")
            import traceback
            traceback.print_exc()
    else:
        console.print("[dim]ℹ️  Discord disabled (no DISCORD_BOT_TOKEN)[/]")
    
    # Telegram
    if TELEGRAM_TOKEN:
        try:
            from lollmsbot.channels.telegram import TelegramChannel
            
            channel = TelegramChannel(
                agent=agent,
                bot_token=TELEGRAM_TOKEN,
            )
            _active_channels["telegram"] = channel
            
            task = asyncio.create_task(channel.start())
            _channel_tasks.append(task)
            console.print("[green]✅ Telegram started[/]")
            
        except Exception as e:
            console.print(f"[red]❌ Telegram failed: {e}[/]")
    else:
        console.print("[dim]ℹ️  Telegram disabled (no TELEGRAM_BOT_TOKEN)[/]")
    
    # Summary
    console.print(f"[bold green]📊 Active channels: {len(_active_channels)}[/]")
    console.print(f"[bold green]🤖 Agent: {agent.name} ({len(agent.tools)} tools)[/]")
    if _document_manager:
        doc_count = len(_document_manager.list_documents())
        console.print(f"[bold green]📚 Documents: {doc_count} indexed[/]")
    if lollms_client:
        console.print(f"[bold green]🔗 LoLLMS: Connected ({LollmsSettings.from_env().host_address})[/]")
    else:
        console.print(f"[yellow]⚠️  LoLLMS: Not connected - tools will work but chat uses fallback mode[/]")
    
    yield
    
    # Cleanup
    console.print("[yellow]🛑 Shutting down...[/]")
    
    for name, channel in _active_channels.items():
        try:
            await channel.stop()
            console.print(f"[dim]  • {name} stopped[/]")
        except Exception as e:
            console.print(f"[red]  • {name} error: {e}[/]")
    
    _active_channels.clear()
    
    for task in _channel_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    _channel_tasks.clear()
    
    # Close agent resources
    if _agent:
        await _agent.close()
        _agent = None
    
    console.print("[green]👋 Gateway shutdown complete[/]")

app.router.lifespan_context = lifespan

def main():
    """Main entry point with argument parsing."""
    global DEBUG_MODE, HOST, PORT
    
    parser = argparse.ArgumentParser(
        description="LollmsBot Gateway Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m lollmsbot.gateway              # Run normally
  python -m lollmsbot.gateway --debug      # Run with debug output
  python -m lollmsbot.gateway --host 0.0.0.0 --port 9000
        """
    )
    parser.add_argument("--host", type=str, default=HOST, help=f"Host to bind to (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to listen on (default: {PORT})")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with rich memory display")
    parser.add_argument("--ui", action="store_true", help="Enable web UI")
    
    args = parser.parse_args()
    
    # Apply arguments
    HOST = args.host
    PORT = args.port
    DEBUG_MODE = args.debug
    
    if args.ui:
        enable_ui()
    
    # Import and run uvicorn
    import uvicorn
    uvicorn.run(
        "lollmsbot.gateway:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="debug" if DEBUG_MODE else "info"
    )

if __name__ == "__main__":
    main()
