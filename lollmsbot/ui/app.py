"""
Web UI application for LollmsBot.

Uses shared Agent for all business logic.
"""

import asyncio
import json
import logging
import os
import time
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Set, Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

from lollmsbot.config import BotConfig, LollmsSettings
from lollmsbot.agent import Agent


logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def send_to(self, websocket: WebSocket, message: dict) -> bool:
        try:
            await websocket.send_json(message)
            return True
        except Exception:
            return False


class PendingFile:
    """Represents a file pending download in the Web UI."""
    def __init__(
        self,
        file_id: str,
        file_path: str,
        filename: str,
        description: str,
        user_id: str,
    ):
        self.file_id = file_id
        self.file_path = file_path
        self.filename = filename
        self.description = description
        self.user_id = user_id
        self.created_at = time.time()
        self.content_type = self._guess_content_type(filename)
    
    def _guess_content_type(self, filename: str) -> Optional[str]:
        """Guess MIME type from filename extension."""
        ext = Path(filename).suffix.lower()
        mime_types = {
            ".html": "text/html",
            ".htm": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".py": "text/x-python",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
            ".zip": "application/zip",
        }
        return mime_types.get(ext, "application/octet-stream")
    
    def is_expired(self, ttl_seconds: float = 3600.0) -> bool:
        """Check if file download has expired."""
        return time.time() - self.created_at > ttl_seconds


class WebUI:
    """
    Web UI for LollmsBot using shared Agent.
    
    All chat processing is delegated to the Agent instance.
    Includes file delivery support with download endpoints.
    """

    def __init__(
        self,
        agent: Agent,
        lollms_settings: Optional[LollmsSettings] = None,
        bot_config: Optional[BotConfig] = None,
        static_dir: Optional[Path] = None,
        templates_dir: Optional[Path] = None,
        verbose: bool = True,
    ) -> None:
        self.agent = agent  # Shared Agent instance!
        self.lollms_settings = lollms_settings or LollmsSettings.from_env()
        self.bot_config = bot_config or BotConfig.from_env()
        self.verbose = verbose
        
        self.package_dir = Path(__file__).parent
        self.static_dir = static_dir or self.package_dir / "static"
        self.templates_dir = templates_dir or self.package_dir / "templates"
        
        # File delivery storage
        self._pending_files: Dict[str, PendingFile] = {}
        self._file_cleanup_task: Optional[asyncio.Task] = None
        self._file_ttl_seconds: float = 3600.0  # Files expire after 1 hour
        
        if self.verbose:
            self._print_startup_banner()
        
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_static_files()
        
        self.manager = ConnectionManager()
        self.active_sessions: Set[str] = set()
        
        # Set up file delivery callback
        self.agent.set_file_delivery_callback(self._deliver_files)
        
        # Set up tool event callback for real-time updates
        self.agent.set_tool_event_callback(self._handle_tool_event)
        
        self.app = self._create_app()
        
        if self.verbose:
            logger.info(f"WebUI initialized with Agent: {agent.name}")
    
    def _generate_file_id(self, user_id: str, filename: str) -> str:
        """Generate unique file ID for download URL."""
        timestamp = str(time.time())
        hash_input = f"{user_id}:{filename}:{timestamp}:{os.urandom(8).hex()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    async def _deliver_files(self, user_id: str, files: List[Dict[str, Any]]) -> bool:
        """Store files for Web UI download delivery.
        
        This is the callback registered with the Agent for file delivery.
        Files are stored with unique IDs and made available via download endpoint.
        Users are notified via WebSocket about available downloads.
        
        Args:
            user_id: Agent-format user ID (e.g., "web:session_abc123").
            files: List of file dicts with 'path', 'filename', 'description' keys.
            
        Returns:
            True if files were registered for delivery successfully.
        """
        if not files:
            logger.debug(f"No files to deliver for {user_id}")
            return True
        
        try:
            logger.info(f"📤 Registering {len(files)} file(s) for Web UI delivery to {user_id}")
            
            file_infos = []
            for file_info in files:
                file_path = file_info.get("path")
                filename = file_info.get("filename") or Path(file_path).name if file_path else "unnamed"
                description = file_info.get("description", "")
                
                if not file_path or not Path(file_path).exists():
                    logger.warning(f"File not found for delivery: {file_path}")
                    continue
                
                # Generate unique file ID
                file_id = self._generate_file_id(user_id, filename)
                
                # Store file info
                pending = PendingFile(
                    file_id=file_id,
                    file_path=file_path,
                    filename=filename,
                    description=description,
                    user_id=user_id,
                )
                self._pending_files[file_id] = pending
                
                file_infos.append({
                    "file_id": file_id,
                    "filename": filename,
                    "description": description,
                    "download_url": f"/download/{file_id}",
                })
                logger.info(f"✅ Registered file '{filename}' with ID {file_id}")
            
            # Notify connected clients about available downloads
            # Find WebSocket connections for this user
            await self._notify_file_available(user_id, file_infos)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register files for delivery to {user_id}: {e}")
            return False
    
    async def _notify_file_available(self, user_id: str, file_infos: List[Dict[str, Any]]) -> None:
        """Notify WebSocket clients about available file downloads."""
        # For simplicity, broadcast to all connections with matching user_id prefix
        message = {
            "type": "files_ready",
            "files": file_infos,
        }
        
        # In a real implementation, you'd track which WebSocket belongs to which user
        # For now, broadcast to all and let client filter
        disconnected = []
        for ws in self.manager.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        
        # Clean up disconnected clients
        for ws in disconnected:
            await self.manager.disconnect(ws)
    
    async def _cleanup_expired_files(self) -> None:
        """Background task to clean up expired file registrations."""
        while True:
            try:
                expired_ids = [
                    file_id for file_id, pending in list(self._pending_files.items())
                    if pending.is_expired(self._file_ttl_seconds)
                ]
                for file_id in expired_ids:
                    del self._pending_files[file_id]
                    logger.debug(f"Cleaned up expired file registration: {file_id}")
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in file cleanup: {e}")
                await asyncio.sleep(60.0)
    
    async def _handle_tool_event(self, event_type: str, tool_name: str, data: Optional[Dict[str, Any]]) -> None:
        """Handle tool execution events from the Agent and broadcast to WebSocket clients."""
        ui_events = {
            "planning_start": {"type": "thinking", "content": "Analyzing your request..."},
            "planning_complete": {
                "type": "tools_planned",
                "content": f"Planning complete: {data.get('tools_count', 0)} tool(s) will be used" if data else "Planning complete",
                "tools": data.get("tools_list", []) if data else [],
            },
            "tool_start": {
                "type": "tool_calling",
                "content": f"Calling tool: {tool_name}...",
                "tool_name": tool_name,
                "parameters": data.get("parameters", {}) if data else {},
            },
            "tool_complete": {
                "type": "tool_complete",
                "content": f"Tool '{tool_name}' completed",
                "tool_name": tool_name,
                "success": data.get("success", False) if data else False,
                "duration": data.get("duration", 0) if data else 0,
            },
            "tool_error": {
                "type": "tool_error",
                "content": f"Tool '{tool_name}' failed: {data.get('error', 'Unknown error') if data else 'Unknown error'}",
                "tool_name": tool_name,
                "error": data.get("error", "Unknown error") if data else "Unknown error",
            },
            "tool_denied": {
                "type": "tool_denied",
                "content": f"Tool '{tool_name}' not available: {data.get('reason', 'Permission denied') if data else 'Permission denied'}",
                "tool_name": tool_name,
            },
        }
        
        ui_message = ui_events.get(event_type)
        if ui_message:
            disconnected = []
            for ws in self.manager.active_connections:
                try:
                    await ws.send_json({
                        "type": "tool_event",
                        "event_type": event_type,
                        **ui_message,
                    })
                except Exception:
                    disconnected.append(ws)
            
            for ws in disconnected:
                await self.manager.disconnect(ws)
    
    def _print_startup_banner(self) -> None:
        """Print startup banner."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text
            
            console = Console()
            
            banner_text = Text()
            banner_text.append("🤖 Web UI\n", style="bold cyan")
            banner_text.append(f"Agent: {self.agent.name}\n", style="dim")
            
            panel = Panel(
                banner_text,
                border_style="blue",
                title="[bold]Web Interface[/bold]",
            )
            console.print()
            console.print(panel)
            console.print()
            
        except ImportError:
            print(f"\n🤖 Web UI - Agent: {self.agent.name}\n")
    
    def _ensure_static_files(self) -> None:
        """Create default static files if missing."""
        css_dir = self.static_dir / "css"
        js_dir = self.static_dir / "js"
        css_dir.mkdir(parents=True, exist_ok=True)
        js_dir.mkdir(parents=True, exist_ok=True)
        
        style_css = css_dir / "style.css"
        if not style_css.exists():
            style_css.write_text(self._default_css(), encoding='utf-8')
            logger.info(f"Created default CSS at {style_css}")
        
        app_js = js_dir / "app.js"
        if not app_js.exists():
            app_js.write_text(self._default_js(), encoding='utf-8')
            logger.info(f"Created default JS at {app_js}")
        
        index_html = self.templates_dir / "index.html"
        if not index_html.exists():
            index_html.write_text(self._default_html_template(), encoding='utf-8')
            logger.info(f"Created default HTML template at {index_html}")
    
    def _default_css(self) -> str:
        """Default CSS."""
        return """/* Minimal CSS for LollmsBot Web UI */
:root { --primary: #6366f1; --bg: #0f172a; --text: #f8fafc; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; }
.chat-container { display: flex; flex-direction: column; height: 100vh; }
.messages { flex: 1; overflow-y: auto; padding: 20px; }
.input-area { padding: 20px; border-top: 1px solid #334155; }
.message { margin: 10px 0; padding: 12px; border-radius: 8px; }
.message.user { background: var(--primary); margin-left: 20%; }
.message.assistant { background: #1e293b; margin-right: 20%; }
input, button { padding: 10px; border-radius: 4px; border: none; }
input { flex: 1; background: #334155; color: var(--text); }
button { background: var(--primary); color: white; cursor: pointer; }
.input-row { display: flex; gap: 10px; }

/* File download styles */
.file-download {
    margin: 10px 0;
    padding: 12px 16px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.file-download-icon {
    font-size: 1.5rem;
}

.file-download-info {
    flex: 1;
}

.file-download-name {
    font-weight: 500;
    color: #f8fafc;
}

.file-download-desc {
    font-size: 0.85rem;
    color: #94a3b8;
}

.file-download-btn {
    padding: 8px 16px;
    background: #10b981;
    color: white;
    border-radius: 6px;
    text-decoration: none;
    font-size: 0.9rem;
    transition: background 0.2s;
}

.file-download-btn:hover {
    background: #059669;
}

.files-section {
    margin: 16px 0;
    padding: 16px;
    background: rgba(99, 102, 241, 0.05);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
}

.files-section-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #6366f1;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
"""
    
    def _default_js(self) -> str:
        """Default JavaScript with file download support."""
        return """// Web UI JavaScript with file download support
console.log('[ChatApp] Script loading...');

class ChatApp {
    constructor() {
        console.log('[ChatApp] Initializing...');
        this.ws = null;
        this.sessionId = 'web_' + Math.random().toString(36).substr(2, 9);
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.pendingFiles = [];
        this.init();
    }
    
    init() {
        console.log('[ChatApp] Setting up event listeners...');
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupEventListeners());
        } else {
            this.setupEventListeners();
        }
        
        this.connect();
    }
    
    setupEventListeners() {
        console.log('[ChatApp] Attaching button and input handlers...');
        
        const sendBtn = document.getElementById('send-btn');
        const messageInput = document.getElementById('message-input');
        
        if (!sendBtn) {
            console.error('[ChatApp] Send button not found!');
            return;
        }
        if (!messageInput) {
            console.error('[ChatApp] Message input not found!');
            return;
        }
        
        sendBtn.addEventListener('click', (e) => {
            console.log('[ChatApp] Send button clicked');
            e.preventDefault();
            this.send();
        });
        
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                console.log('[ChatApp] Enter key pressed');
                e.preventDefault();
                this.send();
            }
        });
        
        console.log('[ChatApp] Event listeners attached successfully');
    }
    
    updateStatus(status, text) {
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        
        if (statusDot) {
            statusDot.className = 'status-dot ' + (status === 'connected' ? '' : 'disconnected');
        }
        if (statusText) {
            statusText.textContent = text;
        }
        console.log('[ChatApp] Status updated:', status, text);
    }
    
    connect() {
        console.log('[ChatApp] Attempting WebSocket connection...');
        
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        let path = window.location.pathname;
        path = path.replace(/\\\\/$/, '');
        
        const wsPath = path ? path + '/ws/chat' : '/ws/chat';
        const wsUrl = wsProtocol + '//' + host + wsPath;
        
        console.log('[ChatApp] WebSocket URL:', wsUrl);
        
        try {
            this.ws = new WebSocket(wsUrl);
        } catch (err) {
            console.error('[ChatApp] Failed to create WebSocket:', err);
            this.updateStatus('error', 'Connection failed');
            return;
        }
        
        this.ws.onopen = () => {
            console.log('[ChatApp] WebSocket connected!');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateStatus('connected', 'Connected');
            this.addMessage('system', 'Connected to AI assistant');
        };
        
        this.ws.onmessage = (e) => {
            console.log('[ChatApp] Message received:', e.data);
            try {
                const data = JSON.parse(e.data);
                
                if (data.type === 'tool_event') {
                    this.handleToolEvent(data);
                    return;
                }
                
                if (data.type === 'files_ready') {
                    this.handleFilesReady(data.files);
                    return;
                }
                
                if (data.type === 'response') {
                    this.addMessage('assistant', data.content);
                } else if (data.type === 'error') {
                    this.addMessage('system', 'Error: ' + data.message);
                } else if (data.type === 'ping') {
                    console.log('[ChatApp] Ping received');
                }
            } catch (err) {
                console.error('[ChatApp] Failed to parse message:', err);
            }
        };
        
        this.ws.onerror = (err) => {
            console.error('[ChatApp] WebSocket error:', err);
            this.updateStatus('error', 'Connection error');
        };
        
        this.ws.onclose = (e) => {
            console.log('[ChatApp] WebSocket closed:', e.code, e.reason);
            this.isConnected = false;
            this.updateStatus('disconnected', 'Disconnected');
            
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
                console.log(`[ChatApp] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
                this.updateStatus('reconnecting', `Reconnecting... (${this.reconnectAttempts})`);
                setTimeout(() => this.connect(), delay);
            } else {
                this.updateStatus('failed', 'Connection failed - refresh page');
                this.addMessage('system', 'Connection lost. Please refresh the page to reconnect.');
            }
        };
    }
    
    handleToolEvent(data) {
        console.log('[ChatApp] Tool event:', data);
        
        const eventType = data.event_type;
        let messageText = '';
        
        switch(eventType) {
            case 'planning_start':
                messageText = '🤔 Thinking...';
                break;
            case 'planning_complete':
                const toolsCount = data.tools ? data.tools.length : 0;
                messageText = toolsCount > 0 ? `📋 Will use: ${data.tools.join(', ')}` : '💬 Responding directly...';
                break;
            case 'tool_calling':
                messageText = `🔧 Calling ${data.tool_name}...`;
                break;
            case 'tool_complete':
                const successIcon = data.success ? '✅' : '❌';
                messageText = `${successIcon} ${data.tool_name} done (${(data.duration || 0).toFixed(1)}s)`;
                break;
            case 'tool_error':
                messageText = `💥 ${data.tool_name} failed: ${data.error}`;
                break;
            default:
                messageText = `[${eventType}]`;
        }
        
        this.removeToolIndicators();
        this.addToolIndicator(messageText);
    }
    
    handleFilesReady(files) {
        console.log('[ChatApp] Files ready:', files);
        this.pendingFiles = files;
        this.showFileDownloads(files);
    }
    
    showFileDownloads(files) {
        const messagesArea = document.getElementById('messages-area');
        if (!messagesArea) return;
        
        // Create files section
        const filesSection = document.createElement('div');
        filesSection.className = 'files-section';
        
        const title = document.createElement('div');
        title.className = 'files-section-title';
        title.innerHTML = '📎 Generated Files';
        filesSection.appendChild(title);
        
        files.forEach(file => {
            const downloadDiv = document.createElement('div');
            downloadDiv.className = 'file-download';
            
            const icon = document.createElement('span');
            icon.className = 'file-download-icon';
            icon.textContent = this.getFileIcon(file.filename);
            
            const info = document.createElement('div');
            info.className = 'file-download-info';
            
            const name = document.createElement('div');
            name.className = 'file-download-name';
            name.textContent = file.filename;
            
            const desc = document.createElement('div');
            desc.className = 'file-download-desc';
            desc.textContent = file.description || 'Generated file ready for download';
            
            info.appendChild(name);
            info.appendChild(desc);
            
            const btn = document.createElement('a');
            btn.className = 'file-download-btn';
            btn.href = file.download_url;
            btn.download = file.filename;
            btn.textContent = 'Download';
            
            downloadDiv.appendChild(icon);
            downloadDiv.appendChild(info);
            downloadDiv.appendChild(btn);
            filesSection.appendChild(downloadDiv);
        });
        
        messagesArea.appendChild(filesSection);
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
    
    getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const icons = {
            html: '🌐', htm: '🌐', css: '🎨', js: '⚡', py: '🐍',
            json: '📋', txt: '📝', md: '📄', csv: '📊',
            png: '🖼️', jpg: '🖼️', jpeg: '🖼️', gif: '🖼️', svg: '🖼️',
            pdf: '📑', zip: '📦', tar: '📦', gz: '📦',
        };
        return icons[ext] || '📄';
    }
    
    removeToolIndicators() {
        const indicators = document.querySelectorAll('.tool-indicator');
        indicators.forEach(el => {
            el.style.opacity = '0.5';
            setTimeout(() => el.remove(), 500);
        });
    }
    
    addToolIndicator(text) {
        const messagesArea = document.getElementById('messages-area');
        if (!messagesArea) return;
        
        const indicator = document.createElement('div');
        indicator.className = 'tool-indicator';
        indicator.style.cssText = `
            padding: 8px 16px;
            margin: 8px 20%;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 8px;
            font-size: 0.85rem;
            color: #94a3b8;
            text-align: center;
            transition: all 0.3s ease;
        `;
        indicator.textContent = text;
        messagesArea.appendChild(indicator);
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
    
    send() {
        console.log('[ChatApp] Send called, connection state:', this.isConnected);
        
        const input = document.getElementById('message-input');
        if (!input) {
            console.error('[ChatApp] Input not found!');
            return;
        }
        
        const text = input.value.trim();
        if (!text) {
            console.log('[ChatApp] Empty message, ignoring');
            return;
        }
        
        if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.error('[ChatApp] WebSocket not connected!');
            this.addMessage('system', 'Not connected - please wait or refresh');
            return;
        }
        
        console.log('[ChatApp] Sending message:', text);
        
        this.removeToolIndicators();
        this.addMessage('user', text);
        input.value = '';
        
        const message = {
            type: 'message',
            session_id: this.sessionId,
            message: text
        };
        
        try {
            this.ws.send(JSON.stringify(message));
            console.log('[ChatApp] Message sent successfully');
        } catch (err) {
            console.error('[ChatApp] Failed to send:', err);
            this.addMessage('system', 'Failed to send message');
        }
    }
    
    addMessage(role, text) {
        console.log('[ChatApp] Adding message:', role, text.substring(0, 50));
        
        this.removeToolIndicators();
        
        const messagesArea = document.getElementById('messages-area');
        if (!messagesArea) {
            console.error('[ChatApp] Messages area not found!');
            return;
        }
        
        const welcomeScreen = document.getElementById('welcome-screen');
        if (welcomeScreen) {
            welcomeScreen.style.display = 'none';
        }
        
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ' + role;
        
        if (role === 'system') {
            msgDiv.style.cssText = 'text-align: center; color: #64748b; font-style: italic; margin: 10px 0;';
        } else if (role === 'user') {
            msgDiv.style.cssText = 'background: #6366f1; color: white; margin-left: 20%; padding: 12px; border-radius: 8px; margin-bottom: 10px;';
        } else if (role === 'assistant') {
            msgDiv.style.cssText = 'background: #1e293b; color: #f8fafc; margin-right: 20%; padding: 12px; border-radius: 8px; margin-bottom: 10px;';
        }
        
        // Simple markdown-like formatting
        let formattedText = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\\`\\`\\`([\\s\\S]*?)\\`\\`\\`/g, '<pre><code>$1</code></pre>')
            .replace(/\\`([^`]+)\\`/g, '<code>$1</code>')
            .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
            .replace(/\\*([^*]+)\\*/g, '<em>$1</em>')
            .replace(/\\n/g, '<br>');
        
        msgDiv.innerHTML = formattedText;
        messagesArea.appendChild(msgDiv);
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }
}

// Initialize
console.log('[ChatApp] Script loaded, waiting for initialization...');
const app = new ChatApp();
window.chatApp = app;
"""
    
    def _default_html_template(self) -> str:
        """Default HTML template with download support."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LollmsBot - AI Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="./static/css/style.css">
</head>
<body>
    <div class="app">
        <header class="header">
            <div class="logo">
                <div class="logo-icon">AI</div>
                <span>LollmsBot</span>
            </div>
            <div class="header-actions">
                <button class="btn-icon" id="menu-btn" title="Menu">☰</button>
                <button class="btn-icon" id="settings-btn" title="Settings">⚙</button>
            </div>
        </header>
        
        <aside class="sidebar">
            <div class="sidebar-section">
                <button class="new-chat-btn" id="new-chat-btn">
                    <span>+</span>
                    New Chat
                </button>
            </div>
            
            <div class="sidebar-section">
                <div class="sidebar-title">Conversations</div>
                <div class="conversation-list" id="conversation-list">
                    <div class="conversation-item active">
                        <div class="conversation-icon">C</div>
                        <span>Current Chat</span>
                    </div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <div class="sidebar-title">Available Tools</div>
                <div class="tools-grid">
                    <div class="tool-badge" title="File operations">Files</div>
                    <div class="tool-badge" title="HTTP requests">HTTP</div>
                    <div class="tool-badge" title="Calendar">Calendar</div>
                    <div class="tool-badge" title="Shell commands">Shell</div>
                </div>
            </div>
            
            <div class="sidebar-section" style="margin-top: auto;">
                <div class="status-indicator">
                    <span class="status-dot" id="status-dot"></span>
                    <span id="status-text">Connecting...</span>
                </div>
            </div>
        </aside>
        
        <main class="chat-container">
            <div class="messages-area" id="messages-area">
                <div class="welcome-screen" id="welcome-screen">
                    <div class="welcome-icon">🤖</div>
                    <div class="welcome-title">Welcome to LollmsBot</div>
                    <div class="welcome-subtitle">Your AI assistant is ready</div>
                    <div style="margin-top: 20px; font-size: 0.9rem; color: #64748b;">
                        <p>I can generate files for you! Try asking me to:</p>
                        <ul style="text-align: left; display: inline-block; margin-top: 8px;">
                            <li>Create an HTML game or app</li>
                            <li>Write Python scripts</li>
                            <li>Generate CSV data files</li>
                            <li>Build ZIP archives</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-container">
                    <textarea 
                        class="message-input" 
                        id="message-input"
                        placeholder="Message LollmsBot... Try 'create a snake game'"
                        rows="1"
                    ></textarea>
                    <div class="input-actions">
                        <button class="send-btn" id="send-btn" title="Send">➤</button>
                    </div>
                </div>
            </div>
        </main>
    </div>
    
    <script src="./static/js/app.js"></script>
</body>
</html>
"""
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with file download endpoints."""
        app = FastAPI(title="LollmsBot Web UI")
        
        # Mount static files
        app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
        templates = Jinja2Templates(directory=str(self.templates_dir))
        
        @app.on_event("startup")
        async def startup_event():
            self._file_cleanup_task = asyncio.create_task(self._cleanup_expired_files())
        
        @app.on_event("shutdown")
        async def shutdown_event():
            if self._file_cleanup_task:
                self._file_cleanup_task.cancel()
        
        @app.get("/")
        async def index(request: Request):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "agent_name": self.agent.name,
                "lollms_host": self.lollms_settings.host_address,
                "max_history": self.bot_config.max_history,
            })
        
        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "ui": "running",
                "agent": self.agent.name,
                "agent_state": self.agent.state.name,
                "websocket_clients": len(self.manager.active_connections),
                "pending_files": len(self._pending_files),
            }
        
        @app.get("/download/{file_id}")
        async def download_file(file_id: str):
            """Download a generated file by its temporary ID."""
            if file_id not in self._pending_files:
                return {"error": "File not found or expired"}, 404
            
            pending = self._pending_files[file_id]
            
            # Check if file still exists
            file_path = Path(pending.file_path)
            if not file_path.exists():
                del self._pending_files[file_id]
                return {"error": "File no longer available"}, 404
            
            # Check expiration
            if pending.is_expired(self._file_ttl_seconds):
                del self._pending_files[file_id]
                return {"error": "File download link has expired"}, 410
            
            return FileResponse(
                path=str(file_path),
                filename=pending.filename,
                media_type=pending.content_type or "application/octet-stream",
            )
        
        @app.get("/files/list")
        async def list_files(user_id: Optional[str] = Query(None)):
            """List pending files."""
            files = []
            for file_id, pending in self._pending_files.items():
                if user_id is None or pending.user_id == user_id:
                    files.append({
                        "file_id": file_id,
                        "filename": pending.filename,
                        "description": pending.description,
                        "download_url": f"/download/{file_id}",
                        "expires_in": int(self._file_ttl_seconds - (time.time() - pending.created_at)),
                    })
            return {"files": files, "count": len(files)}
        
        @app.websocket("/ws/chat")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            session_id: Optional[str] = None
            
            try:
                while True:
                    data = await websocket.receive_json()
                    msg_type = data.get("type")
                    session_id = data.get("session_id", "unknown")
                    
                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                    
                    if msg_type == "message":
                        await self._handle_message(websocket, data, session_id)
                        
            except WebSocketDisconnect:
                logger.info(f"Client disconnected: {session_id}")
            except Exception as exc:
                logger.error(f"WebSocket error: {exc}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Server error: {str(exc)}"
                    })
                except:
                    pass
            finally:
                await self.manager.disconnect(websocket)
        
        return app
    
    async def _handle_message(
        self,
        websocket: WebSocket,
        data: dict,
        session_id: str,
    ) -> None:
        """Handle chat message via Agent with file delivery."""
        message = data.get("message", "").strip()
        if not message:
            await self.manager.send_to(websocket, {
                "type": "error",
                "message": "Empty message"
            })
            return
        
        user_id = f"web:{session_id}"
        
        logger.info(f"Processing message from web user {user_id}: {message[:50]}...")
        
        try:
            result = await self.agent.chat(
                user_id=user_id,
                message=message,
                context={"channel": "web_ui", "session_id": session_id},
            )
            
            if result.get("permission_denied"):
                await self.manager.send_to(websocket, {
                    "type": "error",
                    "message": "Access denied"
                })
                return
            
            if not result.get("success"):
                await self.manager.send_to(websocket, {
                    "type": "error",
                    "message": result.get("error", "Unknown error")
                })
                return
            
            response = result.get("response", "No response")
            
            # Send response first
            await self.manager.send_to(websocket, {
                "type": "response",
                "content": response,
                "tools_used": result.get("tools_used", [])
            })
            
            # Files will be delivered via the callback and subsequent files_ready message
            
        except Exception as exc:
            logger.error(f"Error processing message: {exc}")
            await self.manager.send_to(websocket, {
                "type": "error",
                "message": f"Processing error: {str(exc)}"
            })
    
    def print_server_ready(self, host: str, port: int) -> None:
        """Print server ready message."""
        if not self.verbose:
            return
        
        display_host = "localhost" if host in ("0.0.0.0", "") else host
        
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            
            console = Console()
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Access Point", style="cyan")
            table.add_column("URL", style="green")
            
            table.add_row("Chat UI", f"http://{display_host}:{port}")
            table.add_row("Download Files", f"http://{display_host}:{port}/download/<file_id>")
            table.add_row("List Files", f"http://{display_host}:{port}/files/list")
            
            panel = Panel(
                f"[bold green]✅ Web UI Ready[/bold green]\n\n{table}",
                border_style="green",
            )
            console.print()
            console.print(panel)
            console.print()
            
        except ImportError:
            print(f"\n✅ Web UI running at http://{display_host}:{port}")
            print(f"   File downloads: http://{display_host}:{port}/download/<file_id>\n")
    
    def _print_shutdown_message(self) -> None:
        """Print shutdown message."""
        if self.verbose:
            print("\n👋 Web UI shutting down...\n")
