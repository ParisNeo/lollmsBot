"""
HTTP API channel implementation for LollmsBot.

Uses shared Agent for all business logic.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, status, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from lollmsbot.agent import Agent, PermissionLevel


logger = logging.getLogger(__name__)


@dataclass
class FileDelivery:
    """Represents a file ready for delivery to a user."""
    file_id: str
    original_path: str
    filename: str
    description: str
    user_id: str
    created_at: float
    content_type: Optional[str] = None


class HttpApiChannel:
    """HTTP API channel using shared Agent.
    
    Provides webhook endpoints and REST API for external integration.
    All business logic is delegated to the Agent.
    Includes file delivery endpoints for generated files.
    """

    def __init__(
        self,
        agent: Agent,
        host: str = "localhost",
        port: int = 8080,
        webhook_path: str = "/webhook",
        api_key: Optional[str] = None,
    ):
        self.agent = agent
        self.host = host
        self.port = port
        self.webhook_path = webhook_path
        self.api_key = api_key
        
        self.app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # File delivery storage
        self._pending_files: Dict[str, FileDelivery] = {}
        self._file_cleanup_task: Optional[asyncio.Task] = None
        self._file_ttl_seconds: float = 3600.0  # Files expire after 1 hour
        
        # Register file delivery callback with agent
        self.agent.set_file_delivery_callback(self._deliver_files)
        
        self._setup_app()

    def _generate_file_id(self, user_id: str, filename: str) -> str:
        """Generate unique file ID for download URL."""
        timestamp = str(time.time())
        hash_input = f"{user_id}:{filename}:{timestamp}:{os.urandom(8).hex()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    async def _deliver_files(self, user_id: str, files: List[Dict[str, Any]]) -> bool:
        """Store files for HTTP download delivery.
        
        This is the callback registered with the Agent for file delivery.
        Files are stored with unique IDs and made available via download endpoint.
        
        Args:
            user_id: Agent-format user ID (e.g., "http:anonymous").
            files: List of file dicts with 'path', 'filename', 'description' keys.
            
        Returns:
            True if files were registered for delivery successfully.
        """
        if not files:
            logger.debug(f"No files to deliver for {user_id}")
            return True
        
        try:
            logger.info(f"📤 Registering {len(files)} file(s) for download delivery to {user_id}")
            
            for file_info in files:
                file_path = file_info.get("path")
                filename = file_info.get("filename") or Path(file_path).name if file_path else "unnamed"
                description = file_info.get("description", "")
                
                if not file_path or not Path(file_path).exists():
                    logger.warning(f"File not found for delivery: {file_path}")
                    continue
                
                # Generate unique file ID
                file_id = self._generate_file_id(user_id, filename)
                
                # Determine content type
                content_type = self._guess_content_type(filename)
                
                # Store file delivery info
                delivery = FileDelivery(
                    file_id=file_id,
                    original_path=file_path,
                    filename=filename,
                    description=description,
                    user_id=user_id,
                    created_at=time.time(),
                    content_type=content_type,
                )
                self._pending_files[file_id] = delivery
                logger.info(f"✅ Registered file '{filename}' with ID {file_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register files for delivery to {user_id}: {e}")
            return False

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
            ".tar": "application/x-tar",
            ".gz": "application/gzip",
        }
        return mime_types.get(ext)

    async def _cleanup_expired_files(self) -> None:
        """Background task to clean up expired file registrations."""
        while self._is_running:
            try:
                now = time.time()
                expired = [
                    file_id for file_id, delivery in self._pending_files.items()
                    if now - delivery.created_at > self._file_ttl_seconds
                ]
                for file_id in expired:
                    del self._pending_files[file_id]
                    logger.debug(f"Cleaned up expired file registration: {file_id}")
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in file cleanup: {e}")
                await asyncio.sleep(60.0)

    def _setup_app(self) -> None:
        """Configure FastAPI application."""
        self.app = FastAPI(
            title="LollmsBot HTTP API",
            description="HTTP API for LollmsBot agent with file delivery support",
            version="0.2.0",
        )

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "agent": self.agent.name,
                "running": self._is_running,
                "pending_files": len(self._pending_files),
            }

        @self.app.post(self.webhook_path)
        async def webhook_endpoint(
            request: Request,
            x_api_key: Optional[str] = Header(None),
        ):
            """Handle incoming webhook messages."""
            # Validate API key if configured
            if self.api_key and x_api_key != self.api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                )
            
            try:
                body = await request.json()
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON",
                )
            
            user_id = body.get("user_id", "unknown")
            message = body.get("message", "").strip()
            
            if not message:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="message is required",
                )
            
            # Process via Agent
            result = await self.agent.chat(
                user_id=f"http:{user_id}",
                message=message,
                context={"channel": "http", "source": "webhook"},
            )
            
            # Build response with file download info
            response_data = {
                "success": result.get("success"),
                "response": result.get("response"),
                "error": result.get("error"),
                "tools_used": result.get("tools_used"),
            }
            
            # Add file download URLs if files were generated
            files_generated = result.get("files_to_send", [])
            if files_generated:
                download_urls = []
                for file_info in files_generated:
                    # Find the file_id we assigned
                    for file_id, delivery in self._pending_files.items():
                        if delivery.original_path == file_info.get("path"):
                            download_urls.append({
                                "filename": delivery.filename,
                                "download_url": f"/files/download/{file_id}",
                                "description": delivery.description,
                                "expires_in_seconds": int(self._file_ttl_seconds - (time.time() - delivery.created_at)),
                            })
                            break
                
                response_data["files"] = {
                    "count": len(download_urls),
                    "downloads": download_urls,
                }
            
            return response_data

        @self.app.post("/chat")
        async def chat_endpoint(request: Request):
            """Direct chat endpoint with file delivery support."""
            try:
                body = await request.json()
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON",
                )
            
            user_id = body.get("user_id", "anonymous")
            message = body.get("message", "").strip()
            
            if not message:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="message is required",
                )
            
            result = await self.agent.chat(
                user_id=f"http:{user_id}",
                message=message,
                context={"channel": "http", "source": "direct_api"},
            )
            
            # Build response with file download info
            response_data = {
                "success": result.get("success"),
                "response": result.get("response"),
                "error": result.get("error"),
                "tools_used": result.get("tools_used"),
            }
            
            # Add file download URLs if files were generated
            files_generated = result.get("files_to_send", [])
            if files_generated:
                download_urls = []
                for file_info in files_generated:
                    # Find the file_id we assigned
                    for file_id, delivery in self._pending_files.items():
                        if delivery.original_path == file_info.get("path"):
                            download_urls.append({
                                "filename": delivery.filename,
                                "download_url": f"/files/download/{file_id}",
                                "description": delivery.description,
                                "expires_in_seconds": int(self._file_ttl_seconds - (time.time() - delivery.created_at)),
                            })
                            break
                
                response_data["files"] = {
                    "count": len(download_urls),
                    "downloads": download_urls,
                }
            
            return response_data

        @self.app.get("/files/download/{file_id}")
        async def download_file(file_id: str):
            """Download a generated file by its temporary ID."""
            if file_id not in self._pending_files:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found or expired",
                )
            
            delivery = self._pending_files[file_id]
            
            # Check if file still exists on disk
            file_path = Path(delivery.original_path)
            if not file_path.exists():
                # Clean up stale registration
                del self._pending_files[file_id]
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File no longer available",
                )
            
            # Check expiration
            if time.time() - delivery.created_at > self._file_ttl_seconds:
                del self._pending_files[file_id]
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail="File download link has expired",
                )
            
            # Return file with appropriate content type
            media_type = delivery.content_type or "application/octet-stream"
            
            return FileResponse(
                path=str(file_path),
                filename=delivery.filename,
                media_type=media_type,
            )

        @self.app.get("/files/list")
        async def list_pending_files(user_id: Optional[str] = Query(None)):
            """List pending files for a user (or all if no user specified)."""
            files = []
            for file_id, delivery in self._pending_files.items():
                if user_id is None or delivery.user_id == user_id:
                    files.append({
                        "file_id": file_id,
                        "filename": delivery.filename,
                        "description": delivery.description,
                        "download_url": f"/files/download/{file_id}",
                        "created_at": delivery.created_at,
                        "expires_in_seconds": int(self._file_ttl_seconds - (time.time() - delivery.created_at)),
                    })
            
            return {
                "count": len(files),
                "files": files,
            }

    async def start(self) -> None:
        """Start the HTTP server."""
        if self._is_running:
            return
        
        try:
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
            )
            self._server = uvicorn.Server(config)
            self._server_task = asyncio.create_task(self._server.serve())
            
            # Start file cleanup task
            self._file_cleanup_task = asyncio.create_task(self._cleanup_expired_files())
            
            self._is_running = True
            
            logger.info(f"HTTP API channel started on http://{self.host}:{self.port}")
            logger.info(f"File download endpoint: http://{self.host}:{self.port}/files/download/<file_id>")
            
        except Exception as exc:
            logger.error(f"Failed to start HTTP API channel: {exc}")
            self._is_running = False
            raise

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop cleanup task
        if self._file_cleanup_task:
            self._file_cleanup_task.cancel()
            try:
                await self._file_cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._server:
            self._server.should_exit = True
        
        if self._server_task:
            try:
                await asyncio.wait_for(self._server_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
        
        # Clear pending files
        self._pending_files.clear()
        
        self._server = None
        self._server_task = None
        self._file_cleanup_task = None
        logger.info("HTTP API channel stopped")

    def __repr__(self) -> str:
        status = "running" if self._is_running else "stopped"
        return f"HttpApiChannel({self.host}:{self.port}, {status}, {len(self._pending_files)} pending files)"
