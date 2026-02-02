"""
HTTP API channel implementation for LollmsBot.

This module provides the HttpApiChannel class for bidirectional HTTP communication
via webhooks and REST API endpoints. It supports receiving incoming messages
via webhook endpoints and sending responses via configurable HTTP POST callbacks.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from lollmsbot.channels import Channel
from lollmsbot.agent import Agent


logger = logging.getLogger(__name__)


class HttpApiChannel(Channel):
    """HTTP API channel implementation with bidirectional webhook support.
    
    Provides HTTP-based communication through webhook endpoints for receiving
    messages and configurable HTTP POST callbacks for sending responses.
    Supports optional webhook signature verification for security.
    
    Attributes:
        host: Host address to bind the server to.
        port: Port number to listen on.
        webhook_path: URL path for the webhook endpoint.
        webhook_secret: Optional secret for HMAC signature verification.
        callback_url: Optional URL for sending outbound messages via POST.
        callback_secret: Optional secret for signing outbound callbacks.
        app: FastAPI sub-application for webhook endpoints.
        _server: Uvicorn server instance.
        _server_task: Asyncio task running the server.
        _callback_client: HTTP client session for callbacks.
        _shutdown_event: Event for coordinating graceful shutdown.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        webhook_path: str = "/webhook",
        webhook_secret: Optional[str] = None,
        callback_url: Optional[str] = None,
        callback_secret: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> None:
        """Initialize the HTTP API channel.
        
        Args:
            host: Host address to bind the server to.
            port: Port number to listen on.
            webhook_path: URL path for incoming webhooks (default: /webhook).
            webhook_secret: Optional secret for verifying webhook signatures.
                          When provided, incoming webhooks must include valid
                          X-Webhook-Signature header.
            callback_url: Optional URL for sending outbound messages via HTTP POST.
                         If provided, send_message will POST to this URL.
            callback_secret: Optional secret for signing outbound callbacks.
            agent: Optional Agent instance for message processing.
        """
        super().__init__(name=f"http_api:{host}:{port}", agent=agent)
        
        self.host: str = host
        self.port: int = port
        self.webhook_path: str = webhook_path
        self.webhook_secret: Optional[str] = webhook_secret
        self.callback_url: Optional[str] = callback_url
        self.callback_secret: Optional[str] = callback_secret
        
        self.app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._server_task: Optional[asyncio.Task[None]] = None
        self._callback_client: Optional[Any] = None  # aiohttp.ClientSession
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._message_callback: Optional[
            Callable[[str, str], Awaitable[None]]
        ] = None
        
        self._setup_app()
    
    def _setup_app(self) -> None:
        """Configure the FastAPI sub-application with webhook endpoints."""
        self.app = FastAPI(
            title="LollmsBot HTTP Channel",
            description="Webhook endpoints for HTTP-based bot communication",
            version="0.1.0",
        )
        
        @self.app.post(self.webhook_path)
        async def webhook_endpoint(
            request: Request,
            x_webhook_signature: Optional[str] = Header(None, alias="X-Webhook-Signature"),
            x_message_id: Optional[str] = Header(None, alias="X-Message-ID"),
            x_sender_id: Optional[str] = Header(None, alias="X-Sender-ID"),
            x_timestamp: Optional[str] = Header(None, alias="X-Timestamp"),
        ) -> JSONResponse:
            """Handle incoming webhook messages.
            
            Accepts JSON payloads containing message data. Supports optional
            signature verification and extracts sender information from
            headers or payload.
            
            Headers:
                X-Webhook-Signature: HMAC-SHA256 signature of payload (if secret configured)
                X-Message-ID: Unique identifier for this message
                X-Sender-ID: Identifier of the message sender
                X-Timestamp: Unix timestamp of the message
            
            Request Body:
                {
                    "message": "text content",
                    "sender_id": "optional_override",
                    "metadata": {}
                }
            """
            try:
                body = await request.body()
                payload = await request.json()
            except json.JSONDecodeError as exc:
                logger.error(f"Invalid JSON in webhook payload: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON payload",
                )
            except Exception as exc:
                logger.error(f"Failed to read webhook payload: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to read request body",
                )
            
            # Verify signature if secret is configured
            if self.webhook_secret is not None:
                if x_webhook_signature is None:
                    logger.warning("Webhook missing signature header")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Missing X-Webhook-Signature header",
                    )
                
                if not self._verify_signature(body, x_webhook_signature):
                    logger.warning("Webhook signature verification failed")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid webhook signature",
                    )
            
            # Extract message data
            message_text = payload.get("message", "").strip()
            if not message_text:
                logger.warning("Webhook received empty message")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Message field is required and must not be empty",
                )
            
            # Determine sender ID (header takes precedence over payload)
            sender_id = x_sender_id or payload.get("sender_id", "unknown")
            message_id = x_message_id or payload.get("message_id", f"msg_{int(time.time())}")
            
            # Validate timestamp if provided (prevent replay attacks)
            if x_timestamp is not None and self.webhook_secret is not None:
                try:
                    timestamp = int(x_timestamp)
                    current_time = int(time.time())
                    # Allow 5-minute window
                    if abs(current_time - timestamp) > 300:
                        logger.warning(f"Webhook timestamp too old: {timestamp}")
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Timestamp expired",
                        )
                except ValueError:
                    pass  # Invalid timestamp format, ignore if not strictly enforcing
            
            logger.info(
                f"Received webhook message from {sender_id}: {message_text[:50]}..."
            )
            
            # Store message metadata for potential response routing
            metadata = payload.get("metadata", {})
            metadata.update({
                "message_id": message_id,
                "timestamp": x_timestamp,
                "received_at": int(time.time()),
            })
            
            # Process message asynchronously
            asyncio.create_task(
                self._process_incoming_message(sender_id, message_text, metadata)
            )
            
            return JSONResponse(
                content={
                    "status": "accepted",
                    "message_id": message_id,
                },
                status_code=status.HTTP_202_ACCEPTED,
            )
        
        @self.app.get("/health")
        async def health_check() -> JSONResponse:
            """Health check endpoint for the HTTP channel."""
            return JSONResponse(
                content={
                    "status": "healthy",
                    "channel": self.name,
                    "running": self._is_running,
                    "webhook_path": self.webhook_path,
                    "callback_configured": self.callback_url is not None,
                }
            )
        
        @self.app.get("/webhook/info")
        async def webhook_info() -> JSONResponse:
            """Get webhook configuration information."""
            return JSONResponse(
                content={
                    "webhook_url": f"http://{self.host}:{self.port}{self.webhook_path}",
                    "requires_signature": self.webhook_secret is not None,
                    "signature_header": "X-Webhook-Signature",
                    "supported_headers": [
                        "X-Webhook-Signature",
                        "X-Message-ID",
                        "X-Sender-ID",
                        "X-Timestamp",
                    ],
                }
            )
    
    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify HMAC-SHA256 signature of webhook payload.
        
        Args:
            body: Raw request body bytes.
            signature: Provided signature header value (hex-encoded).
            
        Returns:
            True if signature is valid, False otherwise.
        """
        if self.webhook_secret is None:
            return True
        
        try:
            expected_signature = hmac.new(
                key=self.webhook_secret.encode("utf-8"),
                msg=body,
                digestmod=hashlib.sha256,
            ).hexdigest()
            
            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(
                expected_signature.encode("utf-8"),
                signature.encode("utf-8"),
            )
        except Exception as exc:
            logger.error(f"Signature verification error: {exc}")
            return False
    
    def _generate_signature(self, payload: bytes) -> str:
        """Generate HMAC-SHA256 signature for outbound callback.
        
        Args:
            payload: JSON payload bytes to sign.
            
        Returns:
            Hex-encoded signature string.
        """
        if self.callback_secret is None:
            return ""
        
        return hmac.new(
            key=self.callback_secret.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
    
    async def _ensure_callback_client(self) -> Any:
        """Ensure HTTP client session exists for callbacks.
        
        Returns:
            aiohttp ClientSession instance.
        """
        if self._callback_client is None or self._callback_client.closed:
            try:
                import aiohttp
            except ImportError as exc:
                raise ImportError(
                    "aiohttp is required for HTTP callback functionality. "
                    "Install with: pip install aiohttp"
                ) from exc
            
            timeout = aiohttp.ClientTimeout(total=30)
            self._callback_client = aiohttp.ClientSession(timeout=timeout)
        
        return self._callback_client
    
    async def _process_incoming_message(
        self,
        sender_id: str,
        message: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Process an incoming webhook message.
        
        Routes to registered callback or agent for processing.
        
        Args:
            sender_id: Identifier of the message sender.
            message: Message content.
            metadata: Additional message metadata.
        """
        try:
            if self._message_callback is not None:
                await self._message_callback(sender_id, message)
            elif self.agent is not None:
                response = await self.agent.run(message)
                await self.send_message(sender_id, response)
            else:
                logger.warning(f"No handler configured for message from {sender_id}")
        except Exception as exc:
            logger.error(f"Error processing message from {sender_id}: {exc}")
    
    async def start(self) -> None:
        """Start the HTTP API server and begin listening for webhooks.
        
        Initializes the uvicorn server in a background task to handle
        incoming HTTP requests without blocking.
        """
        if self._is_running:
            logger.warning("HTTP API channel is already running")
            return
        
        if self.app is None:
            raise RuntimeError("FastAPI application not initialized")
        
        try:
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                loop="asyncio",
            )
            self._server = uvicorn.Server(config)
            
            # Run server in background task
            self._server_task = asyncio.create_task(self._server.serve())
            self._is_running = True
            self._shutdown_event.clear()
            
            logger.info(
                f"HTTP API channel started on http://{self.host}:{self.port}"
                f" (webhook: {self.webhook_path})"
            )
            
        except Exception as exc:
            logger.error(f"Failed to start HTTP API channel: {exc}")
            self._is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop the HTTP API server and clean up resources.
        
        Gracefully shuts down the uvicorn server and closes any
        active callback client connections.
        """
        if not self._is_running:
            logger.warning("HTTP API channel is not running")
            return
        
        self._shutdown_event.set()
        
        try:
            if self._server is not None:
                self._server.should_exit = True
            
            if self._server_task is not None:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("HTTP server shutdown timeout, forcing cancellation")
                    self._server_task.cancel()
                    try:
                        await self._server_task
                    except asyncio.CancelledError:
                        pass
            
            # Close callback client
            if self._callback_client is not None:
                await self._callback_client.close()
                self._callback_client = None
            
            self._is_running = False
            self._server = None
            self._server_task = None
            
            logger.info("HTTP API channel stopped successfully")
            
        except Exception as exc:
            logger.error(f"Error stopping HTTP API channel: {exc}")
            self._is_running = False
            raise
    
    async def send_message(self, to: str, content: str) -> bool:
        """Send a message to a specific user via HTTP POST callback.
        
        If callback_url is configured, sends an HTTP POST request with
        the message payload. Otherwise, logs a warning that no callback
        is configured.
        
        Args:
            to: Recipient identifier (user ID).
            content: Message text to send.
            
        Returns:
            True if message was sent successfully, False otherwise.
        """
        if not self._is_running:
            logger.error("Cannot send message: HTTP API channel is not running")
            return False
        
        # If no callback URL configured, we can't send outbound messages
        if self.callback_url is None:
            logger.warning(
                f"No callback_url configured, cannot send message to {to}. "
                f"Message would be: {content[:50]}..."
            )
            return False
        
        try:
            client = await self._ensure_callback_client()
            
            payload: Dict[str, Any] = {
                "recipient_id": to,
                "message": content,
                "timestamp": int(time.time()),
                "channel": self.name,
            }
            
            payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            
            headers: Dict[str, str] = {
                "Content-Type": "application/json",
                "X-Message-Type": "bot_response",
            }
            
            # Add signature if secret is configured
            if self.callback_secret is not None:
                signature = self._generate_signature(payload_bytes)
                headers["X-Callback-Signature"] = signature
            
            async with client.post(
                self.callback_url,
                data=payload_bytes,
                headers=headers,
            ) as response:
                if response.status == 200:
                    logger.debug(f"Callback sent successfully to {to}")
                    return True
                else:
                    response_text = await response.text()
                    logger.error(
                        f"Callback failed with status {response.status}: {response_text}"
                    )
                    return False
                    
        except Exception as exc:
            logger.error(f"Failed to send callback message to {to}: {exc}")
            return False
    
    def on_message(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        """Register a callback for incoming messages.
        
        Args:
            callback: Async function to call when a message is received.
                     Receives (sender_id: str, message: str) parameters.
        """
        self._message_callback = callback
        logger.debug("Message callback registered for HTTP API channel")
    
    def get_webhook_url(self) -> str:
        """Get the full webhook URL for this channel.
        
        Returns:
            Complete URL string for the webhook endpoint.
        """
        return f"http://{self.host}:{self.port}{self.webhook_path}"
    
    def __repr__(self) -> str:
        status = "running" if self._is_running else "stopped"
        callback_info = f", callback={self.callback_url is not None}"
        secured = f", secure={self.webhook_secret is not None}"
        return f"HttpApiChannel({self.host}:{self.port}{self.webhook_path}, {status}{callback_info}{secured})"