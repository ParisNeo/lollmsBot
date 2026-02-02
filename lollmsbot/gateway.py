"""
Gateway module for LollmsBot.

This module provides the Gateway class that serves as the main orchestrator
for the LollmsBot framework, handling HTTP REST endpoints, WebSocket connections,
and message routing to Agent instances.
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.responses import JSONResponse

from lollmsbot.agent import Agent
from lollmsbot.config import Settings, load_settings


logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Enumeration of supported communication channel types."""
    HTTP = auto()
    WEBSOCKET = auto()
    TELEGRAM = auto()
    DISCORD = auto()


@dataclass
class Channel:
    """Represents a registered communication channel."""
    channel_id: str
    channel_type: ChannelType
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    websocket: Optional[WebSocket] = None


@dataclass
class ChatRequest:
    """Request model for chat endpoint."""
    message: str
    agent_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Response model for chat endpoint."""
    response: str
    agent_id: str
    message_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GatewayError(Exception):
    """Base exception for gateway errors."""
    pass


class ChannelNotFoundError(GatewayError):
    """Raised when a channel is not found."""
    pass


class AgentNotFoundError(GatewayError):
    """Raised when an agent is not found."""
    pass


class Gateway:
    """Main orchestrator for LollmsBot framework.
    
    The Gateway class manages the FastAPI application, handles REST endpoints,
    WebSocket connections, and routes messages to appropriate Agent instances.
    It provides a unified interface for multi-channel AI agent communication.
    
    Attributes:
        app: FastAPI application instance.
        settings: Application configuration settings.
        agents: Dictionary of registered Agent instances by ID.
        channels: Dictionary of registered Channel instances by ID.
        active_connections: Set of active WebSocket connections.
        _server: Optional uvicorn server instance.
        _task: Optional asyncio task for running the server.
        _running: Boolean indicating if gateway is running.
        
    Example:
        >>> gateway = Gateway(settings=load_settings())
        >>> gateway.register_agent(agent)
        >>> await gateway.start()
        >>> # Gateway now accepts HTTP and WebSocket connections
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        """Initialize the Gateway.
        
        Args:
            settings: Optional application settings (loaded from env if not provided).
            host: Host address to bind the server to.
            port: Port number to listen on.
        """
        self.settings: Settings = settings or load_settings()
        self.host: str = host
        self.port: int = port
        
        self.agents: Dict[str, Agent] = {}
        self.channels: Dict[str, Channel] = {}
        self.active_connections: Set[WebSocket] = set()
        
        self._server: Optional[uvicorn.Server] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        
        # Initialize FastAPI app with lifespan
        self.app: FastAPI = FastAPI(
            title="LollmsBot Gateway",
            description="Multi-channel AI agent platform API",
            version="0.1.0",
            lifespan=self._lifespan,
        )
        
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Configure FastAPI routes and endpoints."""
        
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest) -> ChatResponse:
            """Process a chat message via HTTP REST API."""
            return await self._handle_chat(request)
        
        @self.app.get("/health")
        async def health_endpoint() -> Dict[str, Any]:
            """Health check endpoint."""
            return await self._handle_health()
        
        @self.app.get("/agents")
        async def agents_endpoint() -> List[Dict[str, Any]]:
            """List all registered agents."""
            return await self._handle_agents_list()
        
        @self.app.get("/agents/{agent_id}")
        async def agent_detail_endpoint(agent_id: str) -> Dict[str, Any]:
            """Get details for a specific agent."""
            agent = self.agents.get(agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent '{agent_id}' not found",
                )
            return {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "state": agent.state.name,
                "tools_count": len(agent.tools),
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time communication."""
            await self._handle_websocket(websocket)
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager for startup and shutdown events.
        
        Handles initialization and cleanup of gateway resources.
        """
        # Startup
        logger.info("LollmsBot Gateway starting up...")
        self._running = True
        self._shutdown_event.clear()
        
        # Initialize default channel
        default_channel = Channel(
            channel_id=str(uuid.uuid4()),
            channel_type=ChannelType.HTTP,
            name="default_http",
        )
        self.channels[default_channel.channel_id] = default_channel
        
        logger.info(f"Gateway initialized on {self.host}:{self.port}")
        logger.info(f"Default channel created: {default_channel.channel_id}")
        
        yield
        
        # Shutdown
        logger.info("LollmsBot Gateway shutting down...")
        self._running = False
        self._shutdown_event.set()
        
        # Close all WebSocket connections
        close_tasks = [
            conn.close(code=1001, reason="Server shutting down")
            for conn in self.active_connections
        ]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        self.active_connections.clear()
        
        # Clear channels
        self.channels.clear()
        
        logger.info("Gateway shutdown complete")
    
    async def _handle_chat(self, request: ChatRequest) -> ChatResponse:
        """Handle chat request from HTTP endpoint.
        
        Args:
            request: ChatRequest containing message and optional agent_id.
            
        Returns:
            ChatResponse with generated response.
            
        Raises:
            HTTPException: If agent not found or processing fails.
        """
        message_id = str(uuid.uuid4())
        
        # Select agent (use first available if not specified, or specified agent)
        agent: Optional[Agent] = None
        if request.agent_id:
            agent = self.agents.get(request.agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent '{request.agent_id}' not found",
                )
        elif self.agents:
            agent = next(iter(self.agents.values()))
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No agents available",
            )
        
        try:
            # Route message through gateway
            response_text = await self.route_message("http", request.message, agent)
            
            return ChatResponse(
                response=response_text,
                agent_id=agent.agent_id,
                message_id=message_id,
                metadata={
                    "channel": "http",
                    "context": request.context or {},
                },
            )
            
        except Exception as exc:
            logger.error(f"Chat processing failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processing failed: {str(exc)}",
            )
    
    async def _handle_health(self) -> Dict[str, Any]:
        """Handle health check request.
        
        Returns:
            Dictionary with health status and gateway information.
        """
        agents_status = [
            {
                "id": agent.agent_id,
                "name": agent.name,
                "state": agent.state.name,
            }
            for agent in self.agents.values()
        ]
        
        return {
            "status": "healthy" if self._running else "unhealthy",
            "gateway": {
                "host": self.host,
                "port": self.port,
                "running": self._running,
            },
            "agents": {
                "count": len(self.agents),
                "agents": agents_status,
            },
            "channels": {
                "count": len(self.channels),
                "types": list(set(ch.channel_type.name for ch in self.channels.values())),
            },
            "websocket_connections": len(self.active_connections),
            "timestamp": datetime.now().isoformat(),
        }
    
    async def _handle_agents_list(self) -> List[Dict[str, Any]]:
        """Handle request to list all registered agents.
        
        Returns:
            List of agent information dictionaries.
        """
        return [
            {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "state": agent.state.name,
                "tools_count": len(agent.tools),
                "memory": {
                    "history_length": len(agent.memory.get("conversation_history", [])),
                },
            }
            for agent in self.agents.values()
        ]
    
    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection for real-time communication.
        
        Args:
            websocket: FastAPI WebSocket instance.
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Create channel for this connection
        channel = Channel(
            channel_id=str(uuid.uuid4()),
            channel_type=ChannelType.WEBSOCKET,
            name=f"ws_{websocket.client.host if websocket.client else 'unknown'}",
            websocket=websocket,
        )
        self.channels[channel.channel_id] = channel
        
        logger.info(f"WebSocket connected: {channel.channel_id}")
        
        try:
            while self._running:
                # Receive message with timeout to allow graceful shutdown checks
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                try:
                    message_data = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "error": "Invalid JSON",
                        "type": "error",
                    }))
                    continue
                
                message = message_data.get("message", "")
                agent_id = message_data.get("agent_id")
                
                # Select agent
                agent: Optional[Agent] = None
                if agent_id:
                    agent = self.agents.get(agent_id)
                if not agent and self.agents:
                    agent = next(iter(self.agents.values()))
                
                if not agent:
                    await websocket.send_text(json.dumps({
                        "error": "No agents available",
                        "type": "error",
                    }))
                    continue
                
                # Process message and stream response
                try:
                    response_text = await self.route_message(
                        f"websocket:{channel.channel_id}",
                        message,
                        agent,
                    )
                    
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "message": response_text,
                        "agent_id": agent.agent_id,
                        "timestamp": datetime.now().isoformat(),
                    }))
                    
                except Exception as exc:
                    logger.error(f"WebSocket message processing failed: {exc}")
                    await websocket.send_text(json.dumps({
                        "error": str(exc),
                        "type": "error",
                    }))
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {channel.channel_id}")
        except Exception as exc:
            logger.error(f"WebSocket error: {exc}")
        finally:
            self.active_connections.discard(websocket)
            self.channels.pop(channel.channel_id, None)
    
    async def start(self) -> None:
        """Start the Gateway server.
        
        This method starts the uvicorn server in a background task,
        allowing the gateway to run concurrently with other asyncio code.
        """
        if self._running:
            logger.warning("Gateway is already running")
            return
        
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio",
        )
        self._server = uvicorn.Server(config)
        
        # Run server in background task
        self._task = asyncio.create_task(self._server.serve())
        self._running = True
        
        logger.info(f"Gateway server started on {self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the Gateway server gracefully.
        
        Initiates shutdown sequence and waits for server to stop.
        """
        if not self._running:
            logger.warning("Gateway is not running")
            return
        
        self._running = False
        self._shutdown_event.set()
        
        if self._server:
            self._server.should_exit = True
        
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Gateway shutdown timeout, forcing cancellation")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        
        self._server = None
        self._task = None
        
        logger.info("Gateway server stopped")
    
    def register_agent(self, agent: Agent) -> str:
        """Register an Agent instance with the gateway.
        
        Args:
            agent: Agent instance to register.
            
        Returns:
            The agent_id of the registered agent.
        """
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent registered: {agent.agent_id} ({agent.name})")
        return agent.agent_id
    
    def unregister_agent(self, agent_id: str) -> Optional[Agent]:
        """Unregister an Agent instance from the gateway.
        
        Args:
            agent_id: ID of the agent to unregister.
            
        Returns:
            The removed Agent if found, None otherwise.
        """
        agent = self.agents.pop(agent_id, None)
        if agent:
            logger.info(f"Agent unregistered: {agent_id}")
        return agent
    
    def register_channel(
        self,
        name: str,
        channel_type: ChannelType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a new communication channel.
        
        Args:
            name: Display name for the channel.
            channel_type: Type of channel (HTTP, WEBSOCKET, etc.).
            metadata: Optional additional channel data.
            
        Returns:
            The channel_id of the registered channel.
        """
        channel_id = str(uuid.uuid4())
        channel = Channel(
            channel_id=channel_id,
            channel_type=channel_type,
            name=name,
            metadata=metadata or {},
        )
        self.channels[channel_id] = channel
        logger.info(f"Channel registered: {channel_id} ({name}, {channel_type.name})")
        return channel_id
    
    def unregister_channel(self, channel_id: str) -> Optional[Channel]:
        """Unregister a channel from the gateway.
        
        Args:
            channel_id: ID of the channel to unregister.
            
        Returns:
            The removed Channel if found, None otherwise.
        """
        channel = self.channels.pop(channel_id, None)
        if channel:
            logger.info(f"Channel unregistered: {channel_id}")
        return channel
    
    async def route_message(
        self,
        source: str,
        message: str,
        agent: Optional[Agent] = None,
    ) -> str:
        """Route a message to an Agent for processing.
        
        This is the core message routing method that directs incoming
        messages from any source to the appropriate agent.
        
        Args:
            source: Identifier for the message source (e.g., "http", "websocket:id").
            message: The message content to process.
            agent: Optional specific agent to use (uses first available if None).
            
        Returns:
            The generated response string from the agent.
            
        Raises:
            AgentNotFoundError: If no agent is available to process the message.
        """
        target_agent = agent
        if target_agent is None:
            if not self.agents:
                raise AgentNotFoundError("No agents available to process message")
            target_agent = next(iter(self.agents.values()))
        
        logger.debug(f"Routing message from '{source}' to agent '{target_agent.agent_id}'")
        
        # Process through agent
        response = await target_agent.run(message)
        
        return response
    
    @property
    def is_running(self) -> bool:
        """Check if the gateway is currently running."""
        return self._running
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get a registered agent by ID.
        
        Args:
            agent_id: Agent identifier.
            
        Returns:
            The Agent instance if found, None otherwise.
        """
        return self.agents.get(agent_id)
    
    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get a registered channel by ID.
        
        Args:
            channel_id: Channel identifier.
            
        Returns:
            The Channel instance if found, None otherwise.
        """
        return self.channels.get(channel_id)
    
    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"Gateway({self.host}:{self.port}, {status}, agents={len(self.agents)}, channels={len(self.channels)})"