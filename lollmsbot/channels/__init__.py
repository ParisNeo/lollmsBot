"""
Channels package for LollmsBot.

This package provides communication channel implementations for different
platforms including Telegram, Discord, and HTTP API. It defines a common
interface through the Channel base class and a registry for managing
multiple channels simultaneously.

The channels module enables LollmsBot to operate across multiple messaging
platforms with a unified interface for message handling and delivery.

Example:
    >>> from lollmsbot.channels import TelegramChannel, ChannelRegistry
    >>> registry = ChannelRegistry()
    >>> telegram = TelegramChannel(token="...")
    >>> registry.register("telegram", telegram)
    >>> registry.start_all()
"""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from lollmsbot.agent import Agent


class Channel(ABC):
    """Abstract base class for communication channels.
    
    All channel implementations must inherit from this class and implement
    the required abstract methods. Channels provide a unified interface
    for connecting to different messaging platforms.
    
    Attributes:
        name: Human-readable name of the channel.
        is_running: Whether the channel is currently active.
        agent: Optional Agent instance for handling messages.
    """
    
    def __init__(
        self,
        name: str,
        agent: Optional[Agent] = None,
    ) -> None:
        """Initialize the channel.
        
        Args:
            name: Human-readable name for this channel instance.
            agent: Optional Agent to process incoming messages.
        """
        self.name: str = name
        self.agent: Optional[Agent] = agent
        self._is_running: bool = False
        self._message_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
    
    @property
    def is_running(self) -> bool:
        """Check if the channel is currently running."""
        return self._is_running
    
    @abstractmethod
    async def start(self) -> None:
        """Start the channel and begin listening for messages.
        
        This method should establish connections to the messaging
        platform and begin processing incoming messages.
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources.
        
        This method should gracefully shut down the channel,
        closing connections and stopping message processing.
        """
        pass
    
    @abstractmethod
    async def send_message(self, to: str, content: str) -> bool:
        """Send a message to a specific recipient.
        
        Args:
            to: Recipient identifier (user ID, chat ID, etc.).
            content: Message content to send.
            
        Returns:
            True if message was sent successfully, False otherwise.
        """
        pass
    
    def on_message(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        """Register a callback for incoming messages.
        
        Args:
            callback: Async function to call when a message is received.
                     Should accept (sender_id: str, message: str) parameters.
        """
        self._message_callback = callback
    
    async def _handle_incoming_message(self, sender_id: str, message: str) -> None:
        """Internal handler for incoming messages.
        
        Routes messages to the registered callback if available,
        or to the agent if configured.
        
        Args:
            sender_id: Identifier of the message sender.
            message: Message content.
        """
        if self._message_callback is not None:
            await self._message_callback(sender_id, message)
        elif self.agent is not None:
            response = await self.agent.run(message)
            await self.send_message(sender_id, response)
    
    def __repr__(self) -> str:
        return f"Channel({self.name}, running={self._is_running})"


class ChannelRegistry:
    """Registry for managing multiple communication channels.
    
    Provides centralized management for multiple channel instances,
    allowing bulk operations like starting/stopping all channels.
    
    Attributes:
        channels: Dictionary of registered channels by name.
    """
    
    def __init__(self) -> None:
        """Initialize an empty channel registry."""
        self._channels: Dict[str, Channel] = {}
    
    @property
    def channels(self) -> Dict[str, Channel]:
        """Get a copy of registered channels."""
        return self._channels.copy()
    
    def register(self, name: str, channel: Channel) -> None:
        """Register a channel with the given name.
        
        Args:
            name: Unique identifier for the channel.
            channel: Channel instance to register.
            
        Raises:
            ValueError: If a channel with this name is already registered.
        """
        if name in self._channels:
            raise ValueError(f"Channel '{name}' is already registered")
        self._channels[name] = channel
    
    def unregister(self, name: str) -> Optional[Channel]:
        """Remove a channel from the registry.
        
        Args:
            name: Name of the channel to remove.
            
        Returns:
            The removed channel if found, None otherwise.
        """
        return self._channels.pop(name, None)
    
    def get(self, name: str) -> Optional[Channel]:
        """Get a registered channel by name.
        
        Args:
            name: Channel name to look up.
            
        Returns:
            The channel if found, None otherwise.
        """
        return self._channels.get(name)
    
    async def start_all(self) -> Dict[str, bool]:
        """Start all registered channels.
        
        Returns:
            Dictionary mapping channel names to success status.
        """
        results: Dict[str, bool] = {}
        for name, channel in self._channels.items():
            try:
                await channel.start()
                results[name] = True
            except Exception:
                results[name] = False
        return results
    
    async def stop_all(self) -> Dict[str, bool]:
        """Stop all registered channels.
        
        Returns:
            Dictionary mapping channel names to success status.
        """
        results: Dict[str, bool] = {}
        for name, channel in self._channels.items():
            try:
                await channel.stop()
                results[name] = True
            except Exception:
                results[name] = False
        return results
    
    def list_channels(self) -> List[str]:
        """Get list of registered channel names.
        
        Returns:
            List of channel names.
        """
        return list(self._channels.keys())
    
    def __len__(self) -> int:
        """Return number of registered channels."""
        return len(self._channels)
    
    def __repr__(self) -> str:
        return f"ChannelRegistry({list(self._channels.keys())})"


# Placeholder imports for channel implementations
# These will be defined in separate module files
class TelegramChannel(Channel):
    """Telegram messaging channel implementation."""
    
    def __init__(self, token: str, **kwargs: Any) -> None:
        super().__init__(name="telegram", **kwargs)
        self.token: str = token
    
    async def start(self) -> None:
        self._is_running = True
    
    async def stop(self) -> None:
        self._is_running = False
    
    async def send_message(self, to: str, content: str) -> bool:
        return True


class DiscordChannel(Channel):
    """Discord messaging channel implementation."""
    
    def __init__(self, token: str, **kwargs: Any) -> None:
        super().__init__(name="discord", **kwargs)
        self.token: str = token
    
    async def start(self) -> None:
        self._is_running = True
    
    async def stop(self) -> None:
        self._is_running = False
    
    async def send_message(self, to: str, content: str) -> bool:
        return True


class HttpApiChannel(Channel):
    """HTTP API channel implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 8080, **kwargs: Any) -> None:
        super().__init__(name="http_api", **kwargs)
        self.host: str = host
        self.port: int = port
    
    async def start(self) -> None:
        self._is_running = True
    
    async def stop(self) -> None:
        self._is_running = False
    
    async def send_message(self, to: str, content: str) -> bool:
        return True


__all__ = [
    "Channel",
    "ChannelRegistry",
    "TelegramChannel",
    "DiscordChannel",
    "HttpApiChannel",
]