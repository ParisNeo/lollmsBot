"""
Storage package for LollmsBot.

This package provides persistence layer for conversation history and agent state.
It includes pluggable storage backends with SQLite as the default implementation,
allowing easy swapping for PostgreSQL, Redis, or custom storage solutions.

The storage layer supports:
- Conversation persistence per user with full message history
- Agent state serialization for checkpoint/resume capabilities
- Multiple backend registration and runtime selection

Example:
    >>> from lollmsbot.storage import StorageRegistry, SqliteStore
    >>> # Register default SQLite backend
    >>> StorageRegistry.register("sqlite", SqliteStore("bot.db"))
    >>> # Use in agent or gateway
    >>> store = StorageRegistry.get("sqlite")
    >>> store.save_conversation("user123", messages)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from lollmsbot.storage.sqlite_store import SqliteStore


class BaseStorage(ABC):
    """Abstract base class for storage backends.
    
    All storage implementations must inherit from this class and implement
    the core persistence methods for conversations and agent state.
    
    The storage backend is responsible for:
    - Persisting conversation history per user
    - Saving and loading agent state for resumption
    - Handling connection lifecycle and error conditions
    
    Attributes:
        backend_name: Identifier for the storage backend type.
    """
    
    backend_name: str = "abstract"
    
    @abstractmethod
    async def save_conversation(self, user_id: str, messages: List[Dict[str, Any]]) -> bool:
        """Save or append conversation messages for a user.
        
        Args:
            user_id: Unique identifier for the user.
            messages: List of message dictionaries with 'role', 'content', 'timestamp' keys.
            
        Returns:
            True if save was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def get_conversation(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a user.
        
        Args:
            user_id: Unique identifier for the user.
            limit: Maximum number of recent messages to retrieve (None for all).
            
        Returns:
            List of message dictionaries ordered chronologically.
        """
        pass
    
    @abstractmethod
    async def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Serialize and save agent state for later resumption.
        
        Args:
            agent_id: Unique identifier for the agent.
            state: Dictionary containing all serializable agent state.
            
        Returns:
            True if save was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load previously saved agent state.
        
        Args:
            agent_id: Unique identifier for the agent.
            
        Returns:
            State dictionary if found, None otherwise.
        """
        pass
    
    @abstractmethod
    async def delete_conversation(self, user_id: str) -> bool:
        """Delete all conversation history for a user.
        
        Args:
            user_id: Unique identifier for the user.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def delete_agent_state(self, agent_id: str) -> bool:
        """Delete saved state for an agent.
        
        Args:
            agent_id: Unique identifier for the agent.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close storage connections and cleanup resources."""
        pass


class StorageRegistry:
    """Registry for managing multiple storage backends.
    
    Provides a centralized registry pattern for storage backend management,
    allowing runtime selection and failover between different storage
    implementations (SQLite, PostgreSQL, Redis, etc.).
    
    The registry supports:
    - Named backend registration
    - Runtime backend retrieval
    - Default backend assignment
    - Bulk cleanup on shutdown
    
    Example:
        >>> StorageRegistry.register("primary", SqliteStore("primary.db"))
        >>> StorageRegistry.register("cache", RedisStore("localhost:6379"))
        >>> StorageRegistry.set_default("primary")
        >>> store = StorageRegistry.get_default()
    """
    
    _backends: Dict[str, BaseStorage] = {}
    _default: Optional[str] = None
    
    @classmethod
    def register(cls, name: str, backend: BaseStorage) -> None:
        """Register a storage backend with a unique name.
        
        Args:
            name: Unique identifier for this backend instance.
            backend: Configured storage backend instance.
            
        Raises:
            ValueError: If name is already registered.
        """
        if name in cls._backends:
            raise ValueError(f"Storage backend '{name}' is already registered")
        cls._backends[name] = backend
        
        # Set as default if first registration
        if cls._default is None:
            cls._default = name
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseStorage]:
        """Retrieve a registered storage backend by name.
        
        Args:
            name: Backend identifier used during registration.
            
        Returns:
            Storage backend instance or None if not found.
        """
        return cls._backends.get(name)
    
    @classmethod
    def get_default(cls) -> Optional[BaseStorage]:
        """Retrieve the default storage backend.
        
        Returns:
            Default backend instance or None if none set.
        """
        if cls._default is None:
            return None
        return cls._backends.get(cls._default)
    
    @classmethod
    def set_default(cls, name: str) -> None:
        """Set the default storage backend.
        
        Args:
            name: Name of registered backend to set as default.
            
        Raises:
            ValueError: If backend is not registered.
        """
        if name not in cls._backends:
            raise ValueError(f"Storage backend '{name}' is not registered")
        cls._default = name
    
    @classmethod
    def unregister(cls, name: str) -> Optional[BaseStorage]:
        """Remove a registered backend and cleanup.
        
        Args:
            name: Backend identifier to remove.
            
        Returns:
            Removed backend instance or None if not found.
        """
        backend = cls._backends.pop(name, None)
        if backend and cls._default == name:
            # Clear default if removed, set to another if available
            cls._default = next(iter(cls._backends), None)
        return backend
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend names.
        
        Returns:
            List of registered backend identifiers.
        """
        return list(cls._backends.keys())
    
    @classmethod
    async def close_all(cls) -> None:
        """Close all registered storage backends."""
        for backend in cls._backends.values():
            await backend.close()
        cls._backends.clear()
        cls._default = None


__all__ = [
    "BaseStorage",
    "SqliteStore",
    "StorageRegistry",
]