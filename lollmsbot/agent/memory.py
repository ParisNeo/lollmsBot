"""
Agent memory management system.

Handles conversation history, user-specific memories, and important fact storage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lollmsbot.agent_config import ConversationTurn


class MemoryManager:
    """Manages agent memory including conversation history and important facts."""
    
    def __init__(self, max_history: int = 10) -> None:
        self._max_history = max_history
        self._conversation_history: List[ConversationTurn] = []
        self._user_histories: Dict[str, List[ConversationTurn]] = {}
        self._working_memory: Dict[str, Any] = {}
        self._context: Dict[str, Any] = {}
        self._important_facts: Dict[str, Any] = {}
        self._memory_lock: asyncio.Lock = asyncio.Lock()
    
    @property
    def conversation_history(self) -> List[ConversationTurn]:
        return list(self._conversation_history)
    
    def get_user_history(self, user_id: str) -> List[ConversationTurn]:
        """Get conversation history for a specific user."""
        return self._user_histories.get(user_id, [])
    
    async def add_to_user_history(self, user_id: str, turn: ConversationTurn) -> None:
        """Add a conversation turn to user's history."""
        async with self._memory_lock:
            if user_id not in self._user_histories:
                self._user_histories[user_id] = []
            
            self._user_histories[user_id].append(turn)
            
            # Trim to max history
            while len(self._user_histories[user_id]) > self._max_history:
                self._user_histories[user_id].pop(0)
    
    def get_important_facts(self, key: Optional[str] = None) -> Any:
        """Get important facts, optionally filtered by key."""
        if key:
            return self._important_facts.get(key)
        return dict(self._important_facts)
    
    async def store_important_fact(
        self,
        key: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> None:
        """Store an important fact with metadata."""
        async with self._memory_lock:
            entry = {
                "data": data,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "access_count": 0,
                "importance": 10.0,
                "tags": ["identity", "high_value"],
            }
            
            if user_id:
                user_key = f"user_{user_id}_facts"
                if user_key not in self._important_facts:
                    self._important_facts[user_key] = []
                self._important_facts[user_key].append(entry)
            
            # Store globally if it's creator identity
            if key == "creator_identity":
                self._important_facts[key] = {
                    **data,
                    "confirmed_by": user_id,
                    "confirmed_at": datetime.now().isoformat(),
                    "trust_level": "high",
                }
    
    def get_working_memory(self) -> Dict[str, Any]:
        """Get current working memory."""
        return dict(self._working_memory)
    
    def set_working_memory(self, key: str, value: Any) -> None:
        """Set a value in working memory."""
        self._working_memory[key] = value
    
    def clear_working_memory(self) -> None:
        """Clear all working memory."""
        self._working_memory.clear()
    
    def format_history_for_prompt(
        self,
        history: List[ConversationTurn],
        max_turns: int = 10,
    ) -> str:
        """Format conversation history for LLM prompt."""
        parts = ["=== CONVERSATION HISTORY ===", ""]
        
        # Add relevant history (last N turns)
        for turn in history[-max_turns:]:
            parts.append(f"User: {turn.user_message}")
            parts.append(f"Assistant: {turn.agent_response}")
            parts.append("")
        
        return "\n".join(parts)
