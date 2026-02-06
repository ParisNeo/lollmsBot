"""
Web UI package for LollmsBot.

Provides a beautiful local web interface for interacting with the AI agent.
Includes real-time chat, conversation history, and tool execution visualization.
"""

from lollmsbot.ui.app import WebUI
from lollmsbot.ui.routes import ui_router

__all__ = ["WebUI", "ui_router"]
