"""Channel adapters for LollmsBot.

All channels use the shared Agent for business logic.
This package provides channel implementations for various messaging platforms.
"""

from lollmsbot.channels.discord import DiscordChannel
from lollmsbot.channels.http_api import HttpApiChannel

# Telegram is optional - import if available
try:
    from lollmsbot.channels.telegram import TelegramChannel
    __all__ = [
        "DiscordChannel",
        "TelegramChannel",
        "HttpApiChannel",
    ]
except ImportError:
    __all__ = [
        "DiscordChannel",
        "HttpApiChannel",
    ]
