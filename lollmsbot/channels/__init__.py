"""Channel adapters for LollmsBot.

All channels use the shared Agent for business logic.
This package provides channel implementations for various messaging platforms.
"""

from lollmsbot.channels.discord import DiscordChannel
from lollmsbot.channels.http_api import HttpApiChannel

# Telegram is optional - import if available
try:
    from lollmsbot.channels.telegram import TelegramChannel
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    TelegramChannel = None  # type: ignore

# WhatsApp is optional - import if dependencies available
try:
    from lollmsbot.channels.whatsapp import WhatsAppChannel
    WHATSAPP_AVAILABLE = True
except ImportError:
    WHATSAPP_AVAILABLE = False
    WhatsAppChannel = None  # type: ignore

__all__ = [
    "DiscordChannel",
    "HttpApiChannel",
]

if TELEGRAM_AVAILABLE:
    __all__.append("TelegramChannel")

if WHATSAPP_AVAILABLE:
    __all__.append("WhatsAppChannel")
