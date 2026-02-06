"""
Telegram channel implementation for LollmsBot.

Uses shared Agent for all business logic.
"""

import logging
from typing import Any, Awaitable, Callable, List, Optional, Set

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        ApplicationBuilder,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    TELEGRAM_AVAILABLE = True
except ImportError as e:
    TELEGRAM_AVAILABLE = False
    TELEGRAM_IMPORT_ERROR = str(e)
    # Create dummy classes for type checking
    class Update: pass
    class Application: pass
    class ApplicationBuilder: pass
    class CommandHandler: pass
    class ContextTypes:
        DEFAULT_TYPE = Any
    class MessageHandler: pass
    class filters:
        TEXT = None
        COMMAND = None

from lollmsbot.agent import Agent, PermissionLevel


logger = logging.getLogger(__name__)


class TelegramChannel:
    """Telegram messaging channel using shared Agent.
    
    All business logic is delegated to the Agent. This class handles
    only Telegram-specific protocol concerns.
    """

    def __init__(
        self,
        agent: Agent,
        bot_token: str,
        allowed_users: Optional[List[int]] = None,
        blocked_users: Optional[List[int]] = None,
    ):
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                f"Telegram support requires 'python-telegram-bot'. "
                f"Install with: pip install 'python-telegram-bot>=20.0' "
                f"Original error: {TELEGRAM_IMPORT_ERROR}"
            )
        
        self.agent = agent
        self.bot_token = bot_token
        self.allowed_users: Optional[Set[int]] = set(allowed_users) if allowed_users else None
        self.blocked_users: Set[int] = set(blocked_users) if blocked_users else set()
        self.application: Optional[Application] = None
        self._is_running = False

    def _can_interact(self, user_id: int) -> tuple[bool, str]:
        """Check if user can interact with bot."""
        if user_id in self.blocked_users:
            return False, "user blocked"
        
        if self.allowed_users is not None:
            if user_id not in self.allowed_users:
                return False, "not in allowed users"
        
        return True, ""

    def _get_user_id(self, tg_user_id: int) -> str:
        """Generate consistent user ID for Agent."""
        return f"telegram:{tg_user_id}"

    async def start(self) -> None:
        """Start the Telegram bot."""
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot is not installed")
        
        if self._is_running:
            logger.warning("Telegram channel is already running")
            return
        
        try:
            self.application = (
                ApplicationBuilder()
                .token(self.bot_token)
                .build()
            )
            
            # Add handlers
            self.application.add_handler(
                CommandHandler("start", self._handle_start_command)
            )
            self.application.add_handler(
                CommandHandler("help", self._handle_help_command)
            )
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
            )
            self.application.add_error_handler(self._handle_error)
            
            # Start
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling(drop_pending_updates=True)
            
            self._is_running = True
            logger.info("Telegram channel started successfully")
            
        except Exception as exc:
            logger.error(f"Failed to start Telegram channel: {exc}")
            self._is_running = False
            raise

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if not self._is_running:
            return
        
        try:
            if self.application:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
                self.application = None
            
            self._is_running = False
            logger.info("Telegram channel stopped successfully")
            
        except Exception as exc:
            logger.error(f"Error stopping Telegram channel: {exc}")
            raise

    async def _handle_start_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /start command."""
        if not update.effective_chat or not update.message:
            return
        
        user_id = update.effective_user.id if update.effective_user else 0
        
        # Check permissions
        can_interact, reason = self._can_interact(user_id)
        if not can_interact:
            await update.message.reply_text("⛔ You don't have permission to use this bot.")
            return
        
        # Get or create user permissions in agent
        agent_user_id = self._get_user_id(user_id)
        
        welcome_text = (
            f"👋 Hello! I'm {self.agent.name}.\n\n"
            f"Send me any message and I'll respond using my AI backend.\n\n"
            f"Your ID: `{user_id}`\n"
            f"Use /help for more information."
        )
        
        await update.message.reply_text(welcome_text, parse_mode="Markdown")

    async def _handle_help_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle /help command."""
        help_text = (
            f"*{self.agent.name} - Available Commands*\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n\n"
            "Just send any message to chat with me!"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def _handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming text messages."""
        if not update.effective_chat or not update.message or not update.message.text:
            return
        
        user_id = update.effective_user.id if update.effective_user else 0
        chat_id = update.effective_chat.id
        
        # Check permissions
        can_interact, reason = self._can_interact(user_id)
        if not can_interact:
            logger.warning(f"Message from unauthorized user {user_id}: {reason}")
            await update.message.reply_text("⛔ Access denied.")
            return
        
        # Build context
        agent_user_id = self._get_user_id(user_id)
        context_data = {
            "channel": "telegram",
            "telegram_user_id": user_id,
            "telegram_username": update.effective_user.username if update.effective_user else None,
            "telegram_chat_id": chat_id,
            "is_dm": update.effective_chat.type == "private",
            "chat_type": update.effective_chat.type,
        }
        
        message_text = update.message.text
        
        logger.info(f"Processing message from Telegram user {user_id}: {message_text[:50]}...")
        
        try:
            # Use Agent for processing
            result = await self.agent.chat(
                user_id=agent_user_id,
                message=message_text,
                context=context_data,
            )
            
            if result.get("permission_denied"):
                await update.message.reply_text("⛔ You don't have permission to use this bot.")
                return
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                await update.message.reply_text(f"❌ Error: {error_msg[:400]}")
                return
            
            response = result.get("response", "No response")
            # Telegram has 4096 char limit
            if len(response) > 4000:
                response = response[:4000] + "\n... (truncated)"
            
            await update.message.reply_text(response)
            
        except Exception as exc:
            logger.error(f"Error processing Telegram message: {exc}")
            try:
                await update.message.reply_text("❌ An error occurred. Please try again.")
            except Exception:
                pass

    async def _handle_error(
        self,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle errors."""
        logger.error(f"Telegram bot error: {context.error}")
        
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ An unexpected error occurred."
                )
            except Exception:
                pass

    def __repr__(self) -> str:
        status = "running" if self._is_running else "stopped"
        restricted = f", restricted={len(self.allowed_users)} users" if self.allowed_users else ""
        return f"TelegramChannel({status}{restricted})"
