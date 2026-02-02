"""
Telegram channel implementation for LollmsBot.

This module provides the TelegramChannel class for integrating with the
Telegram messaging platform using the python-telegram-bot library.
"""

import logging
from typing import Any, Awaitable, Callable, List, Optional, Set

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from lollmsbot.channels import Channel
from lollmsbot.agent import Agent


logger = logging.getLogger(__name__)


class TelegramChannel(Channel):
    """Telegram messaging channel implementation.
    
    Provides integration with Telegram Bot API using python-telegram-bot
    library with async support. Handles message polling and sending.
    
    Attributes:
        bot_token: Telegram bot authentication token.
        allowed_chat_ids: Optional set of allowed chat IDs for security.
        application: The python-telegram-bot Application instance.
    """
    
    def __init__(
        self,
        bot_token: str,
        allowed_chat_ids: Optional[List[int]] = None,
        agent: Optional[Agent] = None,
    ) -> None:
        """Initialize the Telegram channel.
        
        Args:
            bot_token: Telegram bot token from @BotFather.
            allowed_chat_ids: Optional list of allowed chat IDs.
                              If provided, only these chats can interact.
            agent: Optional Agent instance for message processing.
        """
        super().__init__(name="telegram", agent=agent)
        
        self.bot_token: str = bot_token
        self.allowed_chat_ids: Optional[Set[int]] = (
            set(allowed_chat_ids) if allowed_chat_ids else None
        )
        self.application: Optional[Application] = None
        self._message_callback: Optional[
            Callable[[str, str], Awaitable[None]]
        ] = None
    
    async def start(self) -> None:
        """Start the Telegram bot and begin polling for messages.
        
        Initializes the Application, sets up handlers, and starts
        the polling loop to receive updates from Telegram.
        """
        if self._is_running:
            logger.warning("Telegram channel is already running")
            return
        
        try:
            # Build application with bot token
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
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
            )
            self.application.add_error_handler(self._handle_error)
            
            # Start polling
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
        """Stop the Telegram bot and clean up resources.
        
        Gracefully shuts down the polling loop and releases resources.
        """
        if not self._is_running:
            logger.warning("Telegram channel is not running")
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
    
    async def send_message(self, to: str, content: str) -> bool:
        """Send a message to a specific Telegram chat.
        
        Args:
            to: Telegram chat ID (as string).
            content: Message text to send.
            
        Returns:
            True if message was sent successfully, False otherwise.
        """
        if not self.application or not self._is_running:
            logger.error("Cannot send message: Telegram channel is not running")
            return False
        
        try:
            chat_id = int(to)
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=content,
                parse_mode=None,  # Plain text to avoid parsing errors
            )
            logger.debug(f"Message sent to chat {chat_id}")
            return True
            
        except ValueError:
            logger.error(f"Invalid chat ID: {to}")
            return False
            
        except Exception as exc:
            logger.error(f"Failed to send message to {to}: {exc}")
            return False
    
    def on_message(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        """Register a callback for incoming messages.
        
        Args:
            callback: Async function to call when a message is received.
                     Receives (sender_id: str, message: str) parameters.
        """
        self._message_callback = callback
        logger.debug("Message callback registered")
    
    async def _handle_start_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle the /start command from users.
        
        Args:
            update: Telegram update object.
            context: Callback context.
        """
        if not update.effective_chat or not update.message:
            return
        
        welcome_text = (
            "👋 Hello! I'm a LollmsBot agent.\n\n"
            "Send me any message and I'll process it through my AI backend."
        )
        
        await update.message.reply_text(welcome_text)
        logger.info(f"Start command from chat {update.effective_chat.id}")
    
    async def _handle_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle incoming text messages from Telegram.
        
        Wraps Telegram updates into standardized format and routes
        to the registered callback or agent.
        
        Args:
            update: Telegram update object.
            context: Callback context.
        """
        if not update.effective_chat or not update.message or not update.message.text:
            return
        
        chat_id = update.effective_chat.id
        sender_id = str(chat_id)
        message_text = update.message.text
        
        # Check allowed chat IDs if configured
        if self.allowed_chat_ids is not None:
            if chat_id not in self.allowed_chat_ids:
                logger.warning(f"Message from unauthorized chat {chat_id} rejected")
                await update.message.reply_text(
                    "⛔ You are not authorized to use this bot."
                )
                return
        
        logger.info(f"Received message from chat {chat_id}: {message_text[:50]}...")
        
        try:
            if self._message_callback is not None:
                # Use registered callback
                await self._message_callback(sender_id, message_text)
            elif self.agent is not None:
                # Use agent directly
                response = await self.agent.run(message_text)
                await self.send_message(sender_id, response)
            else:
                # No handler configured
                await update.message.reply_text(
                    "⚠️ Bot is not fully configured. No message handler available."
                )
                
        except Exception as exc:
            logger.error(f"Error processing message from {chat_id}: {exc}")
            try:
                await update.message.reply_text(
                    "❌ Sorry, an error occurred while processing your message."
                )
            except Exception:
                pass  # Ignore errors in error handling
    
    async def _handle_error(
        self,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle errors in the telegram bot.
        
        Args:
            update: The update that caused the error, if available.
            context: The callback context containing error information.
        """
        logger.error(f"Telegram bot error: {context.error}")
        
        if update and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "❌ An unexpected error occurred. Please try again later."
                )
            except Exception:
                pass  # Ignore errors in error handler
    
    def __repr__(self) -> str:
        status = "running" if self._is_running else "stopped"
        restricted = (
            f", restricted={len(self.allowed_chat_ids)} chats"
            if self.allowed_chat_ids
            else ""
        )
        return f"TelegramChannel({status}{restricted})"