"""
Discord channel implementation for LollmsBot.

This module provides the DiscordChannel class for integrating with the
Discord messaging platform using the discord.py library.
"""

import logging
from typing import Any, Awaitable, Callable, List, Optional, Set

import discord
from discord.ext import commands

from lollmsbot.channels import Channel
from lollmsbot.agent import Agent


logger = logging.getLogger(__name__)


class DiscordBot(commands.Bot):
    """Discord bot subclass for handling events and messages.
    
    This internal bot class handles Discord-specific events and routes
    them to the parent DiscordChannel for standardized processing.
    
    Attributes:
        channel: Parent DiscordChannel instance for message routing.
        allowed_guild_ids: Optional set of allowed guild IDs for security.
    """
    
    def __init__(
        self,
        channel: "DiscordChannel",
        allowed_guild_ids: Optional[Set[int]] = None,
    ) -> None:
        """Initialize the Discord bot.
        
        Args:
            channel: Parent DiscordChannel instance.
            allowed_guild_ids: Optional set of allowed guild IDs.
        """
        self._parent_channel = channel
        self._allowed_guild_ids = allowed_guild_ids
        
        # Setup intents for message content and guild members
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        
        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None,
        )
    
    async def setup_hook(self) -> None:
        """Called when the bot is setting up."""
        logger.info("Discord bot setup complete")
    
    async def on_ready(self) -> None:
        """Called when the bot is ready and connected."""
        logger.info(f"Discord bot logged in as {self.user} (ID: {self.user.id})")
    
    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming Discord messages.
        
        Routes messages to the parent channel for standardized processing.
        
        Args:
            message: Discord message object.
        """
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Check allowed guilds if configured
        if self._allowed_guild_ids is not None:
            if message.guild is None or message.guild.id not in self._allowed_guild_ids:
                logger.warning(f"Message from unauthorized guild {message.guild.id if message.guild else 'DM'} rejected")
                return
        
        # Ignore DMs if not explicitly allowed (when allowed_guild_ids is set)
        if self._allowed_guild_ids is not None and message.guild is None:
            logger.warning("Direct message rejected (guild-only mode)")
            return
        
        # Convert to standardized format
        sender_id = str(message.author.id)
        message_text = message.content
        
        # Get channel ID for responses
        channel_id = str(message.channel.id)
        
        logger.info(f"Received message from {message.author} in channel {channel_id}: {message_text[:50]}...")
        
        # Store reference for sending responses
        self._parent_channel._last_message_channel = message.channel
        
        # Route to parent channel handler
        await self._parent_channel._handle_incoming_message(sender_id, message_text, message)
        
        # Process commands if any
        await self.process_commands(message)
    
    async def on_error(self, event_method: str, *args: Any, **kwargs: Any) -> None:
        """Handle errors in event processing.
        
        Args:
            event_method: Name of the event that caused the error.
            *args: Positional arguments passed to the event.
            **kwargs: Keyword arguments passed to the event.
        """
        logger.exception(f"Error in Discord event {event_method}")


class DiscordChannel(Channel):
    """Discord messaging channel implementation.
    
    Provides integration with Discord API using discord.py library.
    Handles message events and sending via Discord bot.
    
    Attributes:
        bot_token: Discord bot authentication token.
        allowed_guild_ids: Optional set of allowed guild IDs for security.
        bot: DiscordBot instance for Discord operations.
        _last_message_channel: Reference to last message channel for responses.
        _message_callback: Registered callback for incoming messages.
    """
    
    def __init__(
        self,
        bot_token: str,
        allowed_guild_ids: Optional[List[int]] = None,
        agent: Optional[Agent] = None,
    ) -> None:
        """Initialize the Discord channel.
        
        Args:
            bot_token: Discord bot token from Discord Developer Portal.
            allowed_guild_ids: Optional list of allowed guild (server) IDs.
                              If provided, only these guilds can interact.
            agent: Optional Agent instance for message processing.
        """
        super().__init__(name="discord", agent=agent)
        
        self.bot_token: str = bot_token
        self.allowed_guild_ids: Optional[Set[int]] = (
            set(allowed_guild_ids) if allowed_guild_ids else None
        )
        self.bot: Optional[DiscordBot] = None
        self._last_message_channel: Optional[discord.abc.Messageable] = None
        self._message_callback: Optional[
            Callable[[str, str], Awaitable[None]]
        ] = None
    
    async def start(self) -> None:
        """Start the Discord bot and begin listening for messages.
        
        Initializes the DiscordBot instance and starts the event loop
        to connect to Discord's gateway.
        """
        if self._is_running:
            logger.warning("Discord channel is already running")
            return
        
        try:
            self.bot = DiscordBot(
                channel=self,
                allowed_guild_ids=self.allowed_guild_ids,
            )
            
            # Start the bot in a task (non-blocking for the channel)
            import asyncio
            self._bot_task = asyncio.create_task(self.bot.start(self.bot_token))
            self._is_running = True
            
            logger.info("Discord channel started successfully")
            
        except Exception as exc:
            logger.error(f"Failed to start Discord channel: {exc}")
            self._is_running = False
            raise
    
    async def stop(self) -> None:
        """Stop the Discord bot and clean up resources.
        
        Gracefully closes the Discord connection and stops event processing.
        """
        if not self._is_running:
            logger.warning("Discord channel is not running")
            return
        
        try:
            if self.bot:
                await self.bot.close()
                self.bot = None
            
            self._is_running = False
            logger.info("Discord channel stopped successfully")
            
        except Exception as exc:
            logger.error(f"Error stopping Discord channel: {exc}")
            raise
    
    async def send_message(self, to: str, content: str) -> bool:
        """Send a message to a specific Discord channel or user.
        
        Args:
            to: Discord channel ID or user ID (as string).
            content: Message text to send.
            
        Returns:
            True if message was sent successfully, False otherwise.
        """
        if not self.bot or not self._is_running:
            logger.error("Cannot send message: Discord channel is not running")
            return False
        
        try:
            # Try to get channel from cache
            channel_id = int(to)
            channel = self.bot.get_channel(channel_id)
            
            # If not found, try to fetch it
            if channel is None:
                try:
                    channel = await self.bot.fetch_channel(channel_id)
                except discord.NotFound:
                    # Try as user ID
                    try:
                        user = await self.bot.fetch_user(channel_id)
                        channel = await user.create_dm()
                    except discord.NotFound:
                        logger.error(f"Channel or user not found: {to}")
                        return False
            
            if channel is None:
                logger.error(f"Could not resolve destination: {to}")
                return False
            
            # Send the message
            await channel.send(content)
            logger.debug(f"Message sent to {to}")
            return True
            
        except ValueError:
            logger.error(f"Invalid Discord ID: {to}")
            return False
            
        except discord.Forbidden:
            logger.error(f"Permission denied sending message to {to}")
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
        logger.debug("Message callback registered for Discord channel")
    
    async def _handle_incoming_message(
        self,
        sender_id: str,
        message: str,
        discord_message: discord.Message,
    ) -> None:
        """Internal handler for incoming Discord messages.
        
        Routes messages to the registered callback if available,
        or to the agent if configured. Also handles direct replies.
        
        Args:
            sender_id: Discord user ID of the sender.
            message: Message content.
            discord_message: Original Discord message object.
        """
        if self._message_callback is not None:
            await self._message_callback(sender_id, message)
        elif self.agent is not None:
            try:
                response = await self.agent.run(message)
                
                # Send response as reply to the original message
                await discord_message.reply(response)
                
            except Exception as exc:
                logger.error(f"Error processing message from {sender_id}: {exc}")
                try:
                    await discord_message.reply(
                        "❌ Sorry, an error occurred while processing your message."
                    )
                except Exception:
                    pass  # Ignore errors in error handling
        else:
            # No handler configured
            try:
                await discord_message.reply(
                    "⚠️ Bot is not fully configured. No message handler available."
                )
            except Exception:
                pass
    
    def __repr__(self) -> str:
        status = "running" if self._is_running else "stopped"
        restricted = (
            f", restricted={len(self.allowed_guild_ids)} guilds"
            if self.allowed_guild_ids
            else ""
        )
        user_info = f", user={self.bot.user}" if self.bot and self.bot.user else ""
        return f"DiscordChannel({status}{restricted}{user_info})"