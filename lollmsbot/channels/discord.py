import asyncio
import logging
import platform
import subprocess
from datetime import datetime
from typing import Optional, Set, Dict, Any, List
from dataclasses import dataclass, field

import discord
from discord import Embed, Color, File

from lollmsbot.agent import Agent, PermissionLevel
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DiscordUserSession:
    """Per-user session data for maintaining state."""
    user_id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    max_history: int = 10
    
    def add_exchange(self, user_msg: str, bot_response: str):
        """Add a conversation exchange to history."""
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": bot_response})
        # Trim to max history
        while len(self.history) > self.max_history * 2:
            self.history.pop(0)
    
    def get_context_prompt(self, current_message: str) -> str:
        """Build a prompt with conversation context."""
        if not self.history:
            return current_message
        
        context_parts = []
        
        # Add recent history
        for entry in self.history[-6:]:  # Last 3 exchanges
            prefix = "User" if entry["role"] == "user" else "Assistant"
            context_parts.append(f"{prefix}: {entry['content']}")
        
        context_parts.append(f"User: {current_message}")
        context_parts.append("Assistant:")
        
        return "\n".join(context_parts)


class DiscordChannel:
    """Enhanced Discord bot channel with full LoLLMS Agent capabilities and file delivery."""
    
    def __init__(
        self,
        agent: Agent,
        bot_token: Optional[str] = None,
        allowed_guilds: Optional[Set[int]] = None,
        allowed_users: Optional[Set[int]] = None,
        blocked_users: Optional[Set[int]] = None,
        require_mention_in_guild: bool = True,
        require_mention_in_dm: bool = False,
    ):
        self.agent = agent
        self.bot_token = bot_token
        self.allowed_guilds = allowed_guilds
        self.allowed_users = allowed_users
        self.blocked_users = blocked_users or set()
        self.require_mention_in_guild = require_mention_in_guild
        self.require_mention_in_dm = require_mention_in_dm
        
        self._is_running = False
        self._ready_event = asyncio.Event()
        
        self.intents = discord.Intents.default()
        self.intents.message_content = True
        self.intents.guilds = True
        self.intents.guild_messages = True
        self.intents.dm_messages = True
        
        self.bot = discord.Client(intents=self.intents)
        self._setup_handlers()
        
        # Register file delivery callback with agent
        self.agent.set_file_delivery_callback(self._deliver_files)

    def _setup_handlers(self) -> None:
        """Set up Discord.py event handlers."""
        
        @self.bot.event
        async def on_ready():
            self._is_running = True
            self._ready_event.set()
            logger.info(f"🤖 Discord bot '{self.bot.user}' ready!")
            logger.info(f"   Servers: {len(self.bot.guilds)}")
            for guild in self.bot.guilds:
                logger.info(f"   • {guild.name}")
            print(f"🤖 Discord ready with full agent capabilities!")

        @self.bot.event
        async def on_message(message: discord.Message):
            await self._handle_message(message)

    def _can_interact(self, message: discord.Message) -> tuple[bool, str]:
        """Check if we should process this message."""
        if message.author == self.bot.user:
            return False, "own message"
        if message.author.bot:
            return False, "bot message"
        if message.author.id in self.blocked_users:
            return False, "user blocked"
        if self.allowed_users is not None:
            if message.author.id not in self.allowed_users:
                return False, "not in allowed users"
        if message.guild:
            if self.allowed_guilds is not None:
                if message.guild.id not in self.allowed_guilds:
                    return False, "guild not allowed"
            if self.require_mention_in_guild:
                is_mentioned = self.bot.user in message.mentions
                if not is_mentioned and not self._content_mentions_bot(message):
                    return False, "not mentioned in guild"
        else:
            if self.require_mention_in_dm:
                if self.bot.user not in message.mentions:
                    return False, "not mentioned in DM"
        return True, ""

    def _content_mentions_bot(self, message: discord.Message) -> bool:
        """Check if message content contains bot mention."""
        if not self.bot.user:
            return False
        patterns = [
            f"<@{self.bot.user.id}>",
            f"<@!{self.bot.user.id}>",
        ]
        return any(p in message.content for p in patterns)

    def _extract_clean_content(self, message: discord.Message) -> str:
        """Extract message content, removing bot mentions."""
        if not self.bot.user:
            return message.content.strip()
        
        content = message.content
        patterns = [
            f"<@{self.bot.user.id}>",
            f"<@!{self.bot.user.id}>",
            f"@{self.bot.user.name}",
        ]
        if self.bot.user.discriminator != "0":
            patterns.append(f"@{self.bot.user.name}#{self.bot.user.discriminator}")
        
        for pattern in patterns:
            content = content.replace(pattern, "")
        
        return content.strip()

    def _get_user_id(self, message: discord.Message) -> str:
        """Generate consistent user ID."""
        return f"discord:{message.author.id}"
    
    def _get_discord_user_id(self, agent_user_id: str) -> Optional[int]:
        """Extract Discord user ID from agent user ID."""
        if agent_user_id.startswith("discord:"):
            try:
                return int(agent_user_id.split(":", 1)[1])
            except ValueError:
                pass
        return None

    def _markdown_to_discord(self, text: str) -> str:
        """Convert standard markdown to Discord markdown.
        
        Discord uses:
        *italic* or _italic_ -> ** for bold?? No wait, let me check:
        Actually Discord: **bold**, *italic*, __underline__, ~~strikethrough~~
        
        Standard markdown: **bold**, *italic*, _italic_
        
        So we need to convert:
        - **text** -> **text** (same)
        - *text* or _text_ -> *text* (same for italic)
        - But we need to handle the table format and other Discord-specific formatting
        """
        import re
        
        # Handle code blocks first (preserve them)
        # Discord uses ``` for code blocks, same as markdown
        
        # Convert table-like structures to Discord-friendly format
        # The | column | column | format should be converted
        
        # Replace markdown headers with bold (Discord doesn't have headers)
        text = re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
        
        # Handle bullet points - Discord supports - and * for bullets
        # Just keep as is
        
        # Handle the table syntax - convert to code block or simple format
        lines = text.split('\n')
        result_lines = []
        in_table = False
        table_lines = []
        
        for line in lines:
            # Check if this is a table row
            if '|' in line and not line.strip().startswith('```'):
                if not in_table:
                    in_table = True
                    table_lines = []
                table_lines.append(line)
            else:
                if in_table:
                    # Process accumulated table
                    result_lines.extend(self._convert_table_to_discord(table_lines))
                    in_table = False
                    table_lines = []
                result_lines.append(line)
        
        # Handle remaining table
        if in_table and table_lines:
            result_lines.extend(self._convert_table_to_discord(table_lines))
        
        return '\n'.join(result_lines)
    
    def _convert_table_to_discord(self, table_lines: List[str]) -> List[str]:
        """Convert markdown table to Discord-friendly format."""
        if not table_lines:
            return []
        
        # Simple approach: convert to a code block for alignment
        # Or convert to bold headers with bullet points
        
        # Check if it's a simple 2-column comparison table
        if len(table_lines) >= 3 and all('|' in line for line in table_lines):
            # Parse the table
            rows = []
            for line in table_lines:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:
                    rows.append(cells)
            
            # Skip separator row (contains ---)
            rows = [r for r in rows if not all('-' in c for c in r)]
            
            if len(rows) >= 2:
                # Format as bold headers with values
                result = []
                headers = rows[0]
                for row in rows[1:]:
                    for i, (header, value) in enumerate(zip(headers, row)):
                        result.append(f"**{header}:** {value}")
                    result.append("")  # Empty line between rows
                return result
        
        # Fallback: return as code block
        return ['```'] + table_lines + ['```']

    async def _deliver_files(self, user_id: str, files: List[Dict[str, Any]]) -> bool:
        """Deliver files to a Discord user via DM.
        
        This is the callback registered with the Agent for file delivery.
        Files are always sent via DM to avoid cluttering guild channels.
        
        Args:
            user_id: Agent-format user ID (e.g., "discord:123456").
            files: List of file dicts with 'path', 'filename', 'description' keys.
            
        Returns:
            True if all files were delivered successfully.
        """
        discord_id = self._get_discord_user_id(user_id)
        if not discord_id:
            logger.warning(f"Cannot deliver files: unknown user format {user_id}")
            return False
        
        if not files:
            logger.debug(f"No files to deliver for {user_id}")
            return True
        
        try:
            # Get or create DM channel
            user = await self.bot.fetch_user(discord_id)
            if not user:
                logger.warning(f"Cannot deliver files: user {discord_id} not found")
                return False
            
            dm_channel = await user.create_dm()
            
            logger.info(f"📤 Delivering {len(files)} file(s) to user {discord_id}")
            
            # Send intro message
            file_list = ", ".join([f.get("filename", "unnamed") for f in files])
            if len(files) == 1:
                await dm_channel.send(f"📎 Here's your file: **{file_list}**")
            else:
                await dm_channel.send(f"📎 Here are your {len(files)} files: **{file_list}**")
            
            # Send each file
            success_count = 0
            for file_info in files:
                file_path = file_info.get("path")
                filename = file_info.get("filename") or Path(file_path).name
                description = file_info.get("description", "")
                
                if not file_path or not Path(file_path).exists():
                    logger.warning(f"File not found for delivery: {file_path}")
                    await dm_channel.send(f"⚠️ Could not find file: {filename}")
                    continue
                
                try:
                    # Send with optional description
                    if description:
                        await dm_channel.send(description[:1900])  # Discord limit
                    
                    # Send file
                    discord_file = File(file_path, filename=filename)
                    await dm_channel.send(file=discord_file)
                    success_count += 1
                    logger.info(f"✅ Delivered file {filename} to user {discord_id}")
                except Exception as e:
                    logger.error(f"Failed to send file {filename}: {e}")
                    await dm_channel.send(f"❌ Failed to send {filename}: {str(e)[:100]}")
            
            return success_count == len(files)
            
        except Exception as e:
            logger.error(f"Failed to deliver files to {user_id}: {e}")
            return False

    async def _handle_message(self, message: discord.Message) -> None:
        """Process an incoming Discord message with full agent capabilities."""

        
        location = "DM" if message.guild is None else f"#{message.channel.name}"
        logger.info(f"Message from {message.author} in {location}: '{message.content[:100]}'")
        
        can_interact, reason = self._can_interact(message)
        if not can_interact:
            logger.debug(f"Ignoring: {reason}")
            return
        
        clean_content = self._extract_clean_content(message)
        if not clean_content:
            return
        
        user_id = self._get_user_id(message)
        is_dm = message.guild is None
        
        async with message.channel.typing():
            try:
                # Use the shared Agent for processing
                # The Agent's Soul configuration will be used automatically
                result = await self.agent.chat(
                    user_id=user_id,
                    message=clean_content,
                    context={
                        "channel": "discord",
                        "discord_guild_id": message.guild.id if message.guild else None,
                        "discord_channel_id": message.channel.id,
                        "discord_is_dm": is_dm,
                    },
                )
                
                # Handle permission denied
                if result.get("permission_denied"):
                    await message.reply("⛔ You don't have permission to use this bot.")
                    return
                
                # Handle error
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    await message.reply(f"❌ Error: {error_msg[:500]}")
                    return
                
                # Get response
                response = result.get("response", "No response")
                
                # Convert markdown to Discord format
                discord_response = self._markdown_to_discord(response)
                
                # Get files info
                files_to_send = result.get("files_to_send", [])
                tools_used = result.get("tools_used", [])
                
                # Build response with file info
                final_response = discord_response
                
                # If files were generated, mention them
                if files_to_send:
                    file_count = len(files_to_send)
                    file_list = ", ".join([f.get("filename", "unnamed") for f in files_to_send[:3]])
                    if file_count > 3:
                        file_list += f" and {file_count - 3} more"
                    
                    # Add file delivery notice
                    if is_dm:
                        final_response += f"\n\n📎 I've sent {file_count} file(s): {file_list}"
                    else:
                        final_response += f"\n\n📎 Check your DMs! I've sent you {file_count} file(s): {file_list}"
                
                # Send text response (respecting Discord's 2000 char limit)
                await self._send_response(message, final_response)
                
                # Log tool usage
                if tools_used:
                    logger.info(f"🔧 Tools used for {user_id}: {', '.join(tools_used)}")
                
            except Exception as exc:
                logger.exception(f"Error processing message: {exc}")
                try:
                    await message.reply("🚨 An error occurred. Please try again.")
                except Exception:
                    pass

    async def _send_response(self, message: discord.Message, response: str) -> None:
        """Send response with appropriate formatting."""
        # Discord has 2000 char limit for regular messages
        MAX_LEN = 1950
        
        if len(response) <= MAX_LEN:
            await message.reply(response)
            return
        
        # For long responses, split intelligently
        chunks = self._split_message(response, MAX_LEN)
        
        # Send first chunk as reply, rest as follow-ups
        for i, chunk in enumerate(chunks):
            if i == 0:
                await message.reply(chunk)
            else:
                await message.channel.send(chunk)
            
            # Small delay to avoid rate limiting
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)

    def _split_message(self, text: str, max_len: int = 1950) -> List[str]:
        """Split long message into Discord-compatible chunks."""
        if len(text) <= max_len:
            return [text]
        
        chunks = []
        remaining = text
        
        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break
            
            # Find good break point
            split_at = remaining.rfind('\n\n', 0, max_len)
            if split_at == -1:
                split_at = remaining.rfind('\n', 0, max_len)
            if split_at == -1:
                split_at = remaining.rfind('. ', 0, max_len)
            if split_at == -1:
                split_at = remaining.rfind(' ', 0, max_len)
            if split_at == -1:
                split_at = max_len
            
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip()
        
        return chunks

    async def start(self):
        """Start Discord bot."""
        if not self.bot_token:
            raise ValueError("Discord bot token required")
        
        logger.info("Starting Discord bot with full agent capabilities...")
        logging.getLogger('discord').setLevel(logging.INFO)
        await self.bot.start(self.bot_token)

    async def stop(self):
        """Graceful shutdown."""
        self._is_running = False
        await self.bot.close()
        logger.info("Discord bot stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running and self._ready_event.is_set()

    async def wait_for_ready(self, timeout: float = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def __repr__(self) -> str:
        status = "ready" if self.is_running else "connecting" if self._is_running else "stopped"
        return f"DiscordChannel({status}, agent={self.agent.name})"
