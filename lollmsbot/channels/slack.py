"""
Slack channel implementation for LollmsBot.

Uses shared Agent for all business logic. Supports:
- Slack Bolt SDK for Python (modern, official SDK)
- Socket Mode (WebSocket-based, no public URL needed)
- HTTP Mode (webhook-based, requires public URL)
- Direct messages and channel mentions
- File uploads and rich block formatting
- Thread-aware conversations

This implementation provides a flexible interface that works with
various Slack deployment scenarios while maintaining the same
Agent-based architecture as other channels.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

from lollmsbot.agent import Agent, PermissionLevel


logger = logging.getLogger(__name__)


@dataclass
class SlackMessage:
    """Represents an incoming Slack message."""
    message_id: str
    channel_id: str
    user_id: str
    user_name: Optional[str]
    text: str
    timestamp: str
    thread_ts: Optional[str] = None  # For threaded conversations
    is_dm: bool = False
    files: List[Dict[str, Any]] = None
    mentions_bot: bool = False


class SlackChannel:
    """Slack messaging channel using shared Agent.
    
    Supports two connection modes:
    - 'socket': WebSocket-based (Socket Mode), no public URL needed
    - 'http': HTTP webhook-based, requires public URL
    
    All business logic is delegated to the Agent. This class handles
    only Slack-specific protocol concerns.
    """

    BOLT_AVAILABLE = False
    
    def __init__(
        self,
        agent: Agent,
        bot_token: str,
        signing_secret: Optional[str] = None,
        app_token: Optional[str] = None,  # Required for Socket Mode
        mode: str = "socket",  # 'socket' or 'http'
        allowed_users: Optional[Set[str]] = None,
        allowed_channels: Optional[Set[str]] = None,
        blocked_users: Optional[Set[str]] = None,
        require_mention_in_channel: bool = True,
        bot_name: Optional[str] = None,  # For mention detection
    ):
        self._check_bolt_available()
        
        self.agent = agent
        self.bot_token = bot_token
        self.signing_secret = signing_secret
        self.app_token = app_token
        self.mode = mode
        
        self.allowed_users: Set[str] = set(allowed_users) if allowed_users else set()
        self.allowed_channels: Set[str] = set(allowed_channels) if allowed_channels else set()
        self.blocked_users: Set[str] = set(blocked_users) if blocked_users else set()
        self.require_mention_in_channel = require_mention_in_channel
        self.bot_name = bot_name or "lollmsbot"
        self.bot_user_id: Optional[str] = None  # Will be fetched on start
        
        self._is_running = False
        self._app: Optional[Any] = None  # Slack Bolt App instance
        self._handler: Optional[Any] = None  # SocketModeHandler or None for HTTP
        
        # Track DM channels (no mention required)
        self._dm_channels: Set[str] = set()
        
        # Message queue for async processing
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._response_queue: asyncio.Queue = asyncio.Queue()
    
    def _check_bolt_available(self) -> None:
        """Check if Slack Bolt SDK is available."""
        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler
            self.BOLT_AVAILABLE = True
        except ImportError:
            logger.warning(
                "Slack Bolt SDK not available. "
                "Install with: pip install slack-bolt"
            )
            self.BOLT_AVAILABLE = False
    
    def _can_interact(self, user_id: str, channel_id: str, is_dm: bool = False) -> tuple[bool, str]:
        """Check if user can interact with bot."""
        # Check blocked users
        if user_id in self.blocked_users:
            return False, "user blocked"
        
        # Check allowed users
        if self.allowed_users and user_id not in self.allowed_users:
            logger.info(f"User {user_id} not in allowed list")
            return False, "not in allowed users"
        
        # Check allowed channels (only for non-DM)
        if not is_dm and self.allowed_channels and channel_id not in self.allowed_channels:
            logger.info(f"Channel {channel_id} not in allowed list")
            return False, "channel not allowed"
        
        return True, ""
    
    def _get_user_id(self, slack_user_id: str) -> str:
        """Generate consistent user ID for Agent."""
        return f"slack:{slack_user_id}"
    
    def _normalize_text(self, text: str) -> str:
        """Clean up Slack message text (remove bot mentions, etc.)."""
        # Remove bot mention patterns like <@U12345678>
        import re
        text = re.sub(r'<@[A-Z0-9]+>', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    async def start(self) -> None:
        """Start the Slack channel."""
        if not self.BOLT_AVAILABLE:
            raise ImportError(
                "Slack channel requires 'slack-bolt' package. "
                "Install with: pip install slack-bolt"
            )
        
        if self._is_running:
            logger.warning("Slack channel is already running")
            return
        
        logger.info("=" * 60)
        logger.info("💬 Slack Channel Starting...")
        logger.info(f"   Mode: {self.mode}")
        logger.info(f"   Allowed users: {len(self.allowed_users)}")
        logger.info(f"   Allowed channels: {len(self.allowed_channels)}")
        logger.info(f"   Blocked users: {len(self.blocked_users)}")
        logger.info("=" * 60)
        
        try:
            from slack_bolt import App
            from slack_bolt.adapter.socket_mode import SocketModeHandler
            
            # Initialize Bolt app
            app_kwargs = {"token": self.bot_token}
            if self.mode == "http":
                if not self.signing_secret:
                    raise ValueError("HTTP mode requires signing_secret")
                app_kwargs["signing_secret"] = self.signing_secret
            
            self._app = App(**app_kwargs)
            
            # Register event handlers
            self._register_handlers()
            
            # Get bot info
            await self._fetch_bot_info()
            
            # Start in appropriate mode
            if self.mode == "socket":
                if not self.app_token:
                    raise ValueError("Socket mode requires app_token")
                
                logger.info("🔌 Starting Socket Mode (WebSocket)...")
                self._handler = SocketModeHandler(self._app, self.app_token)
                
                # Run in background
                asyncio.create_task(self._run_socket_mode())
                
            else:  # http mode
                logger.info("🌐 Starting HTTP Mode (webhook)...")
                # HTTP mode requires the app to be mounted in an ASGI server
                # This will be handled by the gateway's lifespan
                pass
            
            self._is_running = True
            
            # Start message processor
            asyncio.create_task(self._process_message_queue())
            asyncio.create_task(self._process_response_queue())
            
            logger.info("✅ Slack channel started successfully")
            
        except Exception as e:
            logger.error(f"❌ Slack channel failed to start: {e}")
            raise
    
    async def _fetch_bot_info(self) -> None:
        """Fetch bot user ID for mention detection."""
        try:
            from slack_sdk.web.async_client import AsyncWebClient
            
            client = AsyncWebClient(token=self.bot_token)
            auth_info = await client.auth_test()
            
            self.bot_user_id = auth_info.get("user_id")
            self.bot_name = auth_info.get("user", self.bot_name)
            
            logger.info(f"🤖 Bot info: @{self.bot_name} (ID: {self.bot_user_id})")
            
        except Exception as e:
            logger.warning(f"Could not fetch bot info: {e}")
    
    def _register_handlers(self) -> None:
        """Register Slack event handlers."""
        
        @self._app.event("message")
        async def handle_message(event, say, client):
            """Handle incoming messages."""
            await self._message_queue.put({
                "event": event,
                "say": say,
                "client": client,
            })
        
        @self._app.event("app_mention")
        async def handle_mention(event, say, client):
            """Handle explicit @mentions."""
            event["mentions_bot"] = True
            await self._message_queue.put({
                "event": event,
                "say": say,
                "client": client,
            })
        
        @self._app.event("member_joined_channel")
        async def handle_join(event, client):
            """Track when bot joins channels."""
            if event.get("user") == self.bot_user_id:
                logger.info(f"✅ Joined channel: {event.get('channel')}")
        
        @self._app.error
        async def handle_error(error, body, logger):
            """Handle errors."""
            logger.error(f"Slack Bolt error: {error}")
            logger.error(f"Event body: {body}")
    
    async def _run_socket_mode(self) -> None:
        """Run Socket Mode handler."""
        try:
            # SocketModeHandler.start() blocks, so we need to run it in a thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._handler.start)
        except Exception as e:
            logger.error(f"Socket mode error: {e}")
    
    async def _process_message_queue(self) -> None:
        """Process incoming messages from queue."""
        while self._is_running:
            try:
                msg_data = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._handle_incoming_message(msg_data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
    
    async def _process_response_queue(self) -> None:
        """Process outgoing responses from queue."""
        while self._is_running:
            try:
                response = await asyncio.wait_for(self._response_queue.get(), timeout=1.0)
                await self._send_response(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing response queue: {e}")
    
    async def _handle_incoming_message(self, msg_data: Dict[str, Any]) -> None:
        """Process a single incoming message."""
        event = msg_data["event"]
        say = msg_data["say"]
        client = msg_data["client"]
        
        # Extract message data
        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        text = event.get("text", "")
        ts = event.get("ts", "")
        thread_ts = event.get("thread_ts")
        
        # Skip messages from the bot itself
        if user_id == self.bot_user_id:
            return
        
        # Skip bot messages
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return
        
        # Determine if DM
        is_dm = channel_id.startswith("D")  # DM channels start with 'D'
        if is_dm:
            self._dm_channels.add(channel_id)
        
        # Check for mention
        mentions_bot = event.get("mentions_bot", False)
        if self.bot_user_id and f"<@{self.bot_user_id}>" in text:
            mentions_bot = True
        
        # In channels, require mention unless disabled
        if not is_dm and self.require_mention_in_channel and not mentions_bot:
            logger.debug(f"Ignoring message in channel {channel_id} without mention")
            return
        
        # Clean text
        clean_text = self._normalize_text(text)
        if not clean_text:
            return
        
        # Check permissions
        can_interact, reason = self._can_interact(user_id, channel_id, is_dm)
        if not can_interact:
            logger.warning(f"Blocked message from {user_id}: {reason}")
            if is_dm or mentions_bot:
                await say("⛔ You don't have permission to use this bot.")
            return
        
        # Get user info
        user_name = None
        try:
            user_info = await client.users_info(user=user_id)
            if user_info.get("ok"):
                user_name = user_info["user"].get("real_name") or user_info["user"].get("name")
        except Exception:
            pass
        
        # Build message object
        message = SlackMessage(
            message_id=ts,
            channel_id=channel_id,
            user_id=user_id,
            user_name=user_name,
            text=clean_text,
            timestamp=ts,
            thread_ts=thread_ts,
            is_dm=is_dm,
            files=event.get("files", []),
            mentions_bot=mentions_bot,
        )
        
        # Queue for processing
        asyncio.create_task(self._process_message(message, say, client))
    
    async def _process_message(self, message: SlackMessage, say: Callable, client: Any) -> None:
        """Process message through the Agent."""
        user_id = self._get_user_id(message.user_id)
        
        logger.info(f"Processing Slack message from {message.user_name or message.user_id}: {message.text[:50]}...")
        
        try:
            result = await self.agent.chat(
                user_id=user_id,
                message=message.text,
                context={
                    "channel": "slack",
                    "slack_user_id": message.user_id,
                    "slack_user_name": message.user_name,
                    "slack_channel_id": message.channel_id,
                    "slack_is_dm": message.is_dm,
                    "slack_thread_ts": message.thread_ts,
                },
            )
            
            if result.get("permission_denied"):
                await say("⛔ You don't have permission to use this bot.")
                return
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                await say(f"❌ Error: {error_msg[:500]}")
                return
            
            response = result.get("response", "No response")
            
            # Slack has a 3000 character limit for blocks, 4000 for text
            # Split long messages if needed
            max_length = 2900  # Conservative limit for blocks
            
            if len(response) > max_length:
                chunks = self._split_message(response, max_length)
                for i, chunk in enumerate(chunks):
                    prefix = f"(Part {i+1}/{len(chunks)}) " if len(chunks) > 1 else ""
                    await self._queue_response(say, chunk, message.thread_ts, prefix)
            else:
                await self._queue_response(say, response, message.thread_ts)
            
            # Handle files if any were generated
            files = result.get("files_to_send", [])
            if files:
                await self._handle_files(say, client, message.channel_id, files, message.thread_ts)
                
        except Exception as exc:
            logger.error(f"Error processing Slack message: {exc}")
            try:
                await say("❌ An error occurred. Please try again.")
            except Exception:
                pass
    
    def _split_message(self, text: str, max_length: int) -> List[str]:
        """Split long message into chunks at sentence boundaries."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            
            # Find a good break point
            break_point = max_length
            
            # Try to break at paragraph
            para_break = text.rfind('\n\n', 0, max_length)
            if para_break > max_length * 0.5:
                break_point = para_break + 2
            
            # Or at sentence end
            elif '. ' in text[:max_length]:
                last_sentence = text.rfind('. ', 0, max_length)
                if last_sentence > max_length * 0.5:
                    break_point = last_sentence + 2
            
            chunks.append(text[:break_point].strip())
            text = text[break_point:].strip()
        
        return chunks
    
    async def _queue_response(self, say: Callable, text: str, thread_ts: Optional[str], prefix: str = "") -> None:
        """Queue a response to be sent."""
        await self._response_queue.put({
            "say": say,
            "text": prefix + text,
            "thread_ts": thread_ts,
        })
    
    async def _send_response(self, response_data: Dict[str, Any]) -> None:
        """Send a response to Slack."""
        try:
            say = response_data["say"]
            text = response_data["text"]
            thread_ts = response_data.get("thread_ts")
            
            # Use blocks for better formatting if it's a complex message
            if self._should_use_blocks(text):
                blocks = self._format_as_blocks(text)
                await say(blocks=blocks, thread_ts=thread_ts)
            else:
                await say(text=text, thread_ts=thread_ts)
                
        except Exception as e:
            logger.error(f"Error sending Slack response: {e}")
    
    def _should_use_blocks(self, text: str) -> bool:
        """Determine if message should use Block Kit formatting."""
        # Use blocks for code blocks, lists, or complex formatting
        return (
            "```" in text or
            text.count('\n') > 5 or
            text.startswith('#') or
            '- ' in text or
            '* ' in text
        )
    
    def _format_as_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Format text as Slack Block Kit blocks."""
        blocks = []
        
        # Split by code blocks
        parts = text.split('```')
        
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code block
                # Truncate if too long
                code = part[:2900] if len(part) > 2900 else part
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"```{code}```"
                    }
                })
            else:  # Regular text
                if part.strip():
                    # Split into chunks if needed
                    chunks = self._chunk_text(part.strip(), 2900)
                    for chunk in chunks:
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": self._slack_escape(chunk)
                            }
                        })
        
        return blocks
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Chunk text for Slack blocks."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            chunks.append(text[:max_length])
            text = text[max_length:]
        
        return chunks
    
    def _slack_escape(self, text: str) -> str:
        """Escape special characters for Slack mrkdwn."""
        # Basic escaping for Slack mrkdwn
        # Note: This is simplified; full implementation would be more robust
        return text
    
    async def _handle_files(
        self,
        say: Callable,
        client: Any,
        channel_id: str,
        files: List[Dict[str, Any]],
        thread_ts: Optional[str] = None
    ) -> None:
        """Handle file uploads to Slack."""
        file_list = []
        
        for file_info in files[:5]:  # Limit to 5 files
            file_path = file_info.get("path")
            filename = file_info.get("filename", "unnamed")
            
            if not file_path or not Path(file_path).exists():
                continue
            
            try:
                # Upload file to Slack
                from slack_sdk.web.async_client import AsyncWebClient
                
                slack_client = AsyncWebClient(token=self.bot_token)
                
                with open(file_path, 'rb') as f:
                    response = await slack_client.files_upload_v2(
                        channel=channel_id,
                        file=f,
                        filename=filename,
                        thread_ts=thread_ts,
                    )
                
                if response.get("ok"):
                    file_list.append(filename)
                    logger.info(f"Uploaded file to Slack: {filename}")
                else:
                    logger.warning(f"Failed to upload file: {response.get('error')}")
                    
            except Exception as e:
                logger.error(f"Error uploading file to Slack: {e}")
        
        # Notify about files
        if file_list:
            files_text = ", ".join(file_list)
            if len(files) > len(file_list):
                files_text += f" (and {len(files) - len(file_list)} more)"
            
            await say(
                text=f"📎 Generated {len(file_list)} file(s): {files_text}",
                thread_ts=thread_ts
            )
    
    def get_app(self) -> Optional[Any]:
        """Get the Bolt app instance for HTTP mode mounting."""
        return self._app
    
    async def stop(self) -> None:
        """Stop the Slack channel."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._handler:
            logger.info("Stopping Socket Mode handler...")
            try:
                self._handler.close()
            except Exception as e:
                logger.warning(f"Error stopping handler: {e}")
            self._handler = None
        
        self._app = None
        
        logger.info("Slack channel stopped")
    
    def __repr__(self) -> str:
        status = "running" if self._is_running else "stopped"
        mode_info = f", mode={self.mode}"
        restricted = ""
        if self.allowed_users:
            restricted += f", users={len(self.allowed_users)}"
        if self.allowed_channels:
            restricted += f", channels={len(self.allowed_channels)}"
        return f"SlackChannel({status}{mode_info}{restricted})"
