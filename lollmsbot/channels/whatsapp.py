"""
WhatsApp channel implementation for LollmsBot.

Uses shared Agent for all business logic. Supports multiple backend options:
- whatsapp-web.js bridge (local, via subprocess/HTTP)
- Twilio WhatsApp API (cloud)
- WhatsApp Business API (official, webhook-based)

This implementation provides a flexible interface that can work with
various WhatsApp integration methods while maintaining the same
Agent-based architecture as other channels.
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from urllib.parse import urlencode, parse_qs

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from lollmsbot.agent import Agent, PermissionLevel


logger = logging.getLogger(__name__)


@dataclass
class WhatsAppMessage:
    """Represents an incoming WhatsApp message."""
    message_id: str
    from_number: str  # Phone number with country code
    from_name: Optional[str]
    body: str
    timestamp: float
    is_group: bool = False
    group_id: Optional[str] = None
    media_url: Optional[str] = None
    media_type: Optional[str] = None


class WhatsAppChannel:
    """WhatsApp messaging channel using shared Agent.
    
    Supports multiple backend implementations:
    - 'web_js': whatsapp-web.js via subprocess bridge (local, free)
    - 'twilio': Twilio WhatsApp API (cloud, requires account)
    - 'business_api': Official WhatsApp Business API (webhook-based)
    
    All business logic is delegated to the Agent. This class handles
    only WhatsApp-specific protocol concerns.
    """

    # Backend availability flags
    WEB_JS_AVAILABLE = False  # Requires Node.js and whatsapp-web.js
    TWILIO_AVAILABLE = False
    
    def __init__(
        self,
        agent: Agent,
        backend: str = "web_js",  # 'web_js', 'twilio', 'business_api'
        # Web.js backend options
        web_js_path: Optional[str] = None,
        # Twilio options
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,  # Twilio WhatsApp number
        # Business API options
        api_token: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        webhook_port: int = 8081,
        # Common options
        allowed_numbers: Optional[List[str]] = None,
        blocked_numbers: Optional[List[str]] = None,
        require_confirmation: bool = True,  # Require user to confirm first message
    ):
        # Check backend availability
        if backend == "web_js":
            self._check_web_js_available()
        elif backend == "twilio":
            self._check_twilio_available()
        elif backend == "business_api":
            if not AIOHTTP_AVAILABLE:
                raise ImportError(
                    "WhatsApp Business API backend requires 'aiohttp'. "
                    "Install with: pip install aiohttp"
                )
        
        self.agent = agent
        self.backend = backend
        self.web_js_path = web_js_path or str(Path.home() / ".lollmsbot" / "whatsapp-bridge")
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.api_token = api_token
        self.webhook_secret = webhook_secret
        self.webhook_port = webhook_port
        
        self.allowed_numbers: Set[str] = set(allowed_numbers) if allowed_numbers else set()
        self.blocked_numbers: Set[str] = set(blocked_numbers) if blocked_numbers else set()
        self.require_confirmation = require_confirmation
        
        self._is_running = False
        self._web_js_process: Optional[subprocess.Popen] = None
        self._webhook_app: Optional[Any] = None
        self._webhook_runner: Optional[Any] = None
        self._session: Optional[Any] = None  # aiohttp ClientSession
        
        # Track confirmed users (for confirmation requirement)
        self._confirmed_users: Set[str] = set()
        
        # Pending message queue for web_js backend
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._send_queue: asyncio.Queue = asyncio.Queue()
    
    def _check_web_js_available(self) -> None:
        """Check if whatsapp-web.js bridge can be used."""
        # Check for Node.js
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.WEB_JS_AVAILABLE = True
                logger.info(f"Node.js available: {result.stdout.strip()}")
            else:
                logger.warning("Node.js not available for whatsapp-web.js backend")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Node.js not found. Install from https://nodejs.org/")
    
    def _check_twilio_available(self) -> None:
        """Check if Twilio SDK is available."""
        try:
            from twilio.rest import Client
            from twilio.twiml.messaging_response import MessagingResponse
            self.TWILIO_AVAILABLE = True
        except ImportError:
            logger.warning(
                "Twilio SDK not available. "
                "Install with: pip install twilio"
            )
    
    def _can_interact(self, phone_number: str, is_first_message: bool = False) -> tuple[bool, str]:
        """Check if user can interact with bot."""
        # Normalize phone number
        normalized = self._normalize_number(phone_number)
        
        if normalized in self.blocked_numbers:
            return False, "number blocked"
        
        if self.allowed_numbers and normalized not in self.allowed_numbers:
            return False, "not in allowed numbers"
        
        # Check confirmation requirement
        if self.require_confirmation and is_first_message and normalized not in self._confirmed_users:
            return False, "confirmation_required"
        
        return True, ""
    
    def _normalize_number(self, number: str) -> str:
        """Normalize phone number to E.164 format."""
        # Remove spaces, dashes, and leading +
        cleaned = number.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        if not cleaned.startswith("+"):
            # Assume US if no country code (add your logic)
            if len(cleaned) == 10:
                cleaned = "+1" + cleaned
            else:
                cleaned = "+" + cleaned
        return cleaned
    
    def _get_user_id(self, phone_number: str) -> str:
        """Generate consistent user ID for Agent."""
        normalized = self._normalize_number(phone_number)
        return f"whatsapp:{normalized}"
    
    async def start(self) -> None:
        """Start the WhatsApp channel."""
        if self._is_running:
            logger.warning("WhatsApp channel is already running")
            return
        
        if self.backend == "web_js":
            await self._start_web_js_backend()
        elif self.backend == "twilio":
            await self._start_twilio_backend()
        elif self.backend == "business_api":
            await self._start_business_api_backend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        self._is_running = True
        logger.info(f"WhatsApp channel started with {self.backend} backend")
    
    async def _start_web_js_backend(self) -> None:
        """Start the whatsapp-web.js bridge."""
        if not self.WEB_JS_AVAILABLE:
            raise RuntimeError(
                "whatsapp-web.js backend requires Node.js. "
                "Install Node.js from https://nodejs.org/ "
                "Then run: npm install whatsapp-web.js qrcode-terminal"
            )
        
        # Ensure bridge script exists
        bridge_js = Path(self.web_js_path) / "bridge.js"
        bridge_js.parent.mkdir(parents=True, exist_ok=True)
        
        if not bridge_js.exists():
            # Create the bridge script
            bridge_js.write_text(self._get_bridge_script())
            logger.info(f"Created whatsapp-web.js bridge at {bridge_js}")
        
        # Install dependencies if needed
        package_json = bridge_js.parent / "package.json"
        if not package_json.exists():
            package_json.write_text(json.dumps({
                "name": "lollmsbot-whatsapp-bridge",
                "version": "1.0.0",
                "dependencies": {
                    "whatsapp-web.js": "^1.23.0",
                    "qrcode-terminal": "^0.12.0"
                }
            }))
            # Run npm install
            logger.info("Installing whatsapp-web.js dependencies...")
            proc = await asyncio.create_subprocess_exec(
                "npm", "install",
                cwd=str(bridge_js.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"npm install failed: {stderr.decode()}")
        
        # Start the bridge process
        logger.info("Starting whatsapp-web.js bridge...")
        logger.info("Scan the QR code with your phone to authenticate")
        
        self._web_js_process = await asyncio.create_subprocess_exec(
            "node", str(bridge_js),
            cwd=str(bridge_js.parent),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Start reader tasks
        asyncio.create_task(self._read_web_js_output())
        asyncio.create_task(self._process_send_queue())
    
    def _get_bridge_script(self) -> str:
        """Get the Node.js bridge script for whatsapp-web.js."""
        return '''
const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

const client = new Client({
    authStrategy: new LocalAuth(),
    puppeteer: {
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    }
});

client.on('qr', (qr) => {
    qrcode.generate(qr, { small: true });
    console.log('QR_RECEIVED');
});

client.on('ready', () => {
    console.log('READY');
});

client.on('message', async (msg) => {
    const data = {
        id: msg.id._serialized,
        from: msg.from,
        body: msg.body,
        timestamp: msg.timestamp,
        isGroup: msg.from.includes('@g.us'),
        hasMedia: msg.hasMedia
    };
    console.log('MSG:' + JSON.stringify(data));
});

// Read commands from stdin
const readline = require('readline');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

rl.on('line', (line) => {
    try {
        const cmd = JSON.parse(line);
        if (cmd.type === 'send') {
            client.sendMessage(cmd.to, cmd.body).then(() => {
                console.log('SENT:' + cmd.id);
            }).catch((err) => {
                console.log('ERROR:' + err.message);
            });
        }
    } catch (e) {
        console.log('PARSE_ERROR:' + e.message);
    }
});

client.initialize();
'''
    
    async def _read_web_js_output(self) -> None:
        """Read output from whatsapp-web.js process."""
        if not self._web_js_process:
            return
        
        while self._is_running and self._web_js_process:
            try:
                line = await self._web_js_process.stdout.readline()
                if not line:
                    break
                
                text = line.decode().strip()
                
                if text == "READY":
                    logger.info("WhatsApp Web ready!")
                elif text.startswith("MSG:"):
                    data = json.loads(text[4:])
                    await self._handle_incoming_message(data)
                elif text.startswith("SENT:"):
                    logger.debug(f"Message sent: {text[5:]}")
                elif text.startswith("ERROR:"):
                    logger.error(f"Send error: {text[6:]}")
                    
            except Exception as e:
                logger.error(f"Error reading web.js output: {e}")
                await asyncio.sleep(1)
    
    async def _process_send_queue(self) -> None:
        """Process outgoing messages from queue."""
        while self._is_running:
            try:
                msg = await asyncio.wait_for(self._send_queue.get(), timeout=1.0)
                if self._web_js_process and self._web_js_process.stdin:
                    cmd = json.dumps({
                        "type": "send",
                        "to": msg["to"],
                        "body": msg["body"],
                        "id": msg.get("id", "")
                    })
                    self._web_js_process.stdin.write((cmd + "\n").encode())
                    await self._web_js_process.stdin.drain()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in send queue: {e}")
    
    async def _handle_incoming_message(self, data: Dict[str, Any]) -> None:
        """Handle message from whatsapp-web.js."""
        # Don't respond to our own messages
        if data.get("fromMe"):
            return
        
        # Check if this is first message from this number
        from_number = data.get("from", "").replace("@c.us", "").replace("@g.us", "")
        is_first = from_number not in self._confirmed_users
        
        can_interact, reason = self._can_interact(from_number, is_first)
        
        if not can_interact:
            if reason == "confirmation_required":
                # Send confirmation request
                await self._send_web_js(
                    data["from"],
                    "👋 Welcome! Reply 'YES' to start chatting with this AI assistant."
                )
                return
            else:
                logger.warning(f"Blocked message from {from_number}: {reason}")
                return
        
        # Check for confirmation response
        if is_first and data.get("body", "").strip().upper() == "YES":
            self._confirmed_users.add(from_number)
            await self._send_web_js(
                data["from"],
                "✅ Confirmed! You can now chat with me. How can I help?"
            )
            return
        
        # Build message object
        message = WhatsAppMessage(
            message_id=data.get("id", ""),
            from_number=from_number,
            from_name=None,  # Could be fetched from contact info
            body=data.get("body", ""),
            timestamp=data.get("timestamp", time.time()),
            is_group="@g.us" in data.get("from", ""),
            group_id=data.get("from") if "@g.us" in data.get("from", "") else None,
            has_media=data.get("hasMedia", False),
        )
        
        await self._process_message(message)
    
    async def _send_web_js(self, to: str, body: str) -> None:
        """Queue a message to send via whatsapp-web.js."""
        await self._send_queue.put({
            "to": to,
            "body": body,
            "id": f"msg_{int(time.time())}"
        })
    
    async def _start_twilio_backend(self) -> None:
        """Start the Twilio webhook receiver."""
        if not self.TWILIO_AVAILABLE:
            raise ImportError(
                "Twilio backend requires 'twilio' package. "
                "Install with: pip install twilio"
            )
        
        if not all([self.account_sid, self.auth_token, self.from_number]):
            raise ValueError("Twilio backend requires account_sid, auth_token, and from_number")
        
        # Create aiohttp session for sending
        self._session = aiohttp.ClientSession()
        
        # Start webhook server for incoming messages
        if AIOHTTP_AVAILABLE:
            self._webhook_app = web.Application()
            self._webhook_app.router.add_post('/webhook/whatsapp', self._handle_twilio_webhook)
            
            self._webhook_runner = web.AppRunner(self._webhook_app)
            await self._webhook_runner.setup()
            
            site = web.TCPSite(self._webhook_runner, 'localhost', self.webhook_port)
            await site.start()
            
            logger.info(f"Twilio webhook server started on port {self.webhook_port}")
            logger.info(f"Configure Twilio webhook URL: http://your-domain:{self.webhook_port}/webhook/whatsapp")
        else:
            logger.warning("aiohttp not available. Cannot receive incoming messages.")
    
    async def _handle_twilio_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming Twilio webhook."""
        try:
            data = await request.post()
            
            from_number = data.get("From", "").replace("whatsapp:", "")
            body = data.get("Body", "")
            
            message = WhatsAppMessage(
                message_id=data.get("MessageSid", ""),
                from_number=from_number,
                from_name=None,
                body=body,
                timestamp=time.time(),
                is_group=False,
                media_url=data.get("MediaUrl0"),
                media_type=data.get("MediaContentType0"),
            )
            
            # Process asynchronously
            asyncio.create_task(self._process_message(message))
            
            # Return empty TwiML response
            from twilio.twiml.messaging_response import MessagingResponse
            resp = MessagingResponse()
            return web.Response(
                text=str(resp),
                content_type="application/xml"
            )
            
        except Exception as e:
            logger.error(f"Error handling Twilio webhook: {e}")
            return web.Response(status=500)
    
    async def _start_business_api_backend(self) -> None:
        """Start the official WhatsApp Business API webhook."""
        if not AIOHTTP_AVAILABLE:
            raise ImportError("Business API backend requires 'aiohttp'")
        
        if not self.api_token:
            raise ValueError("Business API backend requires api_token")
        
        self._session = aiohttp.ClientSession()
        
        self._webhook_app = web.Application()
        self._webhook_app.router.add_get('/webhook/whatsapp', self._verify_webhook)
        self._webhook_app.router.add_post('/webhook/whatsapp', self._handle_business_webhook)
        
        self._webhook_runner = web.AppRunner(self._webhook_app)
        await self._webhook_runner.setup()
        
        site = web.TCPSite(self._webhook_runner, 'localhost', self.webhook_port)
        await site.start()
        
        logger.info(f"WhatsApp Business API webhook server started on port {self.webhook_port}")
        logger.info(f"Configure Meta webhook URL: https://your-domain/webhook/whatsapp")
    
    async def _verify_webhook(self, request: web.Request) -> web.Response:
        """Verify webhook for WhatsApp Business API."""
        # Meta verification
        mode = request.query.get("hub.mode")
        token = request.query.get("hub.verify_token")
        challenge = request.query.get("hub.challenge")
        
        if mode == "subscribe" and token == self.webhook_secret:
            return web.Response(text=challenge)
        
        return web.Response(status=403)
    
    async def _handle_business_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming Business API webhook."""
        try:
            data = await request.json()
            
            # Process entries
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        messages = change.get("value", {}).get("messages", [])
                        
                        for msg in messages:
                            from_number = msg.get("from")
                            body = ""
                            
                            # Extract text or caption
                            if msg.get("type") == "text":
                                body = msg.get("text", {}).get("body", "")
                            elif msg.get("type") in ["image", "video", "document"]:
                                body = msg.get("caption", "[Media received]")
                            
                            message = WhatsAppMessage(
                                message_id=msg.get("id", ""),
                                from_number=from_number,
                                from_name=change.get("value", {}).get("contacts", [{}])[0].get("profile", {}).get("name"),
                                body=body,
                                timestamp=time.time(),
                                is_group=False,
                            )
                            
                            asyncio.create_task(self._process_message(message))
            
            return web.Response(text="OK")
            
        except Exception as e:
            logger.error(f"Error handling Business API webhook: {e}")
            return web.Response(status=500)
    
    async def _process_message(self, message: WhatsAppMessage) -> None:
        """Process message through the Agent."""
        user_id = self._get_user_id(message.from_number)
        
        logger.info(f"Processing WhatsApp message from {message.from_number}: {message.body[:50]}...")
        
        try:
            result = await self.agent.chat(
                user_id=user_id,
                message=message.body,
                context={
                    "channel": "whatsapp",
                    "whatsapp_from": message.from_number,
                    "whatsapp_name": message.from_name,
                    "whatsapp_is_group": message.is_group,
                    "whatsapp_backend": self.backend,
                },
            )
            
            if result.get("permission_denied"):
                await self._send_message(
                    message.from_number,
                    "⛔ You don't have permission to use this bot."
                )
                return
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                await self._send_message(
                    message.from_number,
                    f"❌ Error: {error_msg[:500]}"
                )
                return
            
            response = result.get("response", "No response")
            
            # WhatsApp has a 4096 character limit for messages
            if len(response) > 4000:
                response = response[:4000] + "\n... (message truncated)"
            
            await self._send_message(message.from_number, response)
            
            # Handle files if any were generated
            files = result.get("files_to_send", [])
            if files:
                file_list = ", ".join([f.get("filename", "unnamed") for f in files[:3]])
                if len(files) > 3:
                    file_list += f" and {len(files) - 3} more"
                
                await self._send_message(
                    message.from_number,
                    f"📎 Generated {len(files)} file(s): {file_list}\n"
                    f"Files are available via the web interface or other channels."
                )
            
        except Exception as exc:
            logger.error(f"Error processing WhatsApp message: {exc}")
            try:
                await self._send_message(
                    message.from_number,
                    "❌ An error occurred. Please try again."
                )
            except Exception:
                pass
    
    async def _send_message(self, to_number: str, body: str) -> bool:
        """Send a WhatsApp message."""
        try:
            if self.backend == "web_js":
                # Format for whatsapp-web.js
                chat_id = f"{self._normalize_number(to_number).replace('+', '')}@c.us"
                await self._send_web_js(chat_id, body)
                return True
            
            elif self.backend == "twilio":
                return await self._send_twilio(to_number, body)
            
            elif self.backend == "business_api":
                return await self._send_business_api(to_number, body)
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            return False
    
    async def _send_twilio(self, to_number: str, body: str) -> bool:
        """Send message via Twilio API."""
        if not self._session:
            return False
        
        from twilio.rest import Client
        
        client = Client(self.account_sid, self.auth_token)
        
        try:
            # Use async-friendly approach with executor
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    from_=f"whatsapp:{self.from_number}",
                    body=body,
                    to=f"whatsapp:{self._normalize_number(to_number)}"
                )
            )
            return True
        except Exception as e:
            logger.error(f"Twilio send failed: {e}")
            return False
    
    async def _send_business_api(self, to_number: str, body: str) -> bool:
        """Send message via WhatsApp Business API."""
        if not self._session:
            return False
        
        url = f"https://graph.facebook.com/v18.0/{self.from_number}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": self._normalize_number(to_number),
            "type": "text",
            "text": {"body": body}
        }
        
        try:
            async with self._session.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_token}"}
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Business API send failed: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the WhatsApp channel."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop web.js process
        if self._web_js_process:
            try:
                self._web_js_process.terminate()
                await asyncio.wait_for(self._web_js_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._web_js_process.kill()
            self._web_js_process = None
        
        # Stop webhook server
        if self._webhook_runner:
            await self._webhook_runner.cleanup()
            self._webhook_runner = None
        
        # Close session
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info("WhatsApp channel stopped")
    
    def __repr__(self) -> str:
        status = "running" if self._is_running else "stopped"
        backend_info = f", backend={self.backend}"
        restricted = f", allowed={len(self.allowed_numbers)} numbers" if self.allowed_numbers else ""
        return f"WhatsAppChannel({status}{backend_info}{restricted})"
