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

import os

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
                logger.info(f"✅ Node.js available: {result.stdout.strip()}")
            else:
                logger.warning("⚠️  Node.js returned error code")
                self.WEB_JS_AVAILABLE = False
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"⚠️  Node.js not found: {e}")
            logger.warning("   Install from: https://nodejs.org/")
            self.WEB_JS_AVAILABLE = False
    
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
        # Log raw incoming number for debugging
        logger.debug(f"Checking interaction for raw: '{phone_number}'")
        
        # Normalize phone number
        normalized = self._normalize_number(phone_number)
        logger.debug(f"  Normalized to: '{normalized}'")
        logger.debug(f"  Allowed numbers configured: {self.allowed_numbers}")
        
        if normalized in self.blocked_numbers:
            logger.debug(f"  -> BLOCKED: number in blocked list")
            return False, "number blocked"
        
        if self.allowed_numbers and normalized not in self.allowed_numbers:
            logger.info(f"  -> BLOCKED: {normalized} not in allowed list {self.allowed_numbers}")
            # Also check if raw format matches any allowed (for debugging)
            for allowed in self.allowed_numbers:
                if allowed in phone_number or phone_number in allowed:
                    logger.info(f"     (Note: partial match with {allowed})")
            return False, "not in allowed numbers"
        
        # Check confirmation requirement
        if self.require_confirmation and is_first_message and normalized not in self._confirmed_users:
            return False, "confirmation_required"
        
        logger.debug(f"  -> ALLOWED")
        return True, ""
    
    def _normalize_number(self, number: str) -> str:
        """Normalize phone number to E.164 format.
        
        Handles WhatsApp contact IDs like:
        - 1234567890@c.us (standard user)
        - 1234567890@lid (WhatsApp Business/Enterprise)
        - 1234567890@g.us (group)
        - +1234567890 (already normalized)
        """
        # First, extract number from WhatsApp contact ID
        if "@" in number:
            # Extract the part before @ (the phone number or internal ID)
            number_part = number.split("@")[0]
            
            # @lid sometimes contains non-numeric identifiers
            # Try to extract digits only, or use as-is if it looks like a number
            if number_part.isdigit():
                cleaned = number_part
            else:
                # For @lid or other formats, extract digits or use raw
                import re
                digits_only = re.sub(r'\D', '', number_part)
                if digits_only:
                    cleaned = digits_only
                else:
                    # Can't extract number, return as-is for logging/debugging
                    return number
        else:
            cleaned = number
        
        # Remove formatting characters
        cleaned = cleaned.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        
        # Ensure it starts with +
        if not cleaned.startswith("+"):
            # If it's 10 digits, assume US (+1)
            if len(cleaned) == 10:
                cleaned = "+1" + cleaned
            # If it's more than 10 digits, probably has country code
            elif len(cleaned) > 10:
                cleaned = "+" + cleaned
            else:
                # Short number, add + anyway
                cleaned = "+" + cleaned
        
        return cleaned
    
    def _get_user_id(self, phone_number: str) -> str:
        """Generate consistent user ID for Agent."""
        # Use raw identifier if we can't normalize properly (for logging/debugging)
        normalized = self._normalize_number(phone_number)
        # If normalization failed (returned original with @), use a fallback
        if "@" in normalized:
            # Extract what we can for a stable ID
            safe_id = normalized.replace("@", "_").replace(".", "_")
            return f"whatsapp:{safe_id}"
        return f"whatsapp:{normalized}"
    
    async def start(self) -> None:
        """Start the WhatsApp channel."""
        logger.info("=" * 60)
        logger.info("📱 WhatsApp Channel Starting...")
        logger.info(f"   Backend: {self.backend}")
        logger.info(f"   Web JS Path: {self.web_js_path}")
        logger.info(f"   Allowed numbers: {len(self.allowed_numbers)}")
        logger.info(f"   Blocked numbers: {len(self.blocked_numbers)}")
        logger.info("=" * 60)
        
        if self._is_running:
            logger.warning("WhatsApp channel is already running")
            return
        
        try:
            if self.backend == "web_js":
                logger.info("🔧 Initializing web.js backend...")
                await self._start_web_js_backend()
            elif self.backend == "twilio":
                logger.info("🔧 Initializing Twilio backend...")
                await self._start_twilio_backend()
            elif self.backend == "business_api":
                logger.info("🔧 Initializing Business API backend...")
                await self._start_business_api_backend()
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
            
            self._is_running = True
            logger.info(f"✅ WhatsApp channel started successfully with {self.backend} backend")
            
        except Exception as e:
            logger.error(f"❌ WhatsApp channel failed to start: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise
    
    async def _start_web_js_backend(self) -> None:
        """Start the whatsapp-web.js bridge."""
        logger.info("🔍 Step 1: Checking Node.js availability...")
        
        # Double-check Node.js is actually available
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Node.js not working properly")
            logger.info(f"✅ Node.js detected: {result.stdout.strip()}")
        except Exception as e:
            logger.error("╔════════════════════════════════════════════════════════════╗")
            logger.error("║  ❌ NODE.JS NOT FOUND                                      ║")
            logger.error("║                                                            ║")
            logger.error("║  WhatsApp web.js backend requires Node.js.                 ║")
            logger.error("║  Install from: https://nodejs.org/ (LTS version)          ║")
            logger.error("╚════════════════════════════════════════════════════════════╝")
            raise RuntimeError(f"Node.js required but not found: {e}")
        
        # Ensure bridge script exists (DEFINE bridge_js FIRST)
        logger.info("🔍 Step 2: Setting up bridge script...")
        bridge_js = Path(self.web_js_path) / "bridge.js"
        bridge_js.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"   Bridge directory: {bridge_js.parent}")
        
        if not bridge_js.exists():
            logger.info("   Creating bridge.js script...")
            bridge_js.write_text(self._get_bridge_script())
            logger.info(f"✅ Created whatsapp-web.js bridge at {bridge_js}")
        else:
            logger.info(f"   Bridge script exists: {bridge_js}")
        
        # Check if dependencies are already installed (skip npm check if they are)
        logger.info("🔍 Step 3: Checking dependencies...")
        node_modules = bridge_js.parent / "node_modules"
        package_json = bridge_js.parent / "package.json"
        
        if not package_json.exists():
            logger.info("   Creating package.json...")
            package_json.write_text(json.dumps({
                "name": "lollmsbot-whatsapp-bridge",
                "version": "1.0.0",
                "dependencies": {
                    "whatsapp-web.js": "^1.23.0",
                    "qrcode-terminal": "^0.12.0"
                }
            }, indent=2))
        
        # Check if dependencies already exist
        if node_modules.exists():
            logger.info(f"✅ Dependencies already installed at: {node_modules}")
            logger.info("   Skipping npm check - using existing dependencies")
            skip_npm_check = True
        else:
            skip_npm_check = False
        
        # Only check npm if we need to install
        npm_cmd = "npm"
        if not skip_npm_check:
            logger.info("🔍 Checking npm availability...")
            npm_found = False
            
            # Try multiple methods to find npm
            try:
                # Method 1: Try npm directly (might work if in PATH)
                npm_result = subprocess.run(
                    ["npm", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if npm_result.returncode == 0:
                    logger.info(f"✅ npm detected in PATH: {npm_result.stdout.strip()}")
                    npm_found = True
            except FileNotFoundError:
                logger.info("   npm not in PATH, trying alternative methods...")
            
            # Method 2: Try common Windows paths
            if not npm_found and os.name == 'nt':
                possible_paths = [
                    os.path.expandvars(r"%ProgramFiles%\nodejs\npm.cmd"),
                    os.path.expandvars(r"%ProgramFiles(x86)%\nodejs\npm.cmd"),
                    os.path.expandvars(r"%APPDATA%\npm\npm.cmd"),
                    os.path.expandvars(r"%LOCALAPPDATA%\Programs\nodejs\npm.cmd"),
                    r"C:\Program Files\nodejs\npm.cmd",
                    r"C:\Program Files (x86)\nodejs\npm.cmd",
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        npm_cmd = path
                        npm_found = True
                        logger.info(f"✅ npm found at: {path}")
                        break
            
            # Method 3: Try where command
            if not npm_found:
                try:
                    where_result = subprocess.run(
                        ["where", "npm"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if where_result.returncode == 0 and where_result.stdout.strip():
                        npm_cmd = where_result.stdout.strip().split('\n')[0].strip()
                        npm_found = True
                        logger.info(f"✅ npm found via 'where': {npm_cmd}")
                except Exception:
                    pass
            
            if not npm_found:
                logger.warning("⚠️  npm not found in PATH, but will try to continue...")
                logger.warning("   If dependencies are missing, you'll need to install them manually:")
                logger.warning(f"   cd \"{bridge_js.parent}\"")
                logger.warning("   npm install")
                # Don't raise error - let it fail later if actually needed
        
        # Install dependencies if needed
        if not node_modules.exists():
            logger.info("📦 Dependencies not found, attempting to install...")
            
            try:
                # Run npm install with visible output
                proc = await asyncio.create_subprocess_exec(
                    npm_cmd, "install",
                    cwd=str(bridge_js.parent),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT
                )
                
                # Stream npm output
                npm_output = []
                line_count = 0
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    line_count += 1
                    text = line.decode().strip()
                    npm_output.append(text)
                    if "error" in text.lower() or "ERR!" in text:
                        logger.error(f"   npm: {text}")
                    else:
                        logger.info(f"   npm: {text[:80]}")
                
                await proc.wait()
                
                if proc.returncode != 0:
                    raise RuntimeError(f"npm install failed with code {proc.returncode}")
                
                logger.info(f"✅ Dependencies installed successfully ({line_count} lines of output)")
                
            except FileNotFoundError:
                logger.error("╔════════════════════════════════════════════════════════════╗")
                logger.error("║  ❌ CANNOT RUN NPM                                         ║")
                logger.error("║                                                            ║")
                logger.error("║  npm not found. Please install dependencies manually:      ║")
                logger.error(f"║  cd \"{bridge_js.parent}\"                                  ║")
                logger.error("║  npm install                                               ║")
                logger.error("╚════════════════════════════════════════════════════════════╝")
                raise RuntimeError("npm not found and dependencies not installed")
        else:
            logger.info(f"   Using existing dependencies at: {node_modules}")
        
        # Start the bridge process
        logger.info("")
        logger.info("🔍 Step 4: Starting Node.js bridge process...")
        logger.info("╔════════════════════════════════════════════════════════════╗")
        logger.info("║  📱 WhatsApp Web Authentication                            ║")
        logger.info("║                                                            ║")
        logger.info("║  A QR code will appear BELOW this message shortly.         ║")
        logger.info("║                                                            ║")
        logger.info("║  To authenticate:                                          ║")
        logger.info("║  1. Open WhatsApp on your phone                            ║")
        logger.info("║  2. Go to: Settings → Linked Devices → Link a Device       ║")
        logger.info("║  3. Point your camera at the QR code that appears          ║")
        logger.info("║  4. Wait for 'Authenticated!' message                      ║")
        logger.info("║                                                            ║")
        logger.info("║  ⚠️  Your phone must stay connected to the internet       ║")
        logger.info("║      for the bot to work (not just for QR scan)            ║")
        logger.info("╚════════════════════════════════════════════════════════════╝")
        logger.info("")
        logger.info(f"🚀 Executing: node {bridge_js}")
        logger.info(f"   Working directory: {bridge_js.parent}")
        logger.info("   (QR code will appear as ASCII art below)")
        logger.info("")
        
        try:
            self._web_js_process = await asyncio.create_subprocess_exec(
                "node", str(bridge_js),
                cwd=str(bridge_js.parent),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT  # Merge to capture QR code
            )
            logger.info(f"✅ Node.js process started (PID: {self._web_js_process.pid})")
        except Exception as e:
            logger.error(f"❌ Failed to start Node.js process: {e}")
            raise
        
        # Start reader tasks
        logger.info("🔍 Step 5: Starting background tasks...")
        asyncio.create_task(self._read_web_js_output())
        asyncio.create_task(self._process_send_queue())
        asyncio.create_task(self._monitor_process_health())
        logger.info("✅ Background tasks started")
    
    def _get_bridge_script(self) -> str:
        """Get the Node.js bridge script for whatsapp-web.js."""
        return '''
const { Client, LocalAuth } = require('whatsapp-web.js');

// Custom QR code display that works with subprocess capture
function displayQR(qr) {
    // Use qrcode-terminal with explicit stdout write
    const qrcode = require('qrcode-terminal');
    
    // Force output to stdout with explicit flush
    process.stdout.write('\\n');
    process.stdout.write('╔════════════════════════════════════════════════════════════╗\\n');
    process.stdout.write('║  📱 SCAN THIS QR CODE WITH YOUR PHONE                      ║\\n');
    process.stdout.write('║                                                            ║\\n');
    process.stdout.write('║  1. Open WhatsApp → Settings → Linked Devices             ║\\n');
    process.stdout.write('║  2. Tap "Link a Device"                                    ║\\n');
    process.stdout.write('║  3. Point camera at the QR code below                     ║\\n');
    process.stdout.write('╚════════════════════════════════════════════════════════════╝\\n');
    process.stdout.write('\\n');
    
    // Generate QR code - this writes directly to terminal
    qrcode.generate(qr, { small: false });
    
    // Ensure output is flushed
    process.stdout.write('\\n');
    process.stdout.write('QR_RECEIVED\\n');
    process.stdout.flush();
}

const client = new Client({
    authStrategy: new LocalAuth(),
    puppeteer: {
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    }
});

client.on('qr', (qr) => {
    displayQR(qr);
});

client.on('ready', () => {
    process.stdout.write('\\n');
    process.stdout.write('╔════════════════════════════════════════════════════════════╗\\n');
    process.stdout.write('║  ✅ AUTHENTICATED! WhatsApp Web is ready                   ║\\n');
    process.stdout.write('╚════════════════════════════════════════════════════════════╝\\n');
    process.stdout.write('READY\\n');
    process.stdout.flush();
});

client.on('authenticated', () => {
    process.stdout.write('Authenticated! Session saved.\\n');
    process.stdout.flush();
});

client.on('auth_failure', (msg) => {
    process.stdout.write('ERROR:Authentication failed: ' + msg + '\\n');
    process.stdout.flush();
});

client.on('disconnected', (reason) => {
    process.stdout.write('ERROR:Disconnected: ' + reason + '\\n');
    process.stdout.flush();
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
    process.stdout.write('MSG:' + JSON.stringify(data) + '\\n');
    process.stdout.flush();
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
                process.stdout.write('SENT:' + cmd.id + '\\n');
                process.stdout.flush();
            }).catch((err) => {
                process.stdout.write('ERROR:' + err.message + '\\n');
                process.stdout.flush();
            });
        }
    } catch (e) {
        process.stdout.write('PARSE_ERROR:' + e.message + '\\n');
        process.stdout.flush();
    }
});

// Log startup
process.stdout.write('BRIDGE_STARTING\\n');
process.stdout.flush();

client.initialize().then(() => {
    process.stdout.write('CLIENT_INITIALIZED\\n');
    process.stdout.flush();
}).catch((err) => {
    process.stdout.write('INIT_ERROR:' + err.message + '\\n');
    process.stdout.flush();
});
'''
    
    async def _read_web_js_output(self) -> None:
        """Read output from whatsapp-web.js process."""
        if not self._web_js_process:
            logger.error("❌ No web.js process to read from")
            return
        
        qr_printed = False
        line_count = 0
        qr_lines = []  # Collect lines that might be QR code
        
        logger.info("📡 Reading from Node.js process...")
        logger.info("   (QR code will appear below as ASCII art - look for black/white blocks)")
        
        while self._is_running and self._web_js_process:
            try:
                line = await self._web_js_process.stdout.readline()
                line_count += 1
                
                if not line:
                    # EOF - process may have exited
                    logger.warning(f"⚠️  Node.js process closed (read {line_count} lines)")
                    break
                
                # Decode without stripping to preserve QR code structure
                text = line.decode('utf-8', errors='replace')
                stripped = text.strip()
                
                # Print EVERYTHING to console for first 100 lines (not just debug)
                if line_count <= 100:
                    # Use print directly for QR code visibility
                    if line_count == 1:
                        print()  # New line before output
                    print(text, end='')  # Preserve original line endings
                
                # Check for specific events in stripped text
                if stripped == "READY":
                    logger.info("✅ WhatsApp Web authenticated and ready!")
                    
                elif stripped == "QR_RECEIVED":
                    if not qr_printed:
                        logger.info("")
                        logger.info("╔════════════════════════════════════════════════════════════╗")
                        logger.info("║  📱 QR CODE WAS PRINTED ABOVE - SCROLL UP TO SEE IT!      ║")
                        logger.info("║                                                            ║")
                        logger.info("║  Look for the black and white square ASCII art above.      ║")
                        logger.info("║  If you don't see it, check that your terminal window      ║")
                        logger.info("║  is wide enough (80+ characters recommended).              ║")
                        logger.info("╚════════════════════════════════════════════════════════════╝")
                        logger.info("")
                        qr_printed = True
                        
                elif stripped == "BRIDGE_STARTING":
                    logger.info("🚀 Bridge script is starting...")
                    
                elif stripped == "CLIENT_INITIALIZED":
                    logger.info("🔄 WhatsApp client initializing (this may take 10-30 seconds)...")
                    
                elif stripped.startswith("INIT_ERROR:"):
                    logger.error(f"❌ Bridge initialization failed: {stripped[11:]}")
                    
                elif stripped.startswith("MSG:"):
                    try:
                        data = json.loads(stripped[4:])
                        await self._handle_incoming_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"   Failed to parse message: {e}")
                        
                elif stripped.startswith("SENT:"):
                    logger.debug(f"   Message sent: {stripped[5:]}")
                    
                elif stripped.startswith("ERROR:"):
                    logger.error(f"   WhatsApp error: {stripped[6:]}")
                    
                elif "Authenticated" in stripped or "authenticated" in stripped.lower():
                    logger.info("🔓 WhatsApp session authenticated!")
                    
            except Exception as e:
                logger.error(f"   Error reading output: {e}")
                await asyncio.sleep(1)
        
        logger.warning("📡 Node.js output reader stopped")
    
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
    
    async def _monitor_process_health(self) -> None:
        """Monitor if the Node.js process stays alive."""
        await asyncio.sleep(5)  # Give it time to start
        
        if not self._web_js_process:
            return
        
        for _ in range(3):  # Check a few times
            await asyncio.sleep(2)
            
            if self._web_js_process.returncode is not None:
                logger.error(f"❌ Node.js process exited with code: {self._web_js_process.returncode}")
                logger.error("   The QR code may not have been generated.")
                logger.error("   Common causes:")
                logger.error("   • Node.js version too old (need 16+)")
                logger.error("   • Puppeteer/Chromium failed to launch")
                logger.error("   • Missing dependencies")
                return
        
        logger.info("✅ Node.js process is running normally")
    
    async def stop(self) -> None:
        """Stop the WhatsApp channel."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop web.js process
        if self._web_js_process:
            logger.info("Stopping Node.js process...")
            try:
                self._web_js_process.terminate()
                await asyncio.wait_for(self._web_js_process.wait(), timeout=5.0)
                logger.info("✅ Node.js process stopped cleanly")
            except asyncio.TimeoutError:
                logger.warning("⚠️  Node.js process didn't terminate, killing...")
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
