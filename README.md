# 🤖 LollmsBot - Your Sovereign AI Agent

LollmsBot is a secure, extensible AI assistant framework with persistent memory, multi-channel messaging, and document-aware writing capabilities. Unlike cloud-dependent bots, LollmsBot runs on your hardware, keeps your data private, and evolves with you over time.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Key Features

- **🔐 Privacy-First**: All data stays local. No cloud dependencies for core functionality.
- **🧠 Persistent Memory**: RLM (Recursive Language Model) architecture with SQLite-based memory that compresses, consolidates, and forgets naturally.
- **💬 Multi-Channel**: Discord, Telegram, WhatsApp, HTTP API, and console chat—unified under one agent.
- **📚 Document Writing**: Hierarchical document management for books, reports, and long-form content.
- **🛡️ Security Layer**: Guardian module with prompt injection detection, ethics enforcement, and self-quarantine.
- **⚡ Power Management**: Prevents Windows sleep during long operations.
- **🎭 Customizable Soul**: Personality, values, and expertise defined in markdown—evolves with your relationship.

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/ParisNeo/lollmsBot.git
cd lollmsBot
pip install -e .

# Or install from PyPI (when available)
pip install lollmsbot
```

### First Run

```bash
# Interactive setup wizard (recommended)
lollmsbot wizard

# Start the gateway
lollmsbot gateway

# Or with web UI
lollmsbot gateway --ui

# Or chat in CLI
lollmsbot chat
```

## 🔧 Configuration

### Wizard Setup

The wizard guides you through all configuration:

```bash
lollmsbot wizard
```

**Setup categories:**
- **🔗 AI Backend**: Choose from 15+ LLM providers (OpenAI, Claude, Ollama, etc.)
- **🤖 Discord**: Bot token, allowed users/guilds
- **✈️ Telegram**: Bot token from @BotFather
- **💬 WhatsApp**: Three backend options (see WhatsApp section below)
- **🧬 Soul**: Personality, values, communication style
- **💓 Heartbeat**: Self-maintenance scheduling
- **📚 Skills**: Browse and configure capabilities

### Manual Configuration

Configuration is stored in `~/.lollmsbot/config.json`:

```json
{
  "lollms": {
    "binding_name": "openai",
    "model_name": "gpt-4o-mini",
    "host_address": "https://api.openai.com/v1",
    "api_key": "sk-..."
  },
  "discord": {
    "bot_token": "MTIz...",
    "allowed_users": ["123456789"]
  },
  "telegram": {
    "bot_token": "123456:ABC-DEF..."
  },
  "whatsapp": {
    "backend": "web_js",
    "web_js_path": "~/.lollmsbot/whatsapp-bridge"
  },
  "lollmsbot": {
    "disabled_channels": "",
    "host": "0.0.0.0",
    "port": 8800
  }
}
```

## 💬 Channel Setup

### Discord

```bash
lollmsbot wizard  # Select Discord, enter bot token
```

Or manually:
1. Create app at https://discord.com/developers/applications
2. Bot → Add Bot → Copy token
3. Enable Message Content Intent
4. OAuth2 → URL Generator → bot scope → Administrator
5. Invite bot to server

### Telegram

```bash
lollmsbot wizard  # Select Telegram
```

Or:
1. Message @BotFather on Telegram
2. `/newbot` → follow instructions
3. Copy HTTP API token

### WhatsApp (Three Backend Options)

#### Option 1: whatsapp-web.js (Free, Local)

**Requirements:** Node.js 16+ installed (https://nodejs.org/)

```bash
lollmsbot wizard  # Select WhatsApp → whatsapp-web.js
```

**First-time setup:**
1. Run `lollmsbot gateway`
2. A **QR code appears as ASCII art** in the terminal (black/white blocks)
3. Open WhatsApp on phone → Settings → Linked Devices → Link a Device
4. **Point camera at the terminal screen** (not a screenshot—the actual terminal)
5. Wait for "✅ AUTHENTICATED!"

> ⚠️ **Critical**: Your phone must stay online. The bot stops working if your phone loses connection.

> ⚠️ **Testing**: You **cannot** message yourself on WhatsApp. Use a second phone or group chat.

**Number Format Issues:**
WhatsApp uses internal identifiers like `1234567890@c.us` or `1234567890@lid`. If you see "not in allowed numbers" errors with strange numbers, the bot now auto-extracts phone numbers from these formats. Use E.164 format (`+1234567890`) in your allowed/blocked lists.

#### Option 2: Twilio (Cloud, Paid)

```bash
lollmsbot wizard  # Select WhatsApp → Twilio
```

Requires Twilio account with WhatsApp-enabled number. See [Twilio WhatsApp docs](https://www.twilio.com/whatsapp).

#### Option 3: WhatsApp Business API (Official)

For production deployments. Requires Meta Business verification.

## 🎛️ CLI Commands

### Channel Control (New!)

Quickly disable/enable channels without deleting configuration:

```bash
# List all channels and their status
lollmsbot channels list

# Disable a channel
lollmsbot channels disable whatsapp
lollmsbot channels disable discord
lollmsbot channels disable all

# Re-enable
lollmsbot channels enable whatsapp

# Check status
lollmsbot channels status
```

**Environment variable** (faster for testing):
```bash
# Windows
set LOLLMSBOT_DISABLE_CHANNELS=whatsapp,discord

# Linux/Mac
export LOLLMSBOT_DISABLE_CHANNELS=whatsapp,discord

lollmsbot gateway
```

### Console Chat

Direct terminal interface without web:

```bash
lollmsbot chat
lollmsbot chat --verbose
```

### Gateway Options

```bash
lollmsbot gateway              # Basic gateway
lollmsbot gateway --ui         # With web UI at /ui
lollmsbot gateway --debug      # Rich memory display for debugging
lollmsbot gateway --host 0.0.0.0 --port 9000
```

## 🔌 API Endpoints

### Chat
```bash
curl -X POST http://localhost:8800/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "user_id": "test"}'
```

### Channel Control
```bash
# Disable channel at runtime (requires auth if not localhost)
curl -X POST http://localhost:8800/admin/channels/whatsapp/disable

# Check status
curl http://localhost:8800/admin/channels/status
```

### Document Management
```bash
# Ingest document
curl -X POST http://localhost:8800/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "url", "source": "https://example.com/article"}'

# Create book project
curl -X POST http://localhost:8800/documents/books \
  -d '{"title": "My Book", "references": ["doc_abc123"]}'

# Get writing context
curl -X POST http://localhost:8800/documents/context \
  -d '{"document_id": "book_xyz789", "detail_level": "summary"}'
```

## 🧠 Memory Architecture

LollmsBot uses **RLM (Recursive Language Model)** memory:

| Component | Purpose | Storage |
|-----------|---------|---------|
| **EMS** (External Memory Store) | Long-term compressed facts | SQLite (`~/.lollmsbot/rlm_memory.db`) |
| **RCB** (REPL Context Buffer) | Working memory for current conversation | In-memory with `[[MEMORY:...]]` handles |
| **Soul** | Identity, personality, values | Markdown (`~/.lollmsbot/soul.md`) |

Memory is automatically:
- **Compressed**: Similar memories deduplicated
- **Consolidated**: Related chunks merged into narratives
- **Forgotten**: Low-importance facts fade over time (Ebbinghaus curve)
- **Retained**: High-importance facts (your name, preferences) persist indefinitely

## 🛡️ Security Features

- **Guardian Layer**: Intercepts all inputs/outputs for injection attacks
- **Self-Quarantine**: System disables itself if compromise detected
- **No External Code**: Skills are local only, never downloaded
- **Audit Logging**: All actions logged with integrity hashes
- **Ethics Enforcement**: User-defined `ethics.md` constraints

## ⚡ Power Management (Windows)

Prevents sleep during long AI operations:

```python
# Auto-enabled in gateway
from lollmsbot.power_management import get_power_manager

# Manual control
power = get_power_manager()
power.prevent_sleep(require_display=True)   # Keep awake
power.allow_sleep()                           # Restore normal
```

## 🐛 Troubleshooting

### WhatsApp: "not in allowed numbers" with weird @lid format

**Problem**: Log shows `Blocked message from 12345@lid: not in allowed numbers`

**Solution**: 
1. Use E.164 format in allowed numbers: `+1234567890` not `1234567890`
2. The bot auto-extracts numbers from `@c.us` and `@lid` formats
3. Or disable filtering: `lollmsbot channels disable whatsapp`, edit config to remove `allowed_numbers`, then `lollmsbot channels enable whatsapp`

### WhatsApp QR code not appearing

**Problem**: No QR code in terminal

**Solutions**:
1. **Make terminal wider** (100+ characters)
2. Check Node.js: `node --version` (need 16+)
3. Install dependencies manually:
   ```bash
   cd ~/.lollmsbot/whatsapp-bridge
   npm install
   ```
4. Run with debug: `lollmsbot gateway --debug`

### WhatsApp: "Cannot message yourself"

**Problem**: Testing with your own number doesn't work

**Solution**: WhatsApp blocks self-messaging. Use:
- A friend's phone
- A second WhatsApp number
- A group chat (add bot number to group)

### "Permission denied" errors

**Problem**: Tools or channels refuse to work

**Solutions**:
1. Check user permissions in config
2. For channels: verify `allowed_users`/`allowed_numbers`
3. For tools: Guardian may be blocking—check `~/.lollmsbot/audit.log`

### High memory usage

**Problem**: Bot uses too much RAM

**Solutions**:
1. Heartbeat auto-compresses memory—ensure it's enabled
2. Reduce `max_history` in config
3. Run manual maintenance: check logs for "RLM maintenance" task

## 📁 Project Structure

```
lollmsbot/
├── agent/           # Core agent, memory (RLM), tools
├── channels/        # Discord, Telegram, WhatsApp, HTTP API
├── document_manager.py  # Hierarchical writing support
├── guardian.py      # Security & ethics layer
├── heartbeat.py     # Self-maintenance system
├── power_management.py  # Windows sleep prevention
├── skills.py        # Reusable capability system
├── soul.py          # Identity & personality
├── wizard.py        # Interactive setup
├── cli.py           # Command-line interface
└── gateway.py       # Main API server
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New LLM bindings
- Additional messaging channels
- Skill implementations
- Security enhancements

## 📜 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- Built on [LoLLMS](https://github.com/ParisNeo/lollms) ecosystem
- whatsapp-web.js for WhatsApp Web integration
- ParisNeo for the original vision of sovereign AI