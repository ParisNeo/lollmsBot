# lollmsBot 🤖
[![Apache 2.0](https://img.shields.io/github/license/ParisNeo/lollmsBot?color=blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-%231721F5.svg?&logo=docker&logoColor=white)](https://hub.docker.com/r/parisneo/lollmsbot)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)](https://github.com/ParisNeo/lollmsBot)

> **Your Personal AI Agent**  
> _Built by ParisNeo. For you. Evolves with you._

⚠️ **Alpha Software**: This is actively developed software. Core features are functional; advanced features like automated skill learning are partially implemented.

---

## 🎯 Why I Built This

Most AI assistants are **stateless services** — you prompt them, they respond, and the context disappears. They don't *know* you, and they don't *work for* you over time.

**I wanted an agent that is actually mine:**
- It **remembers** my workflows, preferences, and context across sessions  
- It **runs locally** on my machine (the controller layer), even when using remote AI models via API
- It **evolves exclusively for me** — not optimized for engagement metrics or ad revenue
- It has **extensible tools** for file operations, web requests, calendar management, and safe shell commands

### The Security Lesson

Shared skill repositories can be vulnerable to malicious code injection. **This bot never downloads executable skills from the internet.** Instead, it provides a framework for locally-defined capabilities with full auditability.

---

## ⏰ Why Now?

I didn't build a lollms agent until now because I know there is no chance in hell I could build such a thing and guarantee it doesn't get used for nefarious objectives. Even with the best-aligned models and the best possible prompt engineering, this tool—if hacked—can be used for really bad stuff.

But since **OpenClaw** was released, and basically those who want to harm already have a tool to do so, I want this tool to be on the safe side for those who need the OpenClaw-style automation with **minimized risk**.

I don't promise 100% risk-free operation, but at least it was built with a **security-first structure**. People must accept an ethical charter after installation. I know laws only obligate those who believe in them, but as I say: this is like a car. I've built it to move people from place A to place B. Using it for bad is the responsibility of the one using it.

**My commitment:**
- Guardian layer runs **before** any tool execution, not after
- **No external code execution** — skills are local and auditable
- **Full audit trail** — every action is logged with integrity hashes
- **Self-quarantine capability** — system can shut itself down if compromise is detected

---

## 🧬 Architecture: Your Agent, Your Data

### Flexible Backends (API or Local)

| Setup | Best For | Requirements |
|-------|----------|--------------|
| **API Mode** (OpenAI, Anthropic, OpenRouter, etc.) | High-quality reasoning on any hardware | API key + internet |
| **Local Mode** (Ollama, vLLM, Llama.cpp) | Privacy-sensitive work, offline use | Gaming PC or server with GPU recommended |

**Note**: High-quality local AI requires significant GPU resources (24GB+ VRAM for frontier models). The value is ensuring that *even when using commercial AI services*, your agent's memory, skills, and objectives remain **locally controlled**.

### The 7 Pillars

#### 1. 🧬 Soul — Persistent Identity
Configure personality in `~/.lollmsbot/soul.md`:
- Name, origin story, purpose
- Personality traits with intensity levels
- Core values with priorities
- Communication style preferences
- Expertise domains and limitations

#### 2. 🛡️ Guardian — Prompt Injection Defense
Security layer with:
- **Semantic analysis** of incoming text for injection patterns
- **Sandbox enforcement** for tool boundaries
- **Permission gates** for sensitive operations
- **Self-quarantine** on critical threats

#### 3. 💓 Heartbeat — Self-Maintenance Framework
Pluggable maintenance system (30min default interval):
- **Diagnostic**: Health checks, connectivity tests
- **Memory**: Compression and retention management
- **Security**: Audit log review
- **Skill**: Library maintenance
- **Optimization**: Cache cleaning

#### 4. 🧠 Memory — Your Context, Not Theirs
- Conversations stored **locally** (SQLite via `aiosqlite`)
- **Forgetting curve** applied to rarely accessed memories
- **Consolidation**: Framework for merging related memories
- **Exportable**: You own your data

#### 5. 📚 Skills — Reusable Capabilities
Capability system with:
- **Built-in skills**: File organizer, research synthesizer, meeting prep, code review
- **Skill registry**: Versioning, dependency tracking, search
- **Execution engine**: Logging, error handling, nested skill calls
- **Learning framework**: *Partially implemented* — infrastructure for creating skills from descriptions/demonstrations

#### 6. 🔧 Tools — Controlled Capabilities

| Tool | Function | Safety |
|------|----------|--------|
| `filesystem` | Read/write files, create HTML apps | Path validation, allowed directories only |
| `http` | Web requests (GET/POST/PUT/DELETE) | URL validation, timeout, retry logic, no local IPs |
| `calendar` | Event management with ICS support | Timezone-aware |
| `shell` | Command execution | Explicit allowlist, no shell injection, timeout protection |

#### 7. 🆔 Identity — Multi-Channel Presence
Same brain, different interfaces:
- **Web UI**: Real-time chat with WebSocket, tool visualization, file downloads
- **Discord**: Full bot integration with user allowlisting, file delivery via DM
- **Telegram**: Bot integration with permission controls
- **HTTP API**: REST endpoints with file download support
- **Console Chat**: Direct terminal interface for power users

---

## 🚀 Quick Start

### Option 1: One-Command Install (Fastest)
```bash
pip install lollmsbot
lollmsbot wizard  # Interactive setup
```

Requires Python 3.10+.

### Option 2: Docker (Recommended for isolation)
```bash
git clone https://github.com/ParisNeo/lollmsBot
cd lollmsBot
cp .env.example .env

# Edit .env to add your API keys or local LLM endpoint
docker-compose up -d
```

**Security default**: lollmsBot binds to `127.0.0.1` (localhost only). Remote access requires explicit configuration.

### Option 3: Development Install
```bash
git clone https://github.com/ParisNeo/lollmsBot
cd lollmsBot

# Windows
.\install.bat
.venv\Scripts\activate
lollmsbot wizard

# Linux/macOS
./install.sh
source .venv/bin/activate
lollmsbot wizard
```

---

## 🔒 Security Model

### The "No External Skills" Guarantee
- **Never downloads** executable skills from the internet
- **All capabilities** are locally defined and auditable
- **Sandboxed execution** with tool boundaries

### Prompt Injection Defense
All text processed by the agent passes through the Guardian layer before execution.

### Data Sovereignty
- API keys stored in local `.env` file, never in chat logs
- Conversation history remains local even when using remote AI models

---

## 🎮 Usage Examples

### Example 1: Create an HTML Game

```
You: Create a snake game in HTML

lollmsBot: 🔧 Using filesystem tool to create snake_game.html...

📁 I've created: snake_game.html
```

### Example 2: Check Calendar

```
You: What meetings do I have this week?

lollmsBot: [Uses calendar tool to list events]
```

### Example 3: Web Request

```
You: What's the weather at api.weather.gov?

lollmsBot: [Uses HTTP tool with safe URL validation]
```

---

## 💬 Console Chat Interface

For power users who prefer a terminal-based workflow, lollmsBot includes a rich console chat interface:

```bash
# Start console chat
lollmsbot chat

# With verbose logging
lollmsbot chat --verbose

# Custom agent name
lollmsbot chat --name "MyAssistant"
```

The console chat provides:
- **Rich terminal UI** with syntax highlighting and markdown rendering
- **Full tool access** including file generation with automatic delivery
- **Conversation history** within the session
- **Debug mode** with `--verbose` for detailed tool execution logs

Perfect for:
- Developers who live in the terminal
- Headless server environments
- Quick interactions without opening a browser
- Testing and debugging agent behavior

---

## ⚠️ Alpha Status & Roadmap

**Current State (Alpha)**:
- Core agent loop with tool integration: **stable**
- Multi-channel support (Discord, Telegram, Web, HTTP, Console): **functional**
- Guardian security layer: **operational, undergoing hardening**
- Skill execution engine: **functional**
- **Auto-skill generation**: *framework present, learning algorithms incomplete*
- **Advanced memory consolidation**: *architecture present, implementation partial*

**Known Limitations**:
- Complex multi-step planning (5+ steps) is experimental
- Skill abstraction from natural language requires refinement
- Some heartbeat maintenance tasks are framework-only

---

## 🔧 Configuration

### API Mode (Recommended for Most Users)
```bash
# .env configuration
LOLLMS_BINDING_NAME=openrouter  # or openai, anthropic, groq, etc.
LOLLMS_API_KEY=sk-or-v1-...
LOLLMS_MODEL_NAME=anthropic/claude-3.5-sonnet
```

### Local Mode (For Privacy-Critical Work)
```bash
# Requires Ollama or similar running locally
LOLLMS_BINDING_NAME=ollama
LOLLMS_HOST_ADDRESS=http://localhost:11434
LOLLMS_MODEL_NAME=llama3.2:70b
```

### Security Hardening
```bash
# Default: localhost only
LOLLMSBOT_HOST=127.0.0.1

# Optional: expose with API key (advanced users only)
LOLLMSBOT_HOST=0.0.0.0
LOLLMSBOT_API_KEY=your-strong-generated-key
```

---

## 🤝 Collaboration

This is a **solo project** — building in public. 

Contributions appreciated in:
- **Security Hardening**: Additional prompt injection vectors, sandbox testing
- **Skill Generation**: Improving abstraction from natural language descriptions
- **Documentation**: Tutorials for non-technical users
- **UI/UX**: Web interface improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to submit issues or PRs.

---

## 📄 License

Apache 2.0 — See [LICENSE](LICENSE)

---

## 📊 Implementation Status

This table tracks what's implemented vs. planned. Updated as development progresses.

| Feature | Status | Notes |
|---------|--------|-------|
| **Core Architecture** |||
| Agent message loop | ✅ Stable | Full tool integration, async processing |
| LoLLMS client integration | ✅ Stable | Multiple binding support via lollms-client |
| Permission system | ✅ Stable | User-level permissions, tool allow/deny lists |
| **The 7 Pillars** |||
| 🧬 Soul (Identity) | ✅ Implemented | Personality, traits, values, communication style |
| 🛡️ Guardian (Security) | ⚠️ Partial | Pattern-based injection detection; ML-enhanced detection planned |
| 💓 Heartbeat (Self-maintenance) | ⚠️ Partial | Framework complete; some tasks are placeholders |
| 🧠 Memory (Persistence) | ✅ Implemented | SQLite storage, conversation history |
| Memory compression/consolidation | 🔲 Planned | Architecture present, algorithms incomplete |
| 📚 Skills (Capabilities) | ⚠️ Partial | Registry, execution, built-in skills work; auto-learning framework only |
| 🔧 Tools | ✅ Implemented | Filesystem, HTTP, Calendar, Shell with safety controls |
| 🆔 Identity (Multi-channel) | ✅ Implemented | Web UI, Discord, Telegram, HTTP API, Console Chat |
| **Channels** |||
| Web UI | ✅ Stable | WebSocket chat, file downloads, tool visualization |
| Discord | ✅ Stable | Full bot with DM file delivery, user allowlisting |
| Telegram | ✅ Stable | Bot with permission controls |
| HTTP API | ✅ Stable | REST endpoints, file downloads |
| Console Chat | ✅ Stable | Rich terminal interface for power users |
| Slack | 🔲 Planned | Dependencies present, implementation pending |
| **Security Features** |||
| Prompt injection detection | ⚠️ Basic | Regex patterns; semantic ML analysis planned |
| Tool sandboxing | ✅ Implemented | Path validation, command allowlist |
| Self-quarantine | ✅ Implemented | Guardian can disable operations on threats |
| Audit logging | ✅ Implemented | Security events logged with integrity hashes |
| **Advanced Features** |||
| Skill auto-generation | 🔲 Planned | Framework exists; natural language→skill incomplete |
| Memory forgetting curve | 🔲 Planned | Ebbinghaus-inspired retention not yet active |
| Self-healing | 🔲 Planned | Auto-fix framework present, healing logic incomplete |
| Self-update | 🔲 Planned | Conceptual; would require external update mechanism |
| **Integrations** |||
| Discord voice | 🔲 Planned | Library supports it, no voice handling code yet |
| File encryption at rest | 🔲 Planned | Security model describes it, not implemented |
| **Legend:** ✅ Implemented & Stable | ⚠️ Partial / In Progress | 🔲 Planned / Not Started |

*Last updated: 2026-02-08*

---

**Built by ParisNeo**  
*For the individual, in the spirit of open collaboration.*