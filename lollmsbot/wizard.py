#!/usr/bin/env python
"""
lollmsBot Interactive Setup Wizard - Skills Edition

Now includes:
- Binding-first backend configuration (remote vs local bindings)
- Soul configuration (personality, identity, values)
- Heartbeat settings (self-maintenance frequency, tasks)
- Memory monitoring (compression, retention, optimization)
- Skills management (browse, test, create, configure)
- Ethical Charter acceptance
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
    from rich.markdown import Markdown
    import questionary
    from questionary import Choice
except ImportError:
    print("❌ Install dev deps: pip install -e .[dev]")
    exit(1)

from lollmsbot.config import LollmsSettings
from lollmsbot.lollms_client import build_lollms_client
from lollmsbot.soul import Soul, PersonalityTrait, TraitIntensity, ValueStatement, CommunicationStyle, ExpertiseDomain
from lollmsbot.heartbeat import Heartbeat, HeartbeatConfig, MaintenanceTask, get_heartbeat
from lollmsbot.skills import SkillRegistry, SkillComplexity, get_skill_registry, SkillLearner


console = Console()


@dataclass
class BindingInfo:
    """Information about an LLM binding."""
    name: str
    display_name: str
    category: str  # "remote", "local_server", "local_direct"
    description: str
    default_host: Optional[str] = None
    requires_api_key: bool = True
    supports_ssl_verify: bool = True
    requires_models_path: bool = False
    default_model: Optional[str] = None


# Binding registry - all available bindings
AVAILABLE_BINDINGS: Dict[str, BindingInfo] = {
    # Remote / SaaS bindings
    "lollms": BindingInfo(
        name="lollms",
        display_name="🔗 LoLLMS (Default)",
        category="remote",
        description="LoLLMS WebUI - Local or remote LoLLMS server",
        default_host="http://localhost:9600",
        requires_api_key=False,  # Optional for local, required for remote
        supports_ssl_verify=True,
        default_model=None,
    ),
    "openai": BindingInfo(
        name="openai",
        display_name="🤖 OpenAI",
        category="remote",
        description="OpenAI GPT models (GPT-4, GPT-3.5, etc.)",
        default_host="https://api.openai.com/v1",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="gpt-4o-mini",
    ),
    "azure_openai": BindingInfo(
        name="azure_openai",
        display_name="☁️ Azure OpenAI",
        category="remote",
        description="Microsoft Azure OpenAI Service",
        default_host="https://YOUR_RESOURCE.openai.azure.com/",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="gpt-4",
    ),
    "claude": BindingInfo(
        name="claude",
        display_name="🧠 Anthropic Claude",
        category="remote",
        description="Anthropic Claude models",
        default_host="https://api.anthropic.com",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="claude-3-5-sonnet-20241022",
    ),
    "gemini": BindingInfo(
        name="gemini",
        display_name="💎 Google Gemini",
        category="remote",
        description="Google Gemini models",
        default_host="https://generativelanguage.googleapis.com",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="gemini-1.5-flash",
    ),
    "groq": BindingInfo(
        name="groq",
        display_name="⚡ Groq",
        category="remote",
        description="Groq ultra-fast inference",
        default_host="https://api.groq.com/openai/v1",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="llama-3.1-8b-instant",
    ),
    "grok": BindingInfo(
        name="grok",
        display_name="🐦 xAI Grok",
        category="remote",
        description="xAI Grok models",
        default_host="https://api.x.ai/v1",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="grok-2",
    ),
    "mistral": BindingInfo(
        name="mistral",
        display_name="🌊 Mistral AI",
        category="remote",
        description="Mistral AI models",
        default_host="https://api.mistral.ai/v1",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="mistral-small-latest",
    ),
    "ollama": BindingInfo(
        name="ollama",
        display_name="🦙 Ollama",
        category="local_server",
        description="Ollama local LLM server",
        default_host="http://localhost:11434",
        requires_api_key=False,  # Local by default, key optional for proxy
        supports_ssl_verify=False,  # Usually local
        default_model="llama3.2",
    ),
    "open_router": BindingInfo(
        name="open_router",
        display_name="🌐 OpenRouter",
        category="remote",
        description="OpenRouter - unified API for many models",
        default_host="https://openrouter.ai/api/v1",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="meta-llama/llama-3.1-8b-instruct",
    ),
    "perplexity": BindingInfo(
        name="perplexity",
        display_name="❓ Perplexity",
        category="remote",
        description="Perplexity AI API",
        default_host="https://api.perplexity.ai",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="llama-3.1-sonar-small-128k-online",
    ),
    "novita_ai": BindingInfo(
        name="novita_ai",
        display_name="✨ Novita AI",
        category="remote",
        description="Novita AI inference platform",
        default_host="https://api.novita.ai/v3/openai",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model="meta-llama/llama-3.1-8b-instruct",
    ),
    "litellm": BindingInfo(
        name="litellm",
        display_name="📡 LiteLLM",
        category="remote",
        description="LiteLLM proxy/gateway",
        default_host="http://localhost:4000",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model=None,
    ),
    "hugging_face_inference_api": BindingInfo(
        name="hugging_face_inference_api",
        display_name="🤗 Hugging Face",
        category="remote",
        description="Hugging Face Inference API",
        default_host="https://api-inference.huggingface.co",
        requires_api_key=True,
        supports_ssl_verify=True,
        default_model=None,
    ),
    "openllm": BindingInfo(
        name="openllm",
        display_name="🔧 OpenLLM",
        category="local_server",
        description="BentoML OpenLLM serving",
        default_host="http://localhost:3000",
        requires_api_key=False,
        supports_ssl_verify=True,
        default_model=None,
    ),
    "openwebui": BindingInfo(
        name="openwebui",
        display_name="🌟 OpenWebUI",
        category="local_server",
        description="OpenWebUI backend",
        default_host="http://localhost:8080",
        requires_api_key=True,  # OpenWebUI uses API keys
        supports_ssl_verify=True,
        default_model=None,
    ),
    # Local direct bindings
    "llama_cpp_server": BindingInfo(
        name="llama_cpp_server",
        display_name="🦙 Llama.cpp (Server)",
        category="local_server",
        description="llama.cpp server mode (local)",
        default_host="http://localhost:8080",
        requires_api_key=False,
        supports_ssl_verify=False,
        requires_models_path=True,
        default_model=None,
    ),
    "vllm": BindingInfo(
        name="vllm",
        display_name="🔥 vLLM",
        category="local_server",
        description="vLLM high-throughput inference",
        default_host="http://localhost:8000",
        requires_api_key=False,
        supports_ssl_verify=True,
        requires_models_path=False,
        default_model=None,
    ),
    "tensor_rt": BindingInfo(
        name="tensor_rt",
        display_name="🚀 TensorRT",
        category="local_direct",
        description="NVIDIA TensorRT LLM (local)",
        default_host=None,
        requires_api_key=False,
        supports_ssl_verify=False,
        requires_models_path=True,
        default_model=None,
    ),
    "transformers": BindingInfo(
        name="transformers",
        display_name="🤗 Transformers",
        category="local_direct",
        description="Hugging Face Transformers (local)",
        default_host=None,
        requires_api_key=False,
        supports_ssl_verify=False,
        requires_models_path=True,
        default_model=None,
    ),
}


# Ethical Charter that users must accept
ETHICAL_CHARTER = """
# lollmsBot Ethical Charter

## Preamble

lollmsBot is a powerful tool designed to automate tasks and extend your capabilities 
through AI. With this power comes responsibility. This charter exists to ensure every 
user understands both the potential and the boundaries of ethical use.

## Core Principles

### 1. **Do No Harm**
I will not use lollmsBot to:
- Cause physical, emotional, or financial harm to individuals
- Harass, intimidate, or discriminate against any person or group
- Create, distribute, or deploy malware, exploits, or destructive software
- Engage in unauthorized access to systems or data (hacking)
- Automate attacks, spam, or denial-of-service actions

### 2. **Respect Privacy**
I will:
- Only process data I have legitimate rights to access
- Respect data protection laws and individual privacy rights
- Not use lollmsBot for unauthorized surveillance or data harvesting
- Protect sensitive information with appropriate security measures

### 3. **Transparency**
I will:
- Be honest about AI-generated content when required by context or law
- Not use lollmsBot to deceive, defraud, or manipulate others
- Disclose automation when it could materially affect others' decisions
- Take responsibility for actions performed through this tool

### 4. **Legal Compliance**
I will:
- Use lollmsBot in compliance with all applicable laws and regulations
- Respect intellectual property rights and licensing terms
- Not automate illegal activities, regardless of enforcement likelihood
- Understand that laws vary by jurisdiction and act accordingly

### 5. **Accountability**
I understand that:
- lollmsBot is like a car—built for legitimate transportation, but misuse is my responsibility
- Security features protect against accidental harm, not intentional abuse
- I am the final arbiter of ethical judgment for my specific use case
- This charter doesn't cover every scenario—I must use my conscience

## Security-First Design Acknowledgment

lollmsBot was built with security as a core architectural principle:

- **No External Code Execution**: Skills are local and auditable, never downloaded from the internet
- **Guardian Layer**: All inputs pass through security screening before any action
- **Sandboxed Tools**: Each tool operates within strict boundaries
- **Audit Trail**: Every action is logged with integrity verification
- **Self-Quarantine**: System can disable itself if compromise is detected

These protections exist to minimize **accidental** harm and **unauthorized** use. 
They cannot prevent deliberate misuse by someone with full system access.

## Why This Matters

The release of tools like SimplifiedAgant demonstrates that those who want to cause harm 
already have means to do so. lollmsBot exists to provide the **same automation power** 
to those with legitimate needs—researchers, developers, system administrators, 
and power users—while embedding ethical guardrails into its very architecture.

By accepting this charter, you join a community that believes powerful tools should 
be available to responsible actors, not locked away while bad actors operate freely.

## Acceptance

**By continuing with lollmsBot setup, you affirm that:**

1. You have read and understood this charter
2. You commit to using lollmsBot ethically and legally
3. You accept responsibility for actions performed with this tool
4. You understand that no software can guarantee 100% safety—human judgment is essential

*If you cannot accept these terms, please exit the wizard now. lollmsBot is not 
for everyone, and that's by design.*
"""


class Wizard:
    """Interactive setup wizard for lollmsBot services - Full 7 Pillars Edition."""

    def __init__(self):
        self.config_path = Path.home() / ".lollmsbot" / "config.json"
        self.ethics_path = Path.home() / ".lollmsbot" / "ethics_accepted"
        self.config_path.parent.mkdir(exist_ok=True)
        self.config: Dict[str, Dict[str, Any]] = self._load_config()
        
        # Initialize subsystems for configuration
        self.soul = Soul()
        self.heartbeat = get_heartbeat()
        self.skill_registry = get_skill_registry()
        
        # Track what's been configured
        self._configured: set = set()

    def _load_config(self) -> Dict[str, Dict[str, Any]]:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {}

    def _save_config(self) -> None:
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _show_ethical_charter(self) -> bool:
        """Display and require acceptance of ethical charter. Returns True if accepted."""
        console.clear()
        
        # Display charter with nice formatting
        charter_panel = Panel(
            Markdown(ETHICAL_CHARTER),
            title="[bold red]⚖️ Ethical Charter[/bold red]",
            border_style="red",
            padding=(1, 2)
        )
        console.print(charter_panel)
        console.print()
        
        # Check if already accepted
        if self.ethics_path.exists():
            accepted_hash = self.ethics_path.read_text().strip()
            current_hash = hashlib.sha256(ETHICAL_CHARTER.encode()).hexdigest()[:16]
            
            if accepted_hash == current_hash:
                console.print("[green]✅ Ethical charter previously accepted (version matches)[/]")
                return True
            else:
                console.print("[yellow]⚠️ Ethical charter has been updated since your last acceptance[/]")
        
        # Require explicit acceptance
        console.print("[bold]You must accept the ethical charter to proceed with lollmsBot setup.[/]")
        console.print()
        
        accepted = questionary.confirm(
            "I have read and agree to the Ethical Charter above. "
            "I commit to using lollmsBot responsibly and ethically.",
            default=False
        ).ask()
        
        if not accepted:
            console.print()
            console.print(Panel(
                "[bold]Setup cannot continue without ethical charter acceptance.[/]\n\n"
                "lollmsBot is designed for users who take responsibility for their actions. "
                "If this doesn't align with your intentions, please exit now.\n\n"
                "The wizard will close in 5 seconds...",
                title="[bold red]❌ Acceptance Required[/bold red]",
                border_style="red"
            ))
            import time
            time.sleep(5)
            return False
        
        # Record acceptance with hash of charter content
        charter_hash = hashlib.sha256(ETHICAL_CHARTER.encode()).hexdigest()[:16]
        self.ethics_path.write_text(charter_hash)
        
        console.print()
        console.print("[bold green]✅ Ethical charter accepted. Thank you for committing to responsible use.[/]")
        console.print()
        questionary.press_any_key_to_continue().ask()
        
        return True

    def run_wizard(self) -> None:
        """Main wizard loop - Full Edition with all 7 Pillars."""
        console.clear()
        
        # Beautiful animated banner
        banner = Panel.fit(
            Text.assemble(
                ("🧬 ", "bold magenta"),
                ("lollmsBot", "bold cyan"),
                (" Setup Wizard\n", "bold blue"),
                ("Configure your ", "dim"),
                ("sovereign AI companion", "italic green"),
            ),
            border_style="bright_blue",
            padding=(1, 4),
        )
        console.print(banner)
        console.print()

        # Show current status
        self._show_status_tree()

        # Present and require ethical charter acceptance
        if not self._show_ethical_charter():
            console.print("\n[yellow]👋 Exiting without configuration.[/]")
            return

        while True:
            action = questionary.select(
                "What would you like to configure?",
                choices=[
                    Choice("🔗 AI Backend (Select Binding First)", "lollms"),
                    Choice("🤖 Discord Channel", "discord"),
                    Choice("✈️ Telegram Channel", "telegram"),
                    Choice("💬 WhatsApp Channel", "whatsapp"),
                    Choice("💬 Slack Channel", "slack"),
                    Choice("🧬 Soul (Personality & Identity)", "soul"),
                    Choice("💓 Heartbeat (Self-Maintenance)", "heartbeat"),
                    Choice("🧠 Memory (Storage & Retention)", "memory"),
                    Choice("📚 Skills (Capabilities & Learning)", "skills"),
                    Choice("🔍 Test Connections", "test"),
                    Choice("📄 View Full Configuration", "view"),
                    Choice("💾 Save & Exit", "save"),
                    Choice("❌ Quit Without Saving", "quit"),
                ],
                use_indicator=True,
            ).ask()

            if action == "lollms":
                self.configure_backend()  # New binding-first configuration
            elif action == "discord":
                self.configure_service("discord")
            elif action == "telegram":
                self.configure_service("telegram")
            elif action == "whatsapp":
                self.configure_service("whatsapp")
            elif action == "slack":
                self.configure_service("slack")
            elif action == "soul":
                self.configure_soul()
            elif action == "heartbeat":
                self.configure_heartbeat()
            elif action == "memory":
                self.configure_memory()
            elif action == "skills":
                self.configure_skills()
            elif action == "test":
                self.test_connections()
            elif action == "view":
                self.show_full_config()
            elif action == "save":
                self._save_all()
                console.print("\n[bold green]✅ All configurations saved![/]")
                console.print(f"[dim]Location: {self.config_path}[/]")
                break
            elif action == "quit":
                if questionary.confirm("Discard unsaved changes?", default=False).ask():
                    break
        
        console.print("\n[bold cyan]🚀 Ready to start your lollmsBot journey![/]")
        console.print("[dim]Run: lollmsbot gateway[/]")

    def _show_status_tree(self) -> None:
        """Show configuration status as a tree."""
        tree = Tree("📊 Configuration Status")
        
        # Core services
        services = tree.add("[bold]Services[/]")
        for key in ["lollms", "discord", "telegram", "whatsapp", "slack"]:
            configured = key in self.config and self.config[key]
            status = "✅" if configured else "⭕"
            color = "green" if configured else "dim"
            display_name = {
                "lollms": "AI Backend",
                "discord": "Discord",
                "telegram": "Telegram",
                "whatsapp": "WhatsApp",
                "slack": "Slack",
            }.get(key, key.title())
            services.add(f"[{color}]{status} {display_name}[/{color}]")
        
        # Show current binding if configured
        if "lollms" in self.config and "binding_name" in self.config["lollms"]:
            binding = self.config["lollms"]["binding_name"]
            services.add(f"   [dim cyan]↳ Using: {binding}[/]")
        
        # 7 Pillars
        pillars = tree.add("[bold]7 Pillars[/]")
        soul_ok = self.soul.name != "LollmsBot" or len(self.soul.traits) > 4
        pillars.add(f"{'✅' if soul_ok else '⭕'} [cyan]Soul[/] (identity)")
        pillars.add("✅ [cyan]Guardian[/] (security) - always active")
        
        hb_config = self.heartbeat.config
        pillars.add(f"{'✅' if hb_config.enabled else '⭕'} [cyan]Heartbeat[/] ({hb_config.interval_minutes}min)")
        pillars.add(f"⭕ [dim]Memory[/] (configure in Heartbeat)")
        
        # Skills
        skill_count = len(self.skill_registry._skills)
        pillars.add(f"{'✅' if skill_count > 5 else '⭕'} [cyan]Skills[/] ({skill_count} loaded)")
        
        pillars.add("⭕ [dim]Tools[/] (enabled by default)")
        pillars.add("⭕ [dim]Identity[/] (configure in Soul)")
        
        console.print(tree)
        console.print()

    def configure_backend(self) -> None:
        """Configure AI backend with binding-first selection."""
        console.print("\n[bold blue]🔗 AI Backend Configuration[/]")
        console.print("[dim]Select your LLM provider and configure connection details[/]")
        console.print()

        # Step 1: Select binding category
        console.print("[bold]Step 1: Choose binding category[/]")
        
        category = questionary.select(
            "What type of backend?",
            choices=[
                Choice("🌐 Remote / Cloud APIs (OpenAI, Claude, etc.)", "remote"),
                Choice("🏠 Local Server (Ollama, vLLM, Llama.cpp, etc.)", "local_server"),
                Choice("💻 Local Direct (Transformers, TensorRT - no server)", "local_direct"),
            ],
            use_indicator=True,
        ).ask()

        # Step 2: Select specific binding from category
        console.print(f"\n[bold]Step 2: Select {category.replace('_', ' ').title()} binding[/]")
        
        # Filter bindings by category
        category_bindings = {
            name: info for name, info in AVAILABLE_BINDINGS.items()
            if info.category == category
        }
        
        # Create choices with descriptions
        binding_choices = [
            Choice(
                f"{info.display_name} - {info.description}",
                name
            )
            for name, info in sorted(
                category_bindings.items(),
                key=lambda x: x[1].display_name
            )
        ]
        
        binding_name = questionary.select(
            "Which binding?",
            choices=binding_choices,
            use_indicator=True,
        ).ask()
        
        binding_info = AVAILABLE_BINDINGS[binding_name]
        
        # Step 3: Configure based on binding type
        console.print(f"\n[bold]Step 3: Configure {binding_info.display_name}[/]")
        
        lollms_config = self.config.setdefault("lollms", {})
        lollms_config["binding_name"] = binding_name
        
        # Common configuration
        console.print(Panel(
            f"[bold]{binding_info.display_name}[/]\n"
            f"Category: {binding_info.category.replace('_', ' ').title()}\n"
            f"Description: {binding_info.description}",
            title="Selected Binding",
            border_style="green"
        ))

        # Model name (required for all)
        default_model = binding_info.default_model or ""
        current_model = lollms_config.get("model_name", default_model)
        model_name = questionary.text(
            "Model name",
            default=current_model,
            instruction="e.g., gpt-4o-mini, llama3.2, claude-3-5-sonnet-20241022"
        ).ask()
        lollms_config["model_name"] = model_name

        # Host address (for remote and local_server)
        if binding_info.category in ("remote", "local_server"):
            default_host = binding_info.default_host or "http://localhost:8080"
            current_host = lollms_config.get("host_address", default_host)
            host_address = questionary.text(
                "Host address / API endpoint",
                default=current_host,
                instruction="Full URL including http:// or https://"
            ).ask()
            lollms_config["host_address"] = host_address

            # API key / service key
            if binding_info.requires_api_key:
                has_key = questionary.confirm(
                    "Do you have an API key / service key?",
                    default=True
                ).ask()
                
                if has_key:
                    current_key = lollms_config.get("api_key", "")
                    api_key = questionary.password(
                        "API / Service key",
                        default=current_key,
                    ).ask()
                    lollms_config["api_key"] = api_key
                else:
                    console.print("[yellow]⚠️ Most remote APIs require a key. You can add one later.[/]")
                    lollms_config["api_key"] = ""
            else:
                # Optional key (e.g., for local servers with optional auth)
                current_key = lollms_config.get("api_key", "")
                if current_key or questionary.confirm(
                    "Add optional API key? (for authenticated servers/proxies)",
                    default=bool(current_key)
                ).ask():
                    api_key = questionary.password(
                        "API / Service key (optional)",
                        default=current_key,
                    ).ask()
                    lollms_config["api_key"] = api_key

            # SSL verification (if supported)
            if binding_info.supports_ssl_verify:
                default_verify = lollms_config.get("verify_ssl", True)
                # For local servers, default to False for convenience
                if binding_info.category == "local_server" and "localhost" in host_address:
                    default_verify = lollms_config.get("verify_ssl", False)
                
                verify_ssl = questionary.confirm(
                    "Verify SSL certificates?",
                    default=default_verify
                ).ask()
                lollms_config["verify_ssl"] = verify_ssl
                
                if not verify_ssl:
                    console.print("[yellow]⚠️ SSL verification disabled. Only use for trusted local servers.[/]")
                
                # Custom certificate (advanced)
                if questionary.confirm("Use custom SSL certificate file? (advanced)", default=False).ask():
                    cert_path = questionary.text("Path to certificate file (.pem, .crt):").ask()
                    lollms_config["certificate_file_path"] = cert_path

        # Models path (for local direct bindings and some local servers)
        if binding_info.requires_models_path or (
            binding_info.category == "local_direct" and 
            questionary.confirm("Specify models folder path?", default=True).ask()
        ):
            default_path = str(Path.home() / "models")
            current_path = lollms_config.get("models_path", default_path)
            models_path = questionary.text(
                "Models folder path",
                default=current_path,
                instruction="Directory containing .gguf, .bin, or model files"
            ).ask()
            lollms_config["models_path"] = models_path
            
            # Expand user path
            models_path_expanded = os.path.expanduser(models_path)
            if not Path(models_path_expanded).exists():
                console.print(f"[yellow]⚠️ Path doesn't exist yet: {models_path_expanded}[/]")
                if questionary.confirm("Create this directory?", default=True).ask():
                    Path(models_path_expanded).mkdir(parents=True, exist_ok=True)
                    console.print("[green]✅ Directory created[/]")

        # Step 4: Optional advanced settings
        console.print("\n[bold]Step 4: Advanced settings (optional)[/]")
        
        if questionary.confirm("Configure advanced options?", default=False).ask():
            # Context size
            current_ctx = lollms_config.get("context_size", 4096)
            context_size = IntPrompt.ask(
                "Context size (tokens)",
                default=current_ctx
            )
            lollms_config["context_size"] = context_size
            
            # Temperature
            current_temp = lollms_config.get("temperature", 0.7)
            temperature = FloatPrompt.ask(
                "Default temperature (0-2)",
                default=current_temp
            )
            lollms_config["temperature"] = max(0.0, min(2.0, temperature))

        # Summary and test
        console.print("\n[bold green]✅ Backend configured![/]")
        self._configured.add("lollms")
        
        # Show configuration summary
        self._show_backend_summary(lollms_config, binding_info)
        
        # Offer to test
        if questionary.confirm("Test connection now?", default=True).ask():
            self._test_backend_connection(lollms_config)

    def _show_backend_summary(self, config: Dict[str, Any], binding_info: BindingInfo) -> None:
        """Show a summary of the backend configuration."""
        table = Table(title="Backend Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Binding", binding_info.display_name)
        table.add_row("Model", config.get("model_name", "Not set") or "Not set")
        
        if binding_info.category in ("remote", "local_server"):
            table.add_row("Host", config.get("host_address", "Not set") or "Not set")
            has_key = bool(config.get("api_key"))
            table.add_row("API Key", "✅ Set" if has_key else "⭕ Not set")
            if binding_info.supports_ssl_verify:
                table.add_row("SSL Verify", "✅ Yes" if config.get("verify_ssl", True) else "❌ No")
        
        if binding_info.requires_models_path or config.get("models_path"):
            table.add_row("Models Path", config.get("models_path", "Not set") or "Not set")
        
        table.add_row("Context Size", str(config.get("context_size", 4096)))
        table.add_row("Temperature", str(config.get("temperature", 0.7)))
        
        console.print(table)

    def _test_backend_connection(self, config: Dict[str, Any]) -> None:
        """Test the backend connection."""
        console.print("\n[bold]🧪 Testing connection...[/]")
        
        try:
            # Build settings from config
            settings = LollmsSettings(
                host_address=config.get("host_address", ""),
                api_key=config.get("api_key"),
                binding_name=config.get("binding_name"),
                model_name=config.get("model_name"),
                context_size=config.get("context_size", 4096),
                verify_ssl=config.get("verify_ssl", True),
            )
            
            # Try to build client
            client = build_lollms_client(settings)
            
            if client:
                console.print("[bold green]✅ Client initialized successfully![/]")
                console.print("[dim]Connection appears valid. Full test requires running gateway.[/]")
            else:
                console.print("[yellow]⚠️ Could not initialize client - check configuration[/]")
                
        except Exception as e:
            console.print(f"[red]❌ Connection test failed: {e}[/]")
            console.print("[dim]Tip: Ensure the backend service is running and accessible[/]")

    # Legacy method - kept for backward compatibility but not used in main flow
    def configure_service(self, service_name: str) -> None:
        """Configure a non-backend service (Discord, Telegram, WhatsApp)."""
        if service_name == "lollms":
            # Redirect to new binding-first configuration
            return self.configure_backend()
        
        # Legacy configuration for other services
        SERVICES_CONFIG: Dict[str, Dict[str, Any]] = {
            "discord": {
                "title": "🤖 Discord Bot",
                "fields": [
                    {"name": "bot_token", "prompt": "Discord Bot Token", "secret": True},
                    {"name": "allowed_users", "prompt": "Allowed User IDs (comma-separated, optional)", "optional": True},
                    {"name": "allowed_guilds", "prompt": "Allowed Server IDs (comma-separated, optional)", "optional": True},
                ],
                "setup_instructions": """🤖 Discord Setup (2 min):

1. https://discord.com/developers/applications → [+ New Application]
2. Bot → [Add Bot] → Copy **TOKEN** (MTIz... format)
3. Bot → Privileged Gateway Intents → ✅ Message Content
4. OAuth2 → URL Generator → bot scope → Invite to server""",
            },
            "telegram": {
                "title": "✈️ Telegram Bot",
                "fields": [
                    {"name": "bot_token", "prompt": "Telegram Bot Token (from @BotFather)", "secret": True},
                    {"name": "allowed_users", "prompt": "Allowed User IDs (comma-separated, optional)", "optional": True},
                ],
                "setup_instructions": """✈️ Telegram Setup (1 min):

1. Message @BotFather on Telegram
2. Send /newbot and follow instructions
3. Copy the HTTP API token provided""",
            },
            "whatsapp": {
                "title": "💬 WhatsApp Integration",
                "fields": [],  # Custom handling below
                "setup_instructions": "",  # Custom handling below
            },
            "slack": {
                "title": "💬 Slack Bot",
                "fields": [],  # Custom handling below
                "setup_instructions": "",  # Custom handling below
            },
        }
        
        # Special handling for WhatsApp and Slack
        if service_name == "whatsapp":
            self._configure_whatsapp()
            return
        elif service_name == "slack":
            self._configure_slack()
            return
        
        service = SERVICES_CONFIG.get(service_name)
        if not service:
            console.print(f"[red]Unknown service: {service_name}[/]")
            return
            
        console.print(f"\n[bold yellow]{service['title']}[/]")
        
        if "setup_instructions" in service:
            console.print(Panel(service["setup_instructions"], title="📋 Instructions"))

        service_config = self.config.setdefault(service_name, {})

        for field in service["fields"]:
            current = service_config.get(field["name"])
            default = str(current) if current is not None else field.get("default", "")

            if field.get("type") == "bool":
                value = questionary.confirm(field["prompt"], default=field.get("default", False)).ask()
            elif field.get("secret"):
                value = questionary.password(field["prompt"], default=default).ask()
            else:
                value = questionary.text(field["prompt"], default=default).ask()

            # Parse comma-separated lists
            if "users" in field["name"] or "guilds" in field["name"]:
                if value:
                    value = [v.strip() for v in value.split(",") if v.strip()]
                else:
                    value = []
            elif field.get("optional") and not value:
                continue

            service_config[field["name"]] = value

        self._configured.add(service_name)
        console.print("[green]✅ Updated![/]")

    def configure_soul(self) -> None:
        """Interactive Soul (personality) configuration."""
        console.print("\n[bold magenta]🧬 Soul Configuration[/]")
        
        while True:
            section = questionary.select(
                "Configure aspect:",
                choices=[
                    "🎭 Core Identity (name, purpose, origin)",
                    "🌈 Personality Traits",
                    "⚖️ Core Values",
                    "💬 Communication Style",
                    "🎓 Expertise Domains",
                    "👥 Relationship Stances",
                    "🔍 Preview System Prompt",
                    "💾 Save & Return",
                ]
            ).ask()

            if section == "🎭 Core Identity (name, purpose, origin)":
                self._configure_core_identity()
            elif section == "🌈 Personality Traits":
                self._configure_personality_traits()
            elif section == "⚖️ Core Values":
                self._configure_values()
            elif section == "💬 Communication Style":
                self._configure_communication()
            elif section == "🎓 Expertise Domains":
                self._configure_expertise()
            elif section == "👥 Relationship Stances":
                self._configure_relationships()
            elif section == "🔍 Preview System Prompt":
                self._preview_soul_prompt()
            else:
                self.soul._save()
                self._configured.add("soul")
                console.print("[green]✅ Soul saved![/]")
                break

    def _configure_core_identity(self) -> None:
        """Configure name, purpose, and origin story."""
        console.print("\n[bold]Core Identity[/]")
        
        self.soul.name = questionary.text("AI Name", default=self.soul.name).ask()
        self.soul.purpose = questionary.text("Primary Purpose", default=self.soul.purpose).ask()
        self.soul.origin_story = questionary.text("Origin Story", default=self.soul.origin_story).ask()

    def _configure_personality_traits(self) -> None:
        """Add, edit, or remove personality traits."""
        console.print("\n[bold]Personality Traits[/]")
        
        while True:
            table = Table(title="Current Traits")
            table.add_column("Trait")
            table.add_column("Intensity")
            table.add_column("Description")
            
            for trait in self.soul.traits:
                intensity_emoji = {
                    TraitIntensity.SUBTLE: "◐",
                    TraitIntensity.MODERATE: "◑",
                    TraitIntensity.STRONG: "◕",
                    TraitIntensity.EXTREME: "⬤",
                }.get(trait.intensity, "◑")
                table.add_row(trait.name, f"{intensity_emoji} {trait.intensity.name.lower()}", trait.description[:40])
            
            console.print(table)
            
            action = questionary.select(
                "Action:",
                choices=["➕ Add Trait", "✏️ Edit Trait", "🗑️ Remove Trait", "🔙 Back"]
            ).ask()
            
            if action == "➕ Add Trait":
                name = questionary.text("Trait name (e.g., 'curiosity', 'pragmatism')").ask()
                description = questionary.text("How does this manifest?").ask()
                intensity = questionary.select(
                    "Intensity",
                    choices=["subtle", "moderate", "strong", "extreme"],
                    default="moderate"
                ).ask()
                
                trait = PersonalityTrait(
                    name=name,
                    description=description,
                    intensity=TraitIntensity[intensity.upper()],
                )
                self.soul.traits.append(trait)
                
            elif action == "✏️ Edit Trait" and self.soul.traits:
                trait_names = [t.name for t in self.soul.traits]
                to_edit = questionary.select("Edit which trait?", choices=trait_names).ask()
                trait = next(t for t in self.soul.traits if t.name == to_edit)
                
                trait.description = questionary.text("Description", default=trait.description).ask()
                new_intensity = questionary.select(
                    "Intensity",
                    choices=["subtle", "moderate", "strong", "extreme"],
                    default=trait.intensity.name.lower()
                ).ask()
                trait.intensity = TraitIntensity[new_intensity.upper()]
                
            elif action == "🗑️ Remove Trait" and self.soul.traits:
                to_remove = questionary.select(
                    "Remove which trait?",
                    choices=[t.name for t in self.soul.traits]
                ).ask()
                self.soul.traits = [t for t in self.soul.traits if t.name != to_remove]
            else:
                break

    def _configure_values(self) -> None:
        """Configure core ethical values."""
        console.print("\n[bold]Core Values[/]")
        
        while True:
            table = Table(title="Current Values (by priority)")
            table.add_column("Priority")
            table.add_column("Value")
            table.add_column("Category")
            
            for v in sorted(self.soul.values, key=lambda x: -x.priority):
                priority_color = "red" if v.priority >= 9 else "yellow" if v.priority >= 7 else "green"
                table.add_row(f"[{priority_color}]{v.priority}[/{priority_color}]", v.statement[:50], v.category)
            
            console.print(table)
            
            action = questionary.select(
                "Action:",
                choices=["➕ Add Value", "✏️ Edit Priority", "🗑️ Remove Value", "🔙 Back"]
            ).ask()
            
            if action == "➕ Add Value":
                statement = questionary.text("Value statement").ask()
                category = questionary.text("Category", default="general").ask()
                priority = IntPrompt.ask("Priority (1-10)", default=5)
                self.soul.values.append(ValueStatement(statement, category, max(1, min(10, priority))))
                
            elif action == "✏️ Edit Priority" and self.soul.values:
                statements = [v.statement[:40] + "..." for v in self.soul.values]
                to_edit = questionary.select("Edit which value?", choices=statements).ask()
                val = next(v for v in self.soul.values if v.statement.startswith(to_edit[:20]))
                val.priority = IntPrompt.ask("New priority (1-10)", default=val.priority)
                
            elif action == "🗑️ Remove Value" and self.soul.values:
                to_remove = questionary.select(
                    "Remove which value?",
                    choices=[v.statement[:40] for v in self.soul.values]
                ).ask()
                self.soul.values = [v for v in self.soul.values if not v.statement.startswith(to_remove[:20])]
            else:
                break

    def _configure_communication(self) -> None:
        """Configure communication style."""
        style = self.soul.communication
        
        style.formality = questionary.select(
            "Formality",
            choices=["formal", "casual", "technical", "playful"],
            default=style.formality
        ).ask()
        
        style.verbosity = questionary.select(
            "Default verbosity",
            choices=["terse", "concise", "detailed", "exhaustive"],
            default=style.verbosity
        ).ask()
        
        humor = questionary.select(
            "Humor style",
            choices=["None (serious)", "witty", "dry", "punny", "absurdist"],
            default=style.humor_style or "None (serious)"
        ).ask()
        style.humor_style = None if humor == "None (serious)" else humor
        
        style.emoji_usage = questionary.select(
            "Emoji usage",
            choices=["none", "minimal", "moderate", "liberal"],
            default=style.emoji_usage
        ).ask()

    def _configure_expertise(self) -> None:
        """Configure knowledge domains."""
        console.print("\n[bold]Expertise Domains[/]")
        
        while True:
            table = Table(title="Current Expertise")
            table.add_column("Domain")
            table.add_column("Level")
            table.add_column("Specialties")
            
            for e in self.soul.expertise:
                level_color = {
                    "novice": "red", "competent": "yellow", "expert": "green",
                    "authority": "blue", "pioneer": "magenta",
                }.get(e.level, "white")
                table.add_row(e.domain, f"[{level_color}]{e.level}[/{level_color}]", ", ".join(e.specialties[:2]))
            
            console.print(table)
            
            action = questionary.select(
                "Action:",
                choices=["➕ Add Domain", "🔙 Back"]
            ).ask()
            
            if action == "➕ Add Domain":
                domain = questionary.text("Domain name").ask()
                level = questionary.select(
                    "Competence level",
                    choices=["novice", "competent", "expert", "authority", "pioneer"],
                    default="competent"
                ).ask()
                specialties = [s.strip() for s in questionary.text("Specialties (comma-separated)").ask().split(",") if s.strip()]
                
                self.soul.expertise.append(ExpertiseDomain(domain=domain, level=level, specialties=specialties))
            else:
                break

    def _configure_relationships(self) -> None:
        """Configure relationship stances."""
        console.print("\n[bold]Relationship Stances[/]")
        console.print("[dim]Simplified configuration - full implementation in soul.md[/]")

    def _preview_soul_prompt(self) -> None:
        """Preview the generated system prompt."""
        prompt = self.soul.generate_system_prompt()
        preview = prompt[:1000] + ("..." if len(prompt) > 1000 else "")
        console.print(Panel(preview, title="System Prompt", border_style="cyan"))

    def configure_heartbeat(self) -> None:
        """Configure self-maintenance heartbeat."""
        console.print("\n[bold magenta]💓 Heartbeat Configuration[/]")
        
        config = self.heartbeat.config
        
        config.enabled = questionary.confirm("Enable automatic self-maintenance?", default=config.enabled).ask()
        if not config.enabled:
            self.heartbeat._save_config()
            return
        
        config.interval_minutes = FloatPrompt.ask("Maintenance interval (minutes)", default=config.interval_minutes)
        
        console.print("\n[bold]Maintenance Tasks[/]")
        for task in MaintenanceTask:
            task_name = task.name.replace("_", " ").title()
            config.tasks_enabled[task] = questionary.confirm(
                f"Enable {task_name}?", default=config.tasks_enabled.get(task, True)
            ).ask()
        
        console.print("\n[bold]Self-Healing Behavior[/]")
        config.auto_heal_minor = questionary.confirm("Auto-fix minor issues?", default=config.auto_heal_minor).ask()
        config.confirm_heal_major = questionary.confirm("Confirm before major changes?", default=config.confirm_heal_major).ask()
        
        self.heartbeat.update_config(**{
            k: getattr(config, k) for k in [
                "enabled", "interval_minutes", "tasks_enabled",
                "auto_heal_minor", "confirm_heal_major"
            ]
        })
        
        self.config["heartbeat"] = {
            "enabled": config.enabled,
            "interval_minutes": config.interval_minutes,
            "tasks_enabled": [t.name for t, v in config.tasks_enabled.items() if v],
        }
        self._configured.add("heartbeat")
        console.print("[green]✅ Heartbeat configured![/]")

    def configure_memory(self) -> None:
        """Configure memory and retention settings."""
        console.print("\n[bold magenta]🧠 Memory Configuration[/]")
        
        hb_config = self.heartbeat.config
        
        hb_config.memory_pressure_threshold = FloatPrompt.ask(
            "Memory pressure threshold (0-1)", default=hb_config.memory_pressure_threshold
        )
        hb_config.log_retention_days = IntPrompt.ask(
            "Audit log retention (days)", default=hb_config.log_retention_days
        )
        
        console.print("\n[bold]Forgetting Curve Parameters[/]")
        halflife = FloatPrompt.ask("Memory half-life (days)", default=7.0)
        strength_mult = FloatPrompt.ask("Review strength multiplier", default=2.0)
        
        self.heartbeat.memory_monitor.retention_halflife_days = halflife
        self.heartbeat.memory_monitor.strength_multiplier = strength_mult
        self.heartbeat._save_config()
        
        self.config["memory"] = {
            "pressure_threshold": hb_config.memory_pressure_threshold,
            "log_retention_days": hb_config.log_retention_days,
            "retention_halflife_days": halflife,
            "strength_multiplier": strength_mult,
        }
        self._configured.add("memory")
        console.print("[green]✅ Memory configured![/]")

    def configure_skills(self) -> None:
        """Configure Skills - browse, test, and manage capabilities."""
        console.print("\n[bold magenta]📚 Skills Configuration[/]")
        console.print("[dim]Browse, test, and configure LollmsBot's capabilities[/]")
        
        while True:
            # Show skill statistics
            stats = self._get_skill_stats()
            
            table = Table(title=f"Skills Library ({stats['total']} total)")
            table.add_column("Category")
            table.add_column("Built-in")
            table.add_column("User-created")
            table.add_column("Avg Confidence")
            
            for cat, data in sorted(stats['by_category'].items()):
                table.add_row(
                    cat,
                    str(data['builtin']),
                    str(data['user']),
                    f"{data['avg_confidence']:.0%}"
                )
            
            console.print(table)
            
            action = questionary.select(
                "Skills action:",
                choices=[
                    "🔍 Browse & Search Skills",
                    "📖 View Skill Details",
                    "🧪 Test Skill Execution",
                    "➕ Compose New Skill (from existing)",
                    "📤 Export Skill Library",
                    "📥 Import Skills",
                    "⚙️ Skill Preferences",
                    "🔙 Back to Main Menu",
                ]
            ).ask()
            
            if action == "🔍 Browse & Search Skills":
                self._browse_skills()
            elif action == "📖 View Skill Details":
                self._view_skill_details()
            elif action == "🧪 Test Skill Execution":
                self._test_skill()
            elif action == "➕ Compose New Skill (from existing)":
                self._compose_skill()
            elif action == "📤 Export Skill Library":
                self._export_skills()
            elif action == "📥 Import Skills":
                self._import_skills()
            elif action == "⚙️ Skill Preferences":
                self._skill_preferences()
            else:
                self._configured.add("skills")
                break
    
    def _get_skill_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded skills."""
        skills = list(self.skill_registry._skills.values())
        
        by_category: Dict[str, Dict[str, Any]] = {}
        for skill in skills:
            for cat in skill.metadata.categories or ["uncategorized"]:
                if cat not in by_category:
                    by_category[cat] = {'builtin': 0, 'user': 0, 'confidence_sum': 0, 'count': 0}
                # Simplified: would track builtin vs user properly
                by_category[cat]['count'] += 1
                by_category[cat]['confidence_sum'] += skill.metadata.confidence_score
        
        # Calculate averages
        for cat in by_category:
            data = by_category[cat]
            data['avg_confidence'] = data['confidence_sum'] / data['count'] if data['count'] > 0 else 0
        
        return {
            'total': len(skills),
            'by_category': by_category,
            'by_complexity': {
                c.name: len(self.skill_registry.list_skills(complexity=c))
                for c in SkillComplexity
            },
        }
    
    def _browse_skills(self) -> None:
        """Browse and search skills interactively."""
        search = questionary.text("Search skills (empty for all):").ask()
        
        if search:
            results = self.skill_registry.search(search)
            skills = [s for s, _ in results]
        else:
            category = questionary.select(
                "Filter by category:",
                choices=["All"] + list(self.skill_registry._categories.keys())
            ).ask()
            if category == "All":
                skills = list(self.skill_registry._skills.values())
            else:
                skills = self.skill_registry.list_skills(category=category)
        
        # Display results
        table = Table(title=f"Skills ({len(skills)} found)")
        table.add_column("Name")
        table.add_column("Complexity")
        table.add_column("Description")
        table.add_column("Confidence")
        
        for skill in skills[:20]:  # Limit display
            conf_color = "green" if skill.metadata.confidence_score > 0.8 else "yellow" if skill.metadata.confidence_score > 0.5 else "red"
            table.add_row(
                skill.name,
                skill.metadata.complexity.name,
                skill.metadata.description[:40],
                f"[{conf_color}]{skill.metadata.confidence_score:.0%}[/{conf_color}]"
            )
        
        console.print(table)
    
    def _view_skill_details(self) -> None:
        """View detailed information about a specific skill."""
        skill_name = questionary.select(
            "Select skill:",
            choices=list(self.skill_registry._skills.keys())
        ).ask()
        
        skill = self.skill_registry.get(skill_name)
        if not skill:
            console.print("[red]Skill not found[/]")
            return
        
        md = skill.metadata
        
        details = f"""
[bold]{md.name}[/] v{md.version}
[dim]{md.description}[/]

[bold]Complexity:[/] {md.complexity.name}
[bold]Categories:[/] {', '.join(md.categories)}
[bold]Tags:[/] {', '.join(md.tags)}

[bold]When to use:[/] {md.when_to_use or 'N/A'}
[bold]When NOT to use:[/] {md.when_not_to_use or 'N/A'}

[bold]Parameters:[/]
{chr(10).join(f"  • {p.name} ({p.type}){' [required]' if p.required else ''}: {p.description}" for p in md.parameters)}

[bold]Dependencies:[/]
{chr(10).join(f"  • {d.kind}:{d.name}{' (optional)' if d.optional else ''}" for d in md.dependencies)}

[bold]Statistics:[/]
  • Executed: {md.execution_count} times
  • Success rate: {md.success_rate:.1%}
  • Confidence score: {md.confidence_score:.0%}
"""
        console.print(Panel(details, title=f"Skill: {md.name}", border_style="blue"))
        
        # Show examples if any
        if md.examples:
            console.print("\n[bold]Examples:[/]")
            for i, ex in enumerate(md.examples[:2], 1):
                console.print(Panel(
                    f"Input: {json.dumps(ex.input_params, indent=2)}\n"
                    f"Output: {json.dumps(ex.expected_output, indent=2)}",
                    title=f"Example {i}"
                ))
    
    def _test_skill(self) -> None:
        """Test execute a skill with sample inputs."""
        console.print("[yellow]Note: Full execution requires running agent. Showing validation only.[/]")
        
        skill_name = questionary.select(
            "Select skill to test:",
            choices=list(self.skill_registry._skills.keys())
        ).ask()
        
        skill = self.skill_registry.get(skill_name)
        
        # Gather inputs
        inputs = {}
        for param in skill.metadata.parameters:
            if not param.required:
                if not questionary.confirm(f"Provide optional parameter '{param.name}'?", default=False).ask():
                    continue
            
            value = questionary.text(f"{param.name} ({param.type}): {param.description}").ask()
            
            # Simple type coercion
            if param.type == "number":
                value = float(value) if '.' in value else int(value)
            elif param.type == "boolean":
                value = value.lower() in ('true', 'yes', '1', 'on')
            elif param.type == "array":
                value = [v.strip() for v in value.split(',')]
            elif param.type == "object":
                try:
                    value = json.loads(value)
                except:
                    value = {"raw": value}
            
            inputs[param.name] = value
        
        # Validate
        valid, errors = skill.validate_inputs(inputs)
        if valid:
            console.print("[green]✅ Inputs valid![/]")
            
            # Check dependencies
            # Would need actual agent/tools to check properly
            console.print("[dim]Dependency check: would validate against available tools[/]")
        else:
            console.print("[red]❌ Validation failed:[/]")
            for err in errors:
                console.print(f"  • {err}")
    
    def _compose_skill(self) -> None:
        """Create new skill by composing existing skills."""
        console.print("\n[bold]Compose New Skill[/]")
        console.print("[dim]Combine existing skills into a workflow[/]")
        
        name = questionary.text("Name for new skill:").ask()
        description = questionary.text("What does this skill do?").ask()
        
        # Select component skills
        available = list(self.skill_registry._skills.keys())
        components = []
        
        while True:
            remaining = [s for s in available if s not in components]
            if not remaining:
                break
            
            choice = questionary.select(
                "Add component skill (or Done):",
                choices=["Done"] + remaining
            ).ask()
            
            if choice == "Done":
                break
            
            components.append(choice)
            console.print(f"[green]Added: {choice}[/]")
        
        if len(components) < 1:
            console.print("[yellow]Need at least one component[/]")
            return
        
        # Define data flow (simplified)
        console.print("\n[dim]Data flow would be configured here - mapping outputs to inputs[/]")
        
        # Preview and confirm
        console.print(Panel(
            f"Name: {name}\n"
            f"Description: {description}\n"
            f"Components: {' → '.join(components)}",
            title="New Skill Preview"
        ))
        
        if questionary.confirm("Create this skill?", default=True).ask():
            # Would call skill learner
            console.print("[green]✅ Skill composition recorded (implementation in code)[/]")
    
    def _export_skills(self) -> None:
        """Export skills to file."""
        export_path = Path.home() / ".lollmsbot" / "skills_export.json"
        
        data = {
            "export_date": datetime.now().isoformat(),
            "skills": [skill.to_dict() for skill in self.skill_registry._skills.values()],
        }
        
        export_path.write_text(json.dumps(data, indent=2))
        console.print(f"[green]✅ Exported {len(data['skills'])} skills to {export_path}[/]")
    
    def _import_skills(self) -> None:
        """Import skills from file."""
        import_path = questionary.text("Path to skills file:").ask()
        path = Path(import_path)
        
        if not path.exists():
            console.print("[red]File not found[/]")
            return
        
        try:
            data = json.loads(path.read_text())
            count = len(data.get("skills", []))
            console.print(f"[green]✅ Found {count} skills to import[/]")
            console.print("[dim]Import would validate and register skills here[/]")
        except Exception as e:
            console.print(f"[red]Import failed: {e}[/]")
    
    def _skill_preferences(self) -> None:
        """Configure skill execution preferences."""
        console.print("\n[bold]Skill Preferences[/]")
        
        # Would configure: auto-skill vs manual, confidence thresholds, etc.
        prefs = {
            "auto_skill_selection": questionary.confirm("Allow automatic skill selection?", default=True).ask(),
            "min_confidence_threshold": FloatPrompt.ask("Minimum skill confidence (0-1)", default=0.6),
            "confirm_complex_skills": questionary.confirm("Confirm before complex skill execution?", default=True).ask(),
        }
        
        self.config["skill_preferences"] = prefs
        console.print("[green]✅ Preferences saved[/]")

    def test_connections(self) -> None:
        """Test all configured connections."""
        table = Table(title="🧪 Connection Tests")
        table.add_column("Service")
        table.add_column("Status")
        table.add_column("Details")

        # Test backend first
        lollms_config = self.config.get("lollms", {})
        if lollms_config:
            status, details = self._test_single("lollms", lollms_config)
            # Show model and binding info
            binding_name = lollms_config.get("binding_name", "unknown")
            details = f"{binding_name}: {details}"
            table.add_row("AI Backend", status, details)
        
        # Test other services
        for service_name, svc_config in self.config.items():
            if service_name in ["lollms", "heartbeat", "memory", "soul", "skill_preferences"]:
                continue
                
            status, details = self._test_single(service_name, svc_config)
            display_name = "Discord" if service_name == "discord" else "Telegram" if service_name == "telegram" else service_name.title()
            table.add_row(display_name, status, details)

        # Test Soul
        soul_hash = hashlib.sha256(
            json.dumps(self.soul.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]
        table.add_row("Soul", "✅ VALID", f"Hash: {soul_hash}")

        # Test Heartbeat
        hb_status = self.heartbeat.get_status()
        table.add_row(
            "Heartbeat", 
            "✅ ACTIVE" if hb_status["running"] else "⭕ STOPPED",
            f"Interval: {hb_status['interval_minutes']}min"
        )

        # Test Skills
        skill_stats = self._get_skill_stats()
        table.add_row(
            "Skills",
            "✅ LOADED",
            f"{skill_stats['total']} skills, {len(skill_stats['by_category'])} categories"
        )

        console.print(table)

    def _configure_whatsapp(self) -> None:
        """Configure WhatsApp integration with backend selection."""
        console.print("\n[bold green]💬 WhatsApp Configuration[/]")
        
        # Educational explanation first
        console.print(Panel(
            """[bold]How WhatsApp Integration Works[/bold]

LollmsBot connects to WhatsApp using [cyan]whatsapp-web.js[/cyan], which is a 
Node.js library that controls a real web browser (Chromium) and loads 
web.whatsapp.com just like you would in Chrome.

Here's what happens:

[bold]1. Bridge Launch[/bold]
   • Node.js starts a headless browser (no visible window)
   • Browser navigates to web.whatsapp.com
   • WhatsApp generates a unique QR code for pairing

[bold]2. QR Code Display[/bold]
   • The QR code appears as ASCII art in this terminal
   • It's a big square made of █ and ░ characters
   • You scan it with your phone's WhatsApp camera

[bold]3. Session Pairing[/bold]
   • Your phone links to the browser session
   • WhatsApp Web session is established
   • Session stays active as long as your phone is online

[bold]4. Message Handling[/bold]
   • Incoming messages trigger the bot's AI response
   • Responses are sent back through the browser session
   • Everything happens in real-time

[bold]Important:[/] Your phone must stay connected to the internet.
If your phone loses connection, the bot stops working.""",
            title="📚 WhatsApp Web Architecture",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        # Select backend
        backend = questionary.select(
            "Choose WhatsApp backend:",
            choices=[
                Choice("📱 whatsapp-web.js (free, local, requires Node.js)", "web_js"),
                Choice("☁️ Twilio API (cloud, requires account)", "twilio"),
                Choice("🏢 WhatsApp Business API (official, webhook-based)", "business_api"),
            ],
            use_indicator=True,
        ).ask()
        
        wa_config = self.config.setdefault("whatsapp", {})
        wa_config["backend"] = backend
        
        console.print(f"\n[bold]Configuring {backend} backend...[/]")
        
        if backend == "web_js":
            console.print(Panel(
                """[bold]📱 whatsapp-web.js Setup Guide[/bold]

[cyan]Prerequisites:[/]
• Node.js 16+ installed (download from https://nodejs.org/)
• WhatsApp installed on your phone (iOS or Android)
• Your phone must stay connected to the internet

[cyan]What will happen:[/]

[bold]Step 1: Dependency Installation[/bold]
   The wizard will automatically:
   • Create a bridge script in ~/.lollmsbot/whatsapp-bridge/
   • Run 'npm install' to download required packages
   • This takes 1-2 minutes on first run

[bold]Step 2: QR Code Generation[/bold]
   When you start lollmsbot:
   • A large ASCII QR code appears in the terminal
   • It looks like a black/white square made of block characters
   • You'll see messages: "BRIDGE_STARTING" → "CLIENT_INITIALIZED" → QR code

[bold]Step 3: Scanning the QR Code[/bold]
   [yellow]Important:[/] The QR code is ASCII art in the terminal, not an image!

   1. Open WhatsApp on your phone
   2. Go to: [bold]Settings → Linked Devices → Link a Device[/bold]
   3. Point your phone's camera at the terminal screen
   4. The QR code is the big square of █ and ░ characters
   5. Hold steady until you see "Authenticated!" message

[bold]Step 4: Confirmation[/bold]
   • You'll see "✅ AUTHENTICATED!" in the terminal
   • The bot is now connected to your WhatsApp
   • You can send messages to the bot's number

[cyan]Troubleshooting:[/]
• [bold]QR code not visible?[/] Make terminal window wider (100+ chars)
• [bold]Scan fails?[/] Try refreshing terminal or restarting lollmsbot
• [bold]Connection drops?[/] Check your phone's internet connection
• [bold]Session expires?[/] You'll need to scan QR code again

[cyan]Security Note:[/]
Your WhatsApp session is stored locally on this computer in:
~/.lollmsbot/whatsapp-bridge/.wwebjs_auth/
Only this lollmsbot instance can access your WhatsApp.""",
                title="Complete Setup Instructions",
                border_style="blue",
                padding=(1, 2)
            ))
            
            # Path configuration
            default_path = str(Path.home() / ".lollmsbot" / "whatsapp-bridge")
            current_path = wa_config.get("web_js_path", default_path)
            web_js_path = questionary.text(
                "Bridge script directory",
                default=current_path
            ).ask()
            wa_config["web_js_path"] = web_js_path
            
            # Show what will happen on first run
            console.print(Panel(
                """[bold]Next Steps After Configuration:[/]

When you run [cyan]lollmsbot gateway[/cyan] for the first time:

1. You'll see: "🔍 Step 1: Checking Node.js availability..."
2. Then: "📦 Installing whatsapp-web.js dependencies..."
3. Then: "🚀 Starting Node.js bridge process..."
4. [yellow]A large QR code appears as ASCII art[/yellow]
5. Scan it with your phone (Settings → Linked Devices)
6. See "✅ AUTHENTICATED!" when successful
7. Bot is ready to receive WhatsApp messages!

[bold]Tip:[/] The QR code looks like this:
█████████████████████████████████
██ ▄▄▄▄▄ █▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀ ██
██ █   █ █▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀ ██
██ █▄▄▄█ █▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀▄▀ ██
█████████████████████████████████

(But much bigger - fill your terminal width!)""",
                title="What to Expect on First Run",
                border_style="green"
            ))
            
            # User restrictions
            self._configure_whatsapp_users(wa_config)
            
            # Confirmation requirement
            wa_config["require_confirmation"] = questionary.confirm(
                "Require users to confirm before chatting? (sends welcome message)",
                default=wa_config.get("require_confirmation", True)
            ).ask()
            
        elif backend == "twilio":
            console.print(Panel(
                """☁️ Twilio WhatsApp Setup:

1. Create a Twilio account at https://www.twilio.com/
2. Get a WhatsApp-enabled number from Twilio Console
3. Find your Account SID and Auth Token in Console Dashboard
4. Configure webhook URL in Twilio Console (after starting gateway)

[bold]Note:[/] Twilio has usage costs but is reliable and cloud-based""",
                title="Setup Instructions",
                border_style="blue"
            ))
            
            # Credentials
            current_sid = wa_config.get("account_sid", "")
            account_sid = questionary.text("Twilio Account SID", default=current_sid).ask()
            wa_config["account_sid"] = account_sid
            
            current_token = wa_config.get("auth_token", "")
            auth_token = questionary.password("Twilio Auth Token", default=current_token).ask()
            wa_config["auth_token"] = auth_token
            
            # WhatsApp number
            current_from = wa_config.get("from_number", "")
            from_number = questionary.text(
                "Your Twilio WhatsApp Number (with +country code)",
                default=current_from,
                instruction="e.g., +14155238886"
            ).ask()
            wa_config["from_number"] = from_number
            
            # Webhook port
            current_port = wa_config.get("webhook_port", 8081)
            webhook_port = IntPrompt.ask("Webhook port for incoming messages", default=current_port)
            wa_config["webhook_port"] = webhook_port
            
            self._configure_whatsapp_users(wa_config)
            
        elif backend == "business_api":
            console.print(Panel(
                """🏢 WhatsApp Business API Setup:

1. Create a Meta Developer account at https://developers.facebook.com/
2. Set up a WhatsApp Business account and get API token
3. Configure webhook endpoint in Meta Dashboard
4. Verify webhook with secret token

[bold]Note:[/] This is the official API, best for production use""",
                title="Setup Instructions",
                border_style="blue"
            ))
            
            # API credentials
            current_token = wa_config.get("api_token", "")
            api_token = questionary.password("WhatsApp Business API Token", default=current_token).ask()
            wa_config["api_token"] = api_token
            
            # Phone number ID
            current_phone_id = wa_config.get("from_number", "")
            phone_id = questionary.text(
                "WhatsApp Business Phone Number ID",
                default=current_phone_id
            ).ask()
            wa_config["from_number"] = phone_id
            
            # Webhook configuration
            current_secret = wa_config.get("webhook_secret", "")
            webhook_secret = questionary.password(
                "Webhook Verify Token (for Meta verification)",
                default=current_secret
            ).ask()
            wa_config["webhook_secret"] = webhook_secret
            
            current_port = wa_config.get("webhook_port", 8081)
            webhook_port = IntPrompt.ask("Webhook port", default=current_port)
            wa_config["webhook_port"] = webhook_port
            
            self._configure_whatsapp_users(wa_config)
        
        self._configured.add("whatsapp")
        console.print("[green]✅ WhatsApp configured![/]")
        
        # Show summary with scanning instructions reminder
        self._show_whatsapp_summary(wa_config)
        
        # Final reminder about QR code scanning
        if backend == "web_js":
            console.print(Panel(
                """[bold yellow]📱 IMPORTANT: How to Scan the QR Code[/bold yellow]

When you start lollmsbot, the QR code will appear as [bold]ASCII art[/bold] 
in your terminal - a big square made of block characters (█ and ░).

[yellow]Scanning Instructions:[/]
1. Open WhatsApp on your phone
2. Tap: [bold]Settings (⚙️) → Linked Devices → Link a Device[/bold]
3. Point camera at the terminal screen
4. The QR code is the large black/white square of characters
5. Hold steady until you see "Authenticated!"

[bold]Testing the Bot:[/]
⚠️  [yellow]You CANNOT message yourself![/yellow] WhatsApp doesn't allow it.
   Instead:
   • Ask a friend to message your WhatsApp number
   • Create a group chat and add the bot (your number)
   • Use a second phone/SIM with different WhatsApp

[bold]Common Mistakes:[/]
❌ Don't look for an image file - it's text in the terminal
❌ Don't use a QR scanner app - use WhatsApp's built-in scanner
❌ Don't try to message yourself - it won't work!
✅ Do make terminal window very wide (drag to expand)
✅ Do scroll up if QR code is above current view

[yellow]The QR code will only appear once per session.[/yellow]""",
                title="QR Code Scanning Guide",
                border_style="yellow"
            ))
    
    def _configure_whatsapp_users(self, wa_config: Dict[str, Any]) -> None:
        """Configure allowed/blocked users for WhatsApp."""
        # Allowed numbers
        current_allowed = wa_config.get("allowed_numbers", [])
        allowed_str = ",".join(current_allowed) if current_allowed else ""
        
        allowed_input = questionary.text(
            "Allowed phone numbers (comma-separated, optional - empty = allow all)",
            default=allowed_str,
            instruction="Format: +1234567890, +441234567890"
        ).ask()
        
        if allowed_input.strip():
            wa_config["allowed_numbers"] = [
                n.strip() for n in allowed_input.split(",") if n.strip()
            ]
        else:
            wa_config["allowed_numbers"] = []
        
        # Blocked numbers
        current_blocked = wa_config.get("blocked_numbers", [])
        blocked_str = ",".join(current_blocked) if current_blocked else ""
        
        blocked_input = questionary.text(
            "Blocked phone numbers (comma-separated, optional)",
            default=blocked_str
        ).ask()
        
        if blocked_input.strip():
            wa_config["blocked_numbers"] = [
                n.strip() for n in blocked_input.split(",") if n.strip()
            ]
        else:
            wa_config["blocked_numbers"] = []
    
    def _show_whatsapp_summary(self, wa_config: Dict[str, Any]) -> None:
        """Show WhatsApp configuration summary."""
        table = Table(title="WhatsApp Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        backend = wa_config.get("backend", "unknown")
        backend_display = {
            "web_js": "📱 whatsapp-web.js (local)",
            "twilio": "☁️ Twilio (cloud)",
            "business_api": "🏢 Business API (official)",
        }.get(backend, backend)
        
        table.add_row("Backend", backend_display)
        
        if backend == "web_js":
            table.add_row("Bridge Path", wa_config.get("web_js_path", "Not set"))
        elif backend == "twilio":
            has_sid = bool(wa_config.get("account_sid"))
            has_token = bool(wa_config.get("auth_token"))
            table.add_row("Account SID", "✅ Set" if has_sid else "❌ Not set")
            table.add_row("Auth Token", "✅ Set" if has_token else "❌ Not set")
            table.add_row("From Number", wa_config.get("from_number", "Not set") or "Not set")
            table.add_row("Webhook Port", str(wa_config.get("webhook_port", 8081)))
        elif backend == "business_api":
            has_token = bool(wa_config.get("api_token"))
            table.add_row("API Token", "✅ Set" if has_token else "❌ Not set")
            table.add_row("Phone Number ID", wa_config.get("from_number", "Not set") or "Not set")
            table.add_row("Webhook Port", str(wa_config.get("webhook_port", 8081)))
        
        allowed = wa_config.get("allowed_numbers", [])
        blocked = wa_config.get("blocked_numbers", [])
        table.add_row("Allowed Numbers", str(len(allowed)) if allowed else "All")
        table.add_row("Blocked Numbers", str(len(blocked)))
        table.add_row("Require Confirmation", "Yes" if wa_config.get("require_confirmation") else "No")
        
        console.print(table)
    
    def _configure_slack(self) -> None:
        """Configure Slack integration with mode selection."""
        console.print("\n[bold green]💬 Slack Configuration[/]")
        
        # Educational explanation first
        console.print(Panel(
            """[bold]How Slack Integration Works[/bold]

LollmsBot connects to Slack using the [cyan]Slack Bolt SDK[/cyan], which is the 
official modern Python SDK for building Slack apps.

Two connection modes are available:

[bold]🔌 Socket Mode (Recommended)[/bold]
   • WebSocket-based connection, no public URL needed
   • Works behind firewalls and NAT
   • Best for local development and private networks
   • Requires App-Level Token (starts with xapp-)

[bold]🌐 HTTP Mode[/bold]
   • Webhook-based, requires public HTTPS URL
   • Traditional request/response model
   • Best for production deployments with public endpoints
   • Requires Signing Secret for request verification

[bold]Setup Steps:[/bold]
   1. Create a Slack app at https://api.slack.com/apps
   2. Add Bot Token Scopes: app_mentions:read, chat:write, im:read, im:write
   3. Install app to your workspace
   4. Copy Bot User OAuth Token (starts with xoxb-)
   5. For Socket Mode: enable Socket Mode and copy App-Level Token (xapp-)
   6. For HTTP Mode: add Request URL and copy Signing Secret""",
            title="📚 Slack Architecture",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        # Select mode
        mode = questionary.select(
            "Choose Slack connection mode:",
            choices=[
                Choice("🔌 Socket Mode (WebSocket, no public URL needed)", "socket"),
                Choice("🌐 HTTP Mode (webhook, requires public URL)", "http"),
            ],
            use_indicator=True,
        ).ask()
        
        slack_config = self.config.setdefault("slack", {})
        slack_config["mode"] = mode
        
        console.print(f"\n[bold]Configuring {mode} mode...[/]")
        
        # Bot token (required for both)
        current_token = slack_config.get("bot_token", "")
        bot_token = questionary.password(
            "Bot User OAuth Token (xoxb-...)",
            default=current_token,
            instruction="From OAuth & Permissions page"
        ).ask()
        slack_config["bot_token"] = bot_token
        
        if mode == "socket":
            # App token for Socket Mode
            current_app_token = slack_config.get("app_token", "")
            app_token = questionary.password(
                "App-Level Token (xapp-...)",
                default=current_app_token,
                instruction="From Basic Information → App-Level Tokens"
            ).ask()
            slack_config["app_token"] = app_token
            
            console.print(Panel(
                """[bold]Socket Mode Setup Complete![/bold]

Your bot will connect via WebSocket. No public URL needed.

[bold]Next steps:[/bold]
1. Ensure your bot has these Bot Token Scopes:
   • app_mentions:read
   • chat:write
   • im:read, im:write
   • files:write (for file uploads)

2. Start lollmsbot gateway - the bot will connect automatically

3. Invite @lollmsbot to channels or DM it directly""",
                title="Socket Mode Ready",
                border_style="green"
            ))
            
        else:  # http mode
            # Signing secret for HTTP mode
            current_secret = slack_config.get("signing_secret", "")
            signing_secret = questionary.password(
                "Signing Secret",
                default=current_secret,
                instruction="From Basic Information → Signing Secret"
            ).ask()
            slack_config["signing_secret"] = signing_secret
            
            console.print(Panel(
                """[bold]HTTP Mode Setup[/bold]

You need to configure the Request URL in Slack:

1. Go to Event Subscriptions → Enable Events
2. Request URL: https://your-domain/slack/events
   (Or your custom path if mounted differently)

3. Subscribe to bot events:
   • app_mention
   • message.im (for DMs)

4. Reinstall app after adding scopes""",
                title="HTTP Mode Configuration",
                border_style="yellow"
            ))
        
        # User/channel restrictions
        self._configure_slack_users(slack_config)
        
        # Mention requirement
        slack_config["require_mention"] = questionary.confirm(
            "Require @mention in channels? (DMs always work without mention)",
            default=slack_config.get("require_mention", True)
        ).ask()
        
        self._configured.add("slack")
        console.print("[green]✅ Slack configured![/]")
        
        # Show summary
        self._show_slack_summary(slack_config)
    
    def _configure_slack_users(self, slack_config: Dict[str, Any]) -> None:
        """Configure allowed/blocked users and channels for Slack."""
        # Allowed users (Slack user IDs)
        current_allowed = slack_config.get("allowed_users", [])
        allowed_str = ",".join(current_allowed) if current_allowed else ""
        
        allowed_input = questionary.text(
            "Allowed Slack User IDs (comma-separated, optional - empty = allow all)",
            default=allowed_str,
            instruction="Format: U1234567890, U0987654321 (from user profiles)"
        ).ask()
        
        if allowed_input.strip():
            slack_config["allowed_users"] = [
                u.strip() for u in allowed_input.split(",") if u.strip()
            ]
        else:
            slack_config["allowed_users"] = []
        
        # Allowed channels
        current_channels = slack_config.get("allowed_channels", [])
        channels_str = ",".join(current_channels) if current_channels else ""
        
        channels_input = questionary.text(
            "Allowed Channel IDs (comma-separated, optional)",
            default=channels_str,
            instruction="Format: C1234567890 (from channel details)"
        ).ask()
        
        if channels_input.strip():
            slack_config["allowed_channels"] = [
                c.strip() for c in channels_input.split(",") if c.strip()
            ]
        else:
            slack_config["allowed_channels"] = []
        
        # Blocked users
        current_blocked = slack_config.get("blocked_users", [])
        blocked_str = ",".join(current_blocked) if current_blocked else ""
        
        blocked_input = questionary.text(
            "Blocked Slack User IDs (comma-separated, optional)",
            default=blocked_str
        ).ask()
        
        if blocked_input.strip():
            slack_config["blocked_users"] = [
                u.strip() for u in blocked_input.split(",") if u.strip()
            ]
        else:
            slack_config["blocked_users"] = []
    
    def _show_slack_summary(self, slack_config: Dict[str, Any]) -> None:
        """Show Slack configuration summary."""
        table = Table(title="Slack Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        mode = slack_config.get("mode", "socket")
        mode_display = "🔌 Socket Mode" if mode == "socket" else "🌐 HTTP Mode"
        
        table.add_row("Mode", mode_display)
        
        has_token = bool(slack_config.get("bot_token"))
        table.add_row("Bot Token", "✅ Set" if has_token else "❌ Not set")
        
        if mode == "socket":
            has_app_token = bool(slack_config.get("app_token"))
            table.add_row("App Token", "✅ Set" if has_app_token else "❌ Not set")
        else:
            has_secret = bool(slack_config.get("signing_secret"))
            table.add_row("Signing Secret", "✅ Set" if has_secret else "❌ Not set")
        
        allowed_users = slack_config.get("allowed_users", [])
        allowed_channels = slack_config.get("allowed_channels", [])
        blocked_users = slack_config.get("blocked_users", [])
        
        table.add_row("Allowed Users", str(len(allowed_users)) if allowed_users else "All")
        table.add_row("Allowed Channels", str(len(allowed_channels)) if allowed_channels else "All")
        table.add_row("Blocked Users", str(len(blocked_users)))
        table.add_row("Require Mention", "Yes" if slack_config.get("require_mention") else "No")
        
        console.print(table)
    
    def _test_single(self, service_name: str, config: Dict[str, Any]) -> tuple[str, str]:
        """Test a single service connection."""
        try:
            if service_name == "lollms":
                settings = LollmsSettings(
                    host_address=config.get("host_address", ""),
                    api_key=config.get("api_key"),
                    binding_name=config.get("binding_name"),
                    model_name=config.get("model_name"),
                    context_size=config.get("context_size", 4096),
                    verify_ssl=config.get("verify_ssl", True),
                )
                client = build_lollms_client(settings)
                
                if client:
                    # Show model and binding info
                    binding = config.get("binding_name", "unknown")
                    model = config.get("model_name", "default")
                    return ("✅ READY", f"{binding}/{model}")
                else:
                    return ("❌ ERROR", "Client initialization failed")
                    
            elif service_name == "discord":
                token = config.get("bot_token", "")
                has_token = bool(token)
                allowed_users = config.get("allowed_users", [])
                return (
                    "🔍 CONFIGURED" if has_token else "⭕ NO TOKEN",
                    f"Token: {'✅' if has_token else '❌'}, Users: {len(allowed_users)}"
                )
                
            elif service_name == "telegram":
                token = config.get("bot_token", "")
                has_token = bool(token)
                return (
                    "🔍 CONFIGURED" if has_token else "⭕ NO TOKEN",
                    f"Token: {'✅' if has_token else '❌'}"
                )
            
            elif service_name == "whatsapp":
                backend = config.get("backend", "unknown")
                if backend == "web_js":
                    has_path = bool(config.get("web_js_path"))
                    return (
                        "🔍 CONFIGURED" if has_path else "⭕ INCOMPLETE",
                        f"Backend: {backend}, Path: {'✅' if has_path else '❌'}"
                    )
                elif backend == "twilio":
                    has_creds = all([
                        config.get("account_sid"),
                        config.get("auth_token"),
                        config.get("from_number")
                    ])
                    return (
                        "🔍 CONFIGURED" if has_creds else "⭕ INCOMPLETE",
                        f"Backend: {backend}, Creds: {'✅' if has_creds else '❌'}"
                    )
                elif backend == "business_api":
                    has_token = bool(config.get("api_token"))
                    return (
                        "🔍 CONFIGURED" if has_token else "⭕ INCOMPLETE",
                        f"Backend: {backend}, Token: {'✅' if has_token else '❌'}"
                    )
                return ("⭕ NO BACKEND", "Backend not selected")
                
            return ("❓ SKIP", "-")
            
        except Exception as e:
            return ("❌ ERROR", str(e)[:40])

    def show_full_config(self) -> None:
        """Display complete configuration."""
        console.print("\n[bold]📄 Full Configuration[/]")
        
        # Backend with binding details
        if "lollms" in self.config:
            lollms = self.config["lollms"]
            console.print(Panel(
                json.dumps(lollms, indent=2),
                title=f"AI Backend: {lollms.get('binding_name', 'unknown')}",
                border_style="blue"
            ))
        
        # Other services
        for svc in ["discord", "telegram", "slack"]:
            if svc in self.config:
                # Mask secrets
                safe_config = {k: (v if "token" not in k and "secret" not in k else "***") for k, v in self.config[svc].items()}
                console.print(Panel(
                    json.dumps(safe_config, indent=2),
                    title=svc.title(),
                    border_style="blue"
                ))
        
        # Soul
        console.print(Panel(
            json.dumps(self.soul.to_dict(), indent=2),
            title=f"Soul: {self.soul.name}",
            border_style="magenta"
        ))
        
        # Heartbeat
        hb_status = self.heartbeat.get_status()
        console.print(Panel(
            json.dumps(hb_status, indent=2),
            title="Heartbeat Status",
            border_style="green"
        ))
        
        # Skills
        skill_stats = self._get_skill_stats()
        console.print(Panel(
            json.dumps(skill_stats, indent=2),
            title=f"Skills Library ({skill_stats['total']} skills)",
            border_style="yellow"
        ))

    def _save_all(self) -> None:
        """Save all configurations."""
        self.soul._save()
        self.heartbeat._save_config()
        
        self.config["soul"] = {
            "name": self.soul.name,
            "version": self.soul.version,
            "trait_count": len(self.soul.traits),
            "value_count": len(self.soul.values),
        }
        
        self.config["skills"] = {
            "total_loaded": len(self.skill_registry._skills),
            "categories": list(self.skill_registry._categories.keys()),
        }
        
        self._save_config()


# Entry point
def run_wizard() -> None:
    """CLI entrypoint."""
    try:
        Wizard().run_wizard()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Bye![/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise


if __name__ == "__main__":
    run_wizard()
