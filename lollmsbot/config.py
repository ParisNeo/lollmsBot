#!/usr/bin/env python
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import json

load_dotenv()

console = None  # Forward ref

def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")

@dataclass
class BotConfig:
    """Bot behavior configuration settings."""
    name: str = field(default="LollmsBot")
    max_history: int = field(default=10)
    
    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load from environment variables."""
        return cls(
            name=os.getenv("LOLLMSBOT_NAME", "LollmsBot"),
            max_history=int(os.getenv("LOLLMSBOT_MAX_HISTORY", "10")),
        )

@dataclass
class LollmsSettings:
    """LoLLMS connection settings."""
    host_address: str = field(default="http://localhost:9600")
    api_key: Optional[str] = field(default=None)
    verify_ssl: bool = field(default=True)
    binding_name: Optional[str] = field(default=None)
    model_name: Optional[str] = field(default=None)
    context_size: Optional[int] = field(default=None)

    @classmethod
    def from_env(cls) -> "LollmsSettings":
        """Load from environment variables."""
        global console
        return cls(
            host_address=os.getenv("LOLLMS_HOST_ADDRESS", "http://localhost:9600"),
            api_key=os.getenv("LOLLMS_API_KEY"),
            verify_ssl=_get_bool("LOLLMS_VERIFY_SSL", True),
            binding_name=os.getenv("LOLLMS_BINDING_NAME"),
            model_name=os.getenv("LOLLMS_MODEL_NAME"),
            context_size=int(os.getenv("LOLLMS_CONTEXT_SIZE", "32000")) or None,
        )

    @classmethod
    def from_wizard(cls) -> "LollmsSettings":
        """Load from wizard config."""
        wizard_path = Path.home() / ".lollmsbot" / "config.json"
        if not wizard_path.exists():
            return cls.from_env()
        
        try:
            wizard_data = json.loads(wizard_path.read_text())
            lollms_data = wizard_data.get("lollms", {})
            if lollms_data.get("host_address"):
                console.print("[green]📡 Using wizard config![/]" if console else "Using wizard config")
                return cls(
                    host_address=lollms_data.get("host_address", "http://localhost:9600"),
                    api_key=lollms_data.get("api_key"),
                    verify_ssl=_get_bool(str(lollms_data.get("verify_ssl", True))),
                    binding_name=lollms_data.get("binding_name"),
                )
        except:
            pass
        return cls.from_env()

@dataclass
class GatewaySettings:
    """Gateway server settings."""
    host: str = field(default="localhost")
    port: int = field(default=8800)

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            host=os.getenv("LOLLMSBOT_HOST", "localhost"),
            port=int(os.getenv("LOLLMSBOT_PORT", "8800")),
        )
