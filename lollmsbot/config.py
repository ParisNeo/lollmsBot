# lollmsbot/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # local dev; in prod, rely on real env vars[web:46][web:50]

def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")

@dataclass
class LollmsSettings:
    host_address: str
    api_key: str | None
    verify_ssl: bool
    binding_name: str | None
    model_name: str | None
    context_size: int | None

    @classmethod
    def from_env(cls) -> "LollmsSettings":
        return cls(
            host_address=os.getenv("LOLLMS_HOST_ADDRESS", "http://localhost:9600"),
            api_key=os.getenv("LOLLMS_API_KEY") or None,
            verify_ssl=_get_bool("LOLLMS_VERIFY_SSL", True),
            binding_name=os.getenv("LOLLMS_BINDING_NAME") or None,
            model_name=os.getenv("LOLLMS_MODEL_NAME") or None,
            context_size=int(os.getenv("LOLLMS_CONTEXT_SIZE",32000))
            if os.getenv("LOLLMS_CONTEXT_SIZE")
            else None,
        )
