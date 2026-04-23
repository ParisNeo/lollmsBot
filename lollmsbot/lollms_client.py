from __future__ import annotations

from typing import Any

from lollms_client import LollmsClient  # from lollms-client package[web:21][web:41]

from .config import LollmsSettings


def build_lollms_client(settings: LollmsSettings | None = None) -> LollmsClient:
    """
    Build a LollmsClient either in 'LoLLMS server' mode or 'direct binding' mode
    depending on env settings.
    """
    if settings is None:
        settings = LollmsSettings.from_env()

    # Basic shared kwargs
    client_kwargs: dict[str, Any] = {}

    if settings.host_address:
        client_kwargs["host_address"] = settings.host_address

    # Some lollms_client versions support verify_ssl; if not, this can be removed
    if settings.verify_ssl is False:
        client_kwargs["verify_ssl"] = False

    # Direct binding mode - Ensure binding_name is correctly passed
    binding_name = settings.binding_name or "lollms"

    # Standardize config keys for common bindings
    ctx_size = settings.context_size or 4096

    binding_config = {
        "host_address": settings.host_address,
        "model_name": settings.model_name,
        "service_key": settings.api_key,
        "ctx_size": ctx_size,
        # Common variants for different bindings
        "num_ctx": ctx_size,
        "max_tokens": settings.context_size,
    }

    # Add binding-specific overrides
    if binding_name == "ollama":
        # Ensure URL is clean and includes standard suffix if missing
        host = settings.host_address.rstrip('/')
        binding_config["host_address"] = host
        binding_config["base_url"] = host
        # Ollama specific parameter for context window
        binding_config["options"] = {
            "num_ctx": ctx_size
        }

    return LollmsClient(
        llm_binding_name=binding_name,
        llm_binding_config=binding_config,
    )
