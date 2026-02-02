# lollmsbot/lollms_client.py
from lollms_client import LollmsClient  # from lollms-client package[web:21][web:41]
from .config import LollmsSettings

def build_lollms_client(settings: LollmsSettings | None = None) -> LollmsClient:
    if settings is None:
        settings = LollmsSettings.from_env()

    kwargs: dict = {}

    # host / SSL
    if settings.host_address:
        kwargs["host_address"] = settings.host_address
    if not settings.verify_ssl:
        kwargs["verify_ssl"] = False  # assuming client supports this flag

    # binding + model
    if settings.binding_name:
        # direct binding mode: LollmsClient(binding_name, ...)
        return LollmsClient(
            settings.binding_name,
            host_address=settings.host_address,
            model_name=settings.model_name,
            service_key=settings.api_key,
            ctx_size=settings.context_size,
            **kwargs,
        )
    else:
        # plain LoLLMS server mode
        return LollmsClient(
            host_address=settings.host_address,
            service_key=settings.api_key,
            **kwargs,
        )
