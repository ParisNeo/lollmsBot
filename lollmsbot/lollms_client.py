"""
LoLLMs client for HTTP and WebSocket communication.

This module provides a client for interacting with LoLLMs servers,
supporting both REST API calls for generation and WebSocket connections
for streaming responses.
"""

import asyncio
import json
import ssl
from typing import Any, AsyncGenerator, Dict, Optional, Union
from urllib.parse import urljoin

import aiohttp
import websockets

from lollmsbot.config import LollmsConfig


class LollmsError(Exception):
    """Base exception for LoLLMs client errors."""

    pass


class LollmsConnectionError(LollmsError):
    """Raised when connection to LoLLMs server fails."""

    pass


class LollmsGenerationError(LollmsError):
    """Raised when text generation fails."""

    pass


class LollmsTimeoutError(LollmsError):
    """Raised when a request times out."""

    pass


class LollmsAuthenticationError(LollmsError):
    """Raised when authentication fails."""

    pass


class LollmsClient:
    """Client for interacting with LoLLMs servers via HTTP and WebSocket.

    This client provides both synchronous-style generation through the REST API
    and streaming generation through WebSocket connections. It uses aiohttp for
    HTTP communication and the websockets library for streaming.

    Attributes:
        config: LollmsConfig instance containing server connection details.
        session: Optional aiohttp ClientSession for HTTP requests.
        ws_connection: Optional WebSocket connection for streaming.
        connected: Boolean indicating if WebSocket is connected.

    Example:
        >>> config = LollmsConfig(host="localhost", port=9600)
        >>> client = LollmsClient(config)
        >>> response = await client.generate("Hello, world!")
        >>> await client.disconnect()
    """

    def __init__(self, config: LollmsConfig) -> None:
        """Initialize the LoLLMs client.

        Args:
            config: Configuration object containing server host, port,
                   API key, and timeout settings.
        """
        self.config: LollmsConfig = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self._connected: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        """Check if WebSocket connection is active."""
        return self._connected and self.ws_connection is not None

    def _get_headers(self) -> Dict[str, str]:
        """Generate HTTP headers including authentication if configured."""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _get_ws_url(self, endpoint: str = "/ws/generate") -> str:
        """Construct WebSocket URL from base configuration."""
        base_ws_url = f"ws://{self.config.host}:{self.config.port}"
        return urljoin(base_ws_url + "/", endpoint.lstrip("/"))

    def _get_http_url(self, endpoint: str = "/api/generate") -> str:
        """Construct HTTP URL from base configuration."""
        return urljoin(self.config.base_url + "/", endpoint.lstrip("/"))

    async def connect(self) -> None:
        """Establish WebSocket connection for streaming generation.

        This method creates a persistent WebSocket connection that can be
        used for multiple streaming requests. The connection is maintained
        until explicitly disconnected.

        Raises:
            LollmsConnectionError: If the WebSocket connection fails.
            LollmsAuthenticationError: If authentication is rejected.
        """
        if self.connected:
            return

        ws_url = self._get_ws_url()

        headers: Dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        try:
            self.ws_connection = await websockets.connect(
                ws_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            self._connected = True
        except websockets.InvalidStatusCode as exc:
            if exc.status_code == 401:
                raise LollmsAuthenticationError(
                    f"Authentication failed for {ws_url}"
                ) from exc
            raise LollmsConnectionError(
                f"WebSocket connection failed with status {exc.status_code}: {ws_url}"
            ) from exc
        except websockets.WebSocketException as exc:
            raise LollmsConnectionError(
                f"WebSocket connection failed: {ws_url}"
            ) from exc
        except OSError as exc:
            raise LollmsConnectionError(
                f"Network error connecting to {ws_url}: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Close WebSocket connection and cleanup resources.

        This method safely closes any active WebSocket connection and
        the HTTP session. It is safe to call multiple times.
        """
        async with self._lock:
            if self.ws_connection is not None:
                try:
                    await self.ws_connection.close()
                except Exception:
                    pass
                finally:
                    self.ws_connection = None
                    self._connected = False

            if self.session is not None:
                try:
                    await self.session.close()
                except Exception:
                    pass
                finally:
                    self.session = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists, creating if necessary."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                enable_cleanup_closed=True,
                force_close=True,
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self._get_headers(),
            )
        return self.session

    async def generate(
        self,
        prompt: str,
        personality: Optional[int] = None,
        n_predict: Optional[int] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using the LoLLMs REST API.

        This method sends a generation request to the LoLLMs server and
        returns the complete generated text as a string.

        Args:
            prompt: The input text prompt to generate from.
            personality: Optional personality ID to use for generation.
            n_predict: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 to 1.0+).
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            repeat_penalty: Penalty for repeating tokens.
            repeat_last_n: Number of tokens to check for repetition.
            seed: Random seed for reproducible generation.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            The generated text string.

        Raises:
            LollmsConnectionError: If the HTTP connection fails.
            LollmsGenerationError: If the generation request fails.
            LollmsTimeoutError: If the request times out.
            LollmsAuthenticationError: If authentication fails.

        Example:
            >>> response = await client.generate(
            ...     "Write a haiku about Python",
            ...     n_predict=50,
            ...     temperature=0.8
            ... )
        """
        session = await self._ensure_session()

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
        }

        if personality is not None:
            payload["personality"] = personality
        if n_predict is not None:
            payload["n_predict"] = n_predict
        if seed is not None:
            payload["seed"] = seed

        # Merge any additional kwargs
        payload.update(kwargs)

        url = self._get_http_url("/api/generate")

        try:
            async with session.post(url, json=payload) as response:
                if response.status == 401:
                    raise LollmsAuthenticationError(
                        f"Authentication failed for {url}"
                    )
                elif response.status == 404:
                    raise LollmsConnectionError(
                        f"Generation endpoint not found: {url}"
                    )
                elif response.status >= 400:
                    text = await response.text()
                    raise LollmsGenerationError(
                        f"Generation failed with status {response.status}: {text}"
                    )

                data = await response.json()
                return data.get("generated_text", data.get("text", ""))

        except aiohttp.ClientResponseError as exc:
            raise LollmsGenerationError(f"HTTP error during generation: {exc}") from exc
        except aiohttp.ClientConnectorError as exc:
            raise LollmsConnectionError(f"Cannot connect to {url}: {exc}") from exc
        except asyncio.TimeoutError as exc:
            raise LollmsTimeoutError(
                f"Request timed out after {self.config.timeout}s"
            ) from exc
        except aiohttp.ClientError as exc:
            raise LollmsConnectionError(f"HTTP client error: {exc}") from exc

    async def stream_generate(
        self,
        prompt: str,
        personality: Optional[int] = None,
        n_predict: Optional[int] = None,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        repeat_last_n: int = 64,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream text generation via WebSocket.

        This method establishes a WebSocket connection (if not already connected)
        and streams generated tokens as they are produced by the model.

        Args:
            prompt: The input text prompt to generate from.
            personality: Optional personality ID to use for generation.
            n_predict: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 to 1.0+).
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            repeat_penalty: Penalty for repeating tokens.
            repeat_last_n: Number of tokens to check for repetition.
            seed: Random seed for reproducible generation.
            **kwargs: Additional parameters to pass to the API.

        Yields:
            Generated text tokens as they are produced.

        Raises:
            LollmsConnectionError: If the WebSocket connection fails.
            LollmsGenerationError: If the generation stream encounters an error.
            LollmsAuthenticationError: If authentication fails.

        Example:
            >>> async for token in client.stream_generate("Hello"):
            ...     print(token, end="", flush=True)
        """
        # Connect if not already connected
        if not self.connected:
            await self.connect()

        if self.ws_connection is None:
            raise LollmsConnectionError("WebSocket connection not established")

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "stream": True,
        }

        if personality is not None:
            payload["personality"] = personality
        if n_predict is not None:
            payload["n_predict"] = n_predict
        if seed is not None:
            payload["seed"] = seed

        payload.update(kwargs)

        try:
            await self.ws_connection.send(json.dumps(payload))

            while True:
                try:
                    message = await self.ws_connection.recv()
                    data = json.loads(message)

                    # Check for errors in response
                    if "error" in data:
                        raise LollmsGenerationError(f"Generation error: {data['error']}")

                    # Check for completion
                    if data.get("done", False):
                        break

                    # Yield generated token/text
                    token = data.get("token", data.get("chunk", data.get("text", "")))
                    if token:
                        yield token

                except websockets.ConnectionClosed as exc:
                    raise LollmsConnectionError(
                        f"WebSocket connection closed unexpectedly: {exc}"
                    ) from exc
                except json.JSONDecodeError as exc:
                    raise LollmsGenerationError(
                        f"Invalid JSON in stream response: {exc}"
                    ) from exc

        except websockets.WebSocketException as exc:
            self._connected = False
            raise LollmsConnectionError(f"WebSocket error during streaming: {exc}") from exc

    async def health_check(self) -> Dict[str, Any]:
        """Check if the LoLLMs server is reachable and healthy.

        Returns:
            Dictionary containing server status information.

        Raises:
            LollmsConnectionError: If the server is unreachable.
        """
        session = await self._ensure_session()
        url = self._get_http_url("/api/health")

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    return await response.json()
                return {"status": "unknown", "http_status": response.status}
        except aiohttp.ClientError as exc:
            raise LollmsConnectionError(f"Health check failed: {exc}") from exc

    async def list_personalities(self) -> list[Dict[str, Any]]:
        """List available personalities on the LoLLMs server.

        Returns:
            List of personality dictionaries.

        Raises:
            LollmsConnectionError: If the request fails.
            LollmsAuthenticationError: If authentication fails.
        """
        session = await self._ensure_session()
        url = self._get_http_url("/api/personalities")

        try:
            async with session.get(url) as response:
                if response.status == 401:
                    raise LollmsAuthenticationError("Authentication required")
                response.raise_for_status()
                data = await response.json()
                return data.get("personalities", [])
        except aiohttp.ClientError as exc:
            raise LollmsConnectionError(f"Failed to list personalities: {exc}") from exc

    async def __aenter__(self) -> "LollmsClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Async context manager exit, ensuring cleanup."""
        await self.disconnect()

    def __repr__(self) -> str:
        """String representation of the client."""
        status = "connected" if self.connected else "disconnected"
        return f"LollmsClient({self.config.base_url}, {status})"