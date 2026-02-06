"""
HTTP tool for LollmsBot.

This module provides the HttpTool class for making HTTP requests with
built-in timeout, retry logic, and response parsing. Supports GET, POST,
PUT, and DELETE operations with safe URL validation.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientResponse

from lollmsbot.agent import Tool, ToolResult, ToolError


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    exponential_base: float = 2.0


class HttpTool(Tool):
    """Tool for making HTTP requests with retry logic and safe URL validation.
    
    This tool provides methods for GET, POST, PUT, and DELETE HTTP operations
    with built-in timeout handling, automatic retries with exponential backoff,
    and intelligent response parsing (JSON or text).
    
    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema describing expected parameters.
        default_timeout: Default request timeout in seconds.
        retry_config: Configuration for retry behavior.
        allowed_schemes: Set of allowed URL schemes for security.
        max_response_size: Maximum response size in bytes.
    """
    
    name: str = "http"
    description: str = (
        "Make HTTP requests (GET, POST, PUT, DELETE) to external APIs "
        "and web services. Automatically parses JSON responses and "
        "handles timeouts, retries, and errors gracefully."
    )
    
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["get", "post", "put", "delete"],
                "description": "HTTP method to use",
            },
            "url": {
                "type": "string",
                "description": "Target URL for the request",
            },
            "data": {
                "type": "object",
                "description": "Request body data (for POST, PUT)",
            },
            "headers": {
                "type": "object",
                "description": "Additional HTTP headers",
            },
            "params": {
                "type": "object",
                "description": "URL query parameters",
            },
        },
        "required": ["method", "url"],
    }
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        allowed_schemes: Optional[set[str]] = None,
        max_response_size: int = 10 * 1024 * 1024,  # 10 MB
    ) -> None:
        """Initialize the HttpTool.
        
        Args:
            default_timeout: Default request timeout in seconds.
            retry_config: Retry configuration. Uses defaults if None.
            allowed_schemes: Set of allowed URL schemes. Defaults to http, https.
            max_response_size: Maximum response size in bytes.
        """
        self.default_timeout: float = default_timeout
        self.retry_config: RetryConfig = retry_config or RetryConfig()
        self.allowed_schemes: set[str] = allowed_schemes or {"http", "https"}
        self.max_response_size: int = max_response_size
        
        # Create session with connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: asyncio.Lock = asyncio.Lock()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = ClientTimeout(total=self.default_timeout)
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    enable_cleanup_closed=True,
                    force_close=True,
                )
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    headers={
                        "User-Agent": "LollmsBot-HttpTool/0.1.0",
                        "Accept": "application/json, text/plain, */*",
                    },
                )
            return self._session
    
    def _validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL scheme and format.
        
        Args:
            url: URL to validate.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in self.allowed_schemes:
                allowed = ", ".join(self.allowed_schemes)
                return False, f"URL scheme '{parsed.scheme}' not allowed. Allowed: {allowed}"
            
            # Check netloc (host)
            if not parsed.netloc:
                return False, "URL must include a host"
            
            # Block localhost and private IPs by default for security
            hostname = parsed.hostname
            if hostname:
                blocked_hosts = {"localhost", "127.0.0.1", "localhost", "::1"}
                if hostname in blocked_hosts:
                    return False, f"Access to '{hostname}' is not allowed"
                
                # Check for private IP ranges
                if hostname.startswith("10.") or hostname.startswith("192.168.") or hostname.startswith("172."):
                    return False, f"Access to private IP '{hostname}' is not allowed"
            
            return True, None
            
        except ValueError as exc:
            return False, f"Invalid URL format: {str(exc)}"
    
    async def _execute_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> tuple[bool, Any, Optional[str], Dict[str, Any]]:
        """Execute HTTP request with retry logic.
        
        Args:
            method: HTTP method (get, post, put, delete).
            url: Target URL.
            **kwargs: Additional arguments for aiohttp request.
            
        Returns:
            Tuple of (success, output_data, error_message, metadata).
        """
        session = await self._get_session()
        
        last_error: Optional[str] = None
        metadata: Dict[str, Any] = {
            "attempts": 0,
            "url": url,
            "method": method.upper(),
        }
        
        for attempt in range(self.retry_config.max_retries):
            metadata["attempts"] = attempt + 1
            
            try:
                async with session.request(method, url, **kwargs) as response:
                    return await self._handle_response(response, metadata)
                    
            except asyncio.TimeoutError:
                last_error = f"Request timeout after {self.default_timeout}s"
                
            except ClientError as exc:
                last_error = f"HTTP client error: {str(exc)}"
                
            except Exception as exc:
                last_error = f"Unexpected error: {str(exc)}"
            
            # Calculate delay with exponential backoff
            if attempt < self.retry_config.max_retries - 1:
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay,
                )
                await asyncio.sleep(delay)
        
        # All retries exhausted
        return False, None, last_error, metadata
    
    async def _handle_response(
        self,
        response: ClientResponse,
        metadata: Dict[str, Any],
    ) -> tuple[bool, Any, Optional[str], Dict[str, Any]]:
        """Process HTTP response and parse content.
        
        Args:
            response: aiohttp ClientResponse.
            metadata: Metadata dict to augment.
            
        Returns:
            Tuple of (success, output_data, error_message, metadata).
        """
        metadata["status_code"] = response.status
        metadata["content_type"] = response.content_type or "unknown"
        
        # Check for HTTP error status
        if response.status >= 400:
            try:
                error_text = await response.text()
            except Exception:
                error_text = "Could not read error response"
            
            return False, None, f"HTTP {response.status}: {error_text[:500]}", metadata
        
        # Check response size
        content_length = response.content_length
        if content_length and content_length > self.max_response_size:
            return False, None, f"Response too large: {content_length} bytes", metadata
        
        # Read response content with size limit
        try:
            text = await response.text()
            
            if len(text.encode("utf-8")) > self.max_response_size:
                return False, None, f"Response exceeds max size of {self.max_response_size} bytes", metadata
            
        except Exception as exc:
            return False, None, f"Failed to read response: {str(exc)}", metadata
        
        # Try to parse as JSON
        try:
            parsed = json.loads(text)
            metadata["parsed_as"] = "json"
            return True, parsed, None, metadata
        except json.JSONDecodeError:
            # Return as text
            metadata["parsed_as"] = "text"
            return True, text, None, metadata
    
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute GET request.
        
        Args:
            url: Target URL.
            headers: Optional additional headers.
            params: Optional query parameters.
            
        Returns:
            ToolResult with parsed response data.
        """
        # Validate URL
        is_valid, error = self._validate_url(url)
        if not is_valid:
            return ToolResult(success=False, output=None, error=error)
        
        # Prepare request arguments
        kwargs: Dict[str, Any] = {}
        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        
        success, output, error, metadata = await self._execute_with_retry("GET", url, **kwargs)
        
        return ToolResult(
            success=success,
            output=output,
            error=error,
            execution_time=0.0,  # Tracked per attempt internally
        )
    
    async def post(
        self,
        url: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute POST request.
        
        Args:
            url: Target URL.
            data: Request body data (dict for JSON, str for raw body).
            headers: Optional additional headers.
            params: Optional query parameters.
            
        Returns:
            ToolResult with parsed response data.
        """
        # Validate URL
        is_valid, error = self._validate_url(url)
        if not is_valid:
            return ToolResult(success=False, output=None, error=error)
        
        # Prepare request arguments
        kwargs: Dict[str, Any] = {}
        
        # Determine content type and format data
        if data is not None:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data
        
        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        
        success, output, error, metadata = await self._execute_with_retry("POST", url, **kwargs)
        
        return ToolResult(
            success=success,
            output=output,
            error=error,
        )
    
    async def put(
        self,
        url: str,
        data: Optional[Union[Dict[str, Any], str]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute PUT request.
        
        Args:
            url: Target URL.
            data: Request body data (dict for JSON, str for raw body).
            headers: Optional additional headers.
            params: Optional query parameters.
            
        Returns:
            ToolResult with parsed response data.
        """
        # Validate URL
        is_valid, error = self._validate_url(url)
        if not is_valid:
            return ToolResult(success=False, output=None, error=error)
        
        # Prepare request arguments
        kwargs: Dict[str, Any] = {}
        
        if data is not None:
            if isinstance(data, dict):
                kwargs["json"] = data
            else:
                kwargs["data"] = data
        
        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        
        success, output, error, metadata = await self._execute_with_retry("PUT", url, **kwargs)
        
        return ToolResult(
            success=success,
            output=output,
            error=error,
        )
    
    async def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute DELETE request.
        
        Args:
            url: Target URL.
            headers: Optional additional headers.
            params: Optional query parameters.
            
        Returns:
            ToolResult with parsed response data.
        """
        # Validate URL
        is_valid, error = self._validate_url(url)
        if not is_valid:
            return ToolResult(success=False, output=None, error=error)
        
        # Prepare request arguments
        kwargs: Dict[str, Any] = {}
        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        
        success, output, error, metadata = await self._execute_with_retry("DELETE", url, **kwargs)
        
        return ToolResult(
            success=success,
            output=output,
            error=error,
        )
    
    async def execute(self, **params: Any) -> ToolResult:
        """Execute HTTP request based on parameters.
        
        Main entry point for Tool base class. Dispatches to appropriate
        HTTP method based on the 'method' parameter.
        
        Args:
            **params: Parameters must include:
                - method: HTTP method (get, post, put, delete)
                - url: Target URL
                - data: Request body (for POST, PUT)
                - headers: Additional headers
                - params: Query parameters
                
        Returns:
            ToolResult from the executed HTTP request.
        """
        method = params.get("method", "").lower()
        url = params.get("url")
        
        if not method:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: 'method'",
            )
        
        if not url:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: 'url'",
            )
        
        # Extract optional parameters
        data = params.get("data")
        headers = params.get("headers")
        query_params = params.get("params")
        
        # Dispatch to appropriate method
        if method == "get":
            return await self.get(url, headers=headers, params=query_params)
        
        elif method == "post":
            return await self.post(url, data=data, headers=headers, params=query_params)
        
        elif method == "put":
            return await self.put(url, data=data, headers=headers, params=query_params)
        
        elif method == "delete":
            return await self.delete(url, headers=headers, params=query_params)
        
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown HTTP method: '{method}'. Valid methods: get, post, put, delete",
            )
    
    async def close(self) -> None:
        """Close the aiohttp session and cleanup resources."""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
    
    def __repr__(self) -> str:
        return f"HttpTool(timeout={self.default_timeout}, max_retries={self.retry_config.max_retries})"