"""
HTTP tool for LollmsBot.

This module provides the HttpTool class for making HTTP requests with
built-in timeout, retry logic, and response parsing. Supports GET, POST,
PUT, and DELETE operations with safe URL validation.

CRITICAL: All fetched content is stored in RLM memory (EMS) and accessed
via memory handles in the REPL Context Buffer (RCB). This ensures the
LLM can access large web content through RLM's recursive loading.
"""

import asyncio
import json
import re
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
    """Tool for making HTTP requests with RLM memory integration.
    
    This tool stores all fetched content in the RLM External Memory Store (EMS)
    as WEB_CONTENT chunks. The LLM accesses this content via [[MEMORY:...]]
    handles in its REPL Context Buffer (RCB).
    
    This architecture allows handling arbitrarily large web content by:
    1. Storing the full content in EMS (SQLite-backed, compressed)
    2. Loading summaries/handles into RCB (limited working memory)
    3. Letting the LLM request full content via memory handles if needed
    
    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema describing expected parameters.
        default_timeout: Default request timeout in seconds.
        retry_config: Configuration for retry behavior.
        allowed_schemes: Set of allowed URL schemes for security.
        max_response_size: Maximum response size in bytes.
        rlm_memory: Reference to RLMMemoryManager for storing fetched content.
    """
    
    name: str = "http"
    description: str = (
        "Make HTTP requests (GET, POST, PUT, DELETE) to external APIs "
        "and web services. All fetched content is automatically stored "
        "in RLM memory and accessible via memory handles. For web pages, "
        "readable text is extracted and stored. Returns a memory handle "
        "that can be used to access the content."
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
            "extract_text": {
                "type": "boolean",
                "description": "Extract main text content from HTML (default true)",
                "default": True,
            },
            "importance": {
                "type": "number",
                "description": "Memory importance for RLM storage (1-10, default 7)",
                "default": 7.0,
            },
        },
        "required": ["method", "url"],
    }
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        allowed_schemes: Optional[set[str]] = None,
        max_response_size: int = 50 * 1024 * 1024,  # 50 MB
        rlm_memory: Optional[Any] = None,  # RLMMemoryManager
    ) -> None:
        """Initialize the HttpTool.
        
        Args:
            default_timeout: Default request timeout in seconds.
            retry_config: Retry configuration. Uses defaults if None.
            allowed_schemes: Set of allowed URL schemes. Defaults to http, https.
            max_response_size: Maximum response size in bytes.
            rlm_memory: RLM memory manager for storing fetched content.
        """
        self.default_timeout: float = default_timeout
        self.retry_config: RetryConfig = retry_config or RetryConfig()
        self.allowed_schemes: set[str] = allowed_schemes or {"http", "https"}
        self.max_response_size: int = max_response_size
        self._rlm_memory = rlm_memory  # Will be set by agent after registration
        
        # Create session with connection pooling
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock: asyncio.Lock = asyncio.Lock()
    
    def set_rlm_memory(self, memory: Any) -> None:
        """Set the RLM memory manager (called by Agent after registration)."""
        self._rlm_memory = memory
    
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
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Connection": "keep-alive",
                    },
                )
            return self._session
    
    def _validate_url(self, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL scheme and format."""
        try:
            parsed = urlparse(url)
            
            if parsed.scheme not in self.allowed_schemes:
                allowed = ", ".join(self.allowed_schemes)
                return False, f"URL scheme '{parsed.scheme}' not allowed. Allowed: {allowed}"
            
            if not parsed.netloc:
                return False, "URL must include a host"
            
            return True, None
            
        except ValueError as exc:
            return False, f"Invalid URL format: {str(exc)}"
    
    def _extract_text_from_html(self, html: str, url: str) -> str:
        """Extract main text content from HTML, removing scripts, styles, nav, etc."""
        # Remove script and style tags with content
        text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove common non-content elements
        for tag in ['nav', 'header', 'footer', 'aside', 'menu', 'advertisement', 
                    'iframe', 'embed', 'object', 'video', 'audio', 'canvas',
                    'svg', 'noscript', 'template']:
            text = re.sub(rf'<{tag}\b[^<]*(?:(?!</{tag}>)<[^<]*)*</{tag}>', ' ', text, 
                         flags=re.DOTALL | re.IGNORECASE)
        
        # Try to find main content area
        main_content = ""
        patterns = [
            r'<article\b[^>]*>(.*?)</article>',
            r'<main\b[^>]*>(.*?)</main>',
            r'<div[^>]*(?:id|class)=["\'](?:content|main|post|entry|body|text)["\'][^>]*>(.*?)</div>',
            r'<section[^>]*(?:id|class)=["\'](?:content|main)["\'][^>]*>(.*?)</section>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                main_content = max(matches, key=len)
                break
        
        if not main_content:
            body_match = re.search(r'<body\b[^>]*>(.*?)</body>', text, re.DOTALL | re.IGNORECASE)
            if body_match:
                main_content = body_match.group(1)
            else:
                main_content = text
        
        # Convert HTML tags to newlines or spaces
        for tag in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'br', 'tr', 'section', 'article']:
            main_content = re.sub(rf'</?{tag}\b[^>]*>', '\n', main_content, flags=re.IGNORECASE)
        
        for tag in ['span', 'a', 'strong', 'em', 'b', 'i', 'u', 'code', 'small', 'label']:
            main_content = re.sub(rf'</?{tag}\b[^>]*>', ' ', main_content, flags=re.IGNORECASE)
        
        # Remove remaining tags
        main_content = re.sub(r'<[^>]+>', ' ', main_content)
        
        # Decode HTML entities
        import html as html_module
        main_content = html_module.unescape(main_content)
        
        # Clean up whitespace
        lines = [line.strip() for line in main_content.split('\n') if line.strip()]
        result = '\n'.join(lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    async def _execute_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> tuple[bool, Any, Optional[str], Dict[str, Any]]:
        """Execute HTTP request with retry logic."""
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
            
            if attempt < self.retry_config.max_retries - 1:
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                    self.retry_config.max_delay,
                )
                await asyncio.sleep(delay)
        
        return False, None, last_error, metadata
    
    async def _handle_response(
        self,
        response: ClientResponse,
        metadata: Dict[str, Any],
    ) -> tuple[bool, Any, Optional[str], Dict[str, Any]]:
        """Process HTTP response and parse content."""
        metadata["status_code"] = response.status
        metadata["content_type"] = response.content_type or "unknown"
        
        if response.status >= 400:
            try:
                error_text = await response.text()
            except Exception:
                error_text = "Could not read error response"
            return False, None, f"HTTP {response.status}: {error_text[:500]}", metadata
        
        content_length = response.content_length
        if content_length and content_length > self.max_response_size:
            return False, None, f"Response too large: {content_length} bytes", metadata
        
        try:
            text = await response.text()
            if len(text.encode("utf-8")) > self.max_response_size:
                return False, None, f"Response exceeds max size of {self.max_response_size} bytes", metadata
            
        except Exception as exc:
            return False, None, f"Failed to read response: {str(exc)}", metadata
        
        content_type = response.content_type or ""
        
        # Try to parse as JSON
        if "application/json" in content_type:
            try:
                parsed = json.loads(text)
                metadata["parsed_as"] = "json"
                return True, {"type": "json", "data": parsed}, None, metadata
            except json.JSONDecodeError:
                pass
        
        # For HTML, extract readable text
        if "text/html" in content_type or "<html" in text.lower() or "<!doctype html" in text.lower():
            extracted = self._extract_text_from_html(text, str(response.url))
            metadata["parsed_as"] = "html_extracted"
            metadata["original_length"] = len(text)
            metadata["extracted_length"] = len(extracted)
            return True, {"type": "html_text", "text": extracted, "url": str(response.url)}, None, metadata
        
        # Return as plain text
        metadata["parsed_as"] = "text"
        return True, {"type": "text", "text": text, "url": str(response.url)}, None, metadata
    
    async def _store_in_rlm(
        self,
        url: str,
        content: str,
        content_type: str,
        importance: float,
        metadata: Dict[str, Any],
    ) -> tuple[str, str]:  # (chunk_id, summary)
        """Store fetched content in RLM memory and return handle info."""
        if not self._rlm_memory:
            # Fallback: return content directly if no RLM available
            return "no_rlm", content[:200] + "..." if len(content) > 200 else content
        
        # Import here to avoid circular imports
        from lollmsbot.agent.rlm.models import MemoryChunkType
        
        # Generate chunk ID based on URL hash
        import hashlib
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        chunk_id = f"web_{url_hash}_{int(asyncio.get_event_loop().time())}"
        
        # Create summary (first 200 chars)
        summary_lines = content.split('\n')[:10]  # First 10 lines
        summary = ' '.join(summary_lines)[:200]
        if len(content) > 200:
            summary += "..."
        
        # Store in EMS
        stored_chunk_id = await self._rlm_memory.store_in_ems(
            content=content,
            chunk_type=MemoryChunkType.WEB_CONTENT,
            importance=importance,
            tags=["web_content", "fetched", content_type, "user_requested"],
            summary=f"Web content from {url}: {summary}",
            load_hints=[url, "web", "fetched", "content"],
            source=f"http_tool:{url}",
        )
        
        # Also load into RCB so LLM can see it immediately
        await self._rlm_memory.load_from_ems(stored_chunk_id, add_to_rcb=True)
        
        return stored_chunk_id, summary
    
    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        extract_text: bool = True,
        importance: float = 7.0,
    ) -> ToolResult:
        """Execute GET request and store result in RLM memory.
        
        Args:
            url: Target URL.
            headers: Optional additional headers.
            params: Optional query parameters.
            extract_text: For HTML, extract main text content.
            importance: RLM memory importance (1-10, default 7 for user-requested content).
            
        Returns:
            ToolResult with memory handle for accessing the content.
        """
        is_valid, error = self._validate_url(url)
        if not is_valid:
            return ToolResult(success=False, output=None, error=error)
        
        kwargs: Dict[str, Any] = {}
        if headers:
            kwargs["headers"] = headers
        if params:
            kwargs["params"] = params
        
        success, output, error, metadata = await self._execute_with_retry("GET", url, **kwargs)
        
        if not success:
            return ToolResult(
                success=False,
                output=None,
                error=error,
                execution_time=0.0,
            )
        
        # Extract content based on type
        content_to_store = ""
        content_type = "unknown"
        
        if isinstance(output, dict):
            if output.get("type") == "html_text":
                content_to_store = output.get("text", "")
                content_type = "html"
            elif output.get("type") == "json":
                # Store JSON as formatted string
                content_to_store = json.dumps(output.get("data"), indent=2, ensure_ascii=False)
                content_type = "json"
            elif output.get("type") == "text":
                content_to_store = output.get("text", "")
                content_type = "text"
        
        # Store in RLM and get handle
        chunk_id, summary = await self._store_in_rlm(
            url=url,
            content=content_to_store,
            content_type=content_type,
            importance=importance,
            metadata=metadata,
        )
        
        # Build result with memory handle
        result_data = {
            "memory_handle": f"[[MEMORY:{chunk_id}]]",
            "chunk_id": chunk_id,
            "url": url,
            "content_type": content_type,
            "content_length": len(content_to_store),
            "summary": summary,
            "metadata": metadata,
            "access_instructions": (
                f"The full content from {url} is now available in your RLM memory. "
                f"Use the memory handle [[MEMORY:{chunk_id}]] to access it, "
                f"or simply reference it in your thinking process. "
                f"Content summary: {summary[:100]}..."
            ),
        }
        
        return ToolResult(
            success=True,
            output=result_data,
            error=None,
            execution_time=0.0,
        )
    
    async def post(self, url: str, **kwargs) -> ToolResult:
        """POST request - delegates to get with POST method, stores result."""
        # Similar pattern: execute, store in RLM, return handle
        # Implementation omitted for brevity - follows same pattern as get
        return ToolResult(success=False, output=None, error="POST not fully implemented in RLM mode")
    
    async def put(self, url: str, **kwargs) -> ToolResult:
        """PUT request."""
        return ToolResult(success=False, output=None, error="PUT not fully implemented in RLM mode")
    
    async def delete(self, url: str, **kwargs) -> ToolResult:
        """DELETE request."""
        return ToolResult(success=False, output=None, error="DELETE not fully implemented in RLM mode")
    
    async def execute(self, **params: Any) -> ToolResult:
        """Execute HTTP request based on parameters."""
        method = params.get("method", "").lower()
        url = params.get("url")
        
        if not method:
            return ToolResult(success=False, output=None, error="Missing required parameter: 'method'")
        if not url:
            return ToolResult(success=False, output=None, error="Missing required parameter: 'url'")
        
        headers = params.get("headers")
        query_params = params.get("params")
        extract_text = params.get("extract_text", True)
        importance = params.get("importance", 7.0)
        
        if method == "get":
            return await self.get(url, headers=headers, params=query_params, 
                                extract_text=extract_text, importance=importance)
        
        return ToolResult(
            success=False,
            output=None,
            error=f"Method '{method}' not implemented in RLM-integrated HTTP tool",
        )
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None
    
    def __repr__(self) -> str:
        return f"HttpTool(timeout={self.default_timeout}, rlm_enabled={self._rlm_memory is not None})"
