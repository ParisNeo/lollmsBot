"""
Search Tools - Internet search capabilities for LollmsBot.

Provides unified search interface with multiple backends:
- Google Search (via Custom Search JSON API)
- DuckDuckGo (via duckduckgo-search library)
- Stack Overflow (via Stack Exchange API)
- Twitter/X (via tweepy library)
- Wikipedia (via wikipedia library)
- News API (via newsapi-python)
- Reddit (via praw library)

All tools support caching and rate limiting.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    from googleapiclient.discovery import build as google_build
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

from lollmsbot.agent import Tool, ToolResult
from lollmsbot.agent.rlm import MemoryChunkType

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result."""
    title: str
    url: str
    snippet: str
    source: str  # Which search provider
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    score: float = 0.0


class SearchCache:
    """Simple file-based cache for search results."""
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        self.cache_dir = cache_dir or (Path.home() / ".lollmsbot" / "search_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, provider: str, query: str) -> str:
        """Generate cache file name."""
        import hashlib
        key = f"{provider}:{query.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
    
    def get(self, provider: str, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached results if not expired."""
        key = self._get_cache_key(provider, query)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            data = json.loads(cache_path.read_text())
            cached_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            
            if datetime.now() - cached_time > self.ttl:
                return None  # Expired
            
            return data.get("results")
        except Exception:
            return None
    
    def set(self, provider: str, query: str, results: List[Dict[str, Any]]) -> None:
        """Cache search results."""
        key = self._get_cache_key(provider, query)
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                "cached_at": datetime.now().isoformat(),
                "provider": provider,
                "query": query,
                "results": results,
            }
            cache_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to cache search: {e}")


class SearchManager:
    """
    Unified search manager with multiple backends.
    
    Handles API keys, rate limiting, and result aggregation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.cache = SearchCache()
        
        # API keys from config or env
        self.google_api_key = self.config.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
        self.google_cx = self.config.get("google_cx") or os.getenv("GOOGLE_CUSTOM_SEARCH_CX")
        
        self.twitter_bearer = self.config.get("twitter_bearer_token") or os.getenv("TWITTER_BEARER_TOKEN")
        self.twitter_api_key = self.config.get("twitter_api_key") or os.getenv("TWITTER_API_KEY")
        self.twitter_api_secret = self.config.get("twitter_api_secret") or os.getenv("TWITTER_API_SECRET")
        self.twitter_access_token = self.config.get("twitter_access_token") or os.getenv("TWITTER_ACCESS_TOKEN")
        self.twitter_access_secret = self.config.get("twitter_access_secret") or os.getenv("TWITTER_ACCESS_SECRET")
        
        self.news_api_key = self.config.get("news_api_key") or os.getenv("NEWSAPI_KEY")
        
        self.reddit_client_id = self.config.get("reddit_client_id") or os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = self.config.get("reddit_client_secret") or os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = self.config.get("reddit_user_agent") or os.getenv("REDDIT_USER_AGENT", "LollmsBot/1.0")
        
        # Rate limiting
        self._last_search: Dict[str, datetime] = {}
        self._min_interval = timedelta(seconds=1)  # Min 1 second between searches
        
        # Initialize clients
        self._init_clients()
    
    def _init_clients(self) -> None:
        """Initialize API clients."""
        self._google_client = None
        self._twitter_client = None
        self._newsapi_client = None
        self._reddit_client = None
        
        # Google
        if GOOGLE_AVAILABLE and self.google_api_key and self.google_cx:
            try:
                self._google_client = google_build("customsearch", "v1", developerKey=self.google_api_key)
                logger.info("✅ Google Search client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google client: {e}")
        
        # Twitter
        if TWEEPY_AVAILABLE and self.twitter_bearer:
            try:
                self._twitter_client = tweepy.Client(
                    bearer_token=self.twitter_bearer,
                    wait_on_rate_limit=True,
                )
                logger.info("✅ Twitter client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Twitter client: {e}")
        
        # News API
        if NEWSAPI_AVAILABLE and self.news_api_key:
            try:
                self._newsapi_client = NewsApiClient(api_key=self.news_api_key)
                logger.info("✅ NewsAPI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAPI client: {e}")
        
        # Reddit
        if PRAW_AVAILABLE and self.reddit_client_id and self.reddit_client_secret:
            try:
                self._reddit_client = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent,
                )
                logger.info("✅ Reddit client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit client: {e}")
    
    def _check_rate_limit(self, provider: str) -> bool:
        """Check if we can search (rate limiting)."""
        last = self._last_search.get(provider)
        if last and datetime.now() - last < self._min_interval:
            return False
        self._last_search[provider] = datetime.now()
        return True
    
    async def search_duckduckgo(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search using DuckDuckGo (no API key required).
        
        This is the most accessible search - works without configuration.
        """
        if not DDGS_AVAILABLE:
            logger.warning("duckduckgo-search library not installed")
            return []
        
        # Check cache
        cached = self.cache.get("duckduckgo", query)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        if not self._check_rate_limit("duckduckgo"):
            logger.warning("Rate limit hit for DuckDuckGo")
            return []
        
        results = []
        try:
            with DDGS() as ddgs:
                # Text search
                ddg_results = ddgs.text(query, max_results=max_results)
                
                for r in ddg_results:
                    result = SearchResult(
                        title=r.get("title", "No title"),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="duckduckgo",
                    )
                    results.append(result)
                
                # Also try news
                try:
                    news_results = ddgs.news(query, max_results=max(3, max_results // 3))
                    for r in news_results:
                        result = SearchResult(
                            title=r.get("title", "No title"),
                            url=r.get("url", ""),
                            snippet=r.get("body", ""),
                            source="duckduckgo_news",
                            published_date=r.get("date"),
                            author=r.get("source"),
                        )
                        results.append(result)
                except Exception as e:
                    logger.debug(f"DDG news search failed: {e}")
                
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        
        # Cache results
        if results:
            self.cache.set("duckduckgo", query, [r.__dict__ for r in results])
        
        return results
    
    async def search_google(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search using Google Custom Search API.
        
        Requires:
        - GOOGLE_API_KEY
        - GOOGLE_CUSTOM_SEARCH_CX (search engine ID)
        """
        if not GOOGLE_AVAILABLE or not self._google_client:
            logger.info("Google Search not configured - using DuckDuckGo fallback")
            return []
        
        # Check cache
        cached = self.cache.get("google", query)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        if not self._check_rate_limit("google"):
            return []
        
        results = []
        try:
            response = self._google_client.cse().list(
                q=query,
                cx=self.google_cx,
                num=min(max_results, 10),  # Google max is 10 per query
            ).execute()
            
            for item in response.get("items", []):
                result = SearchResult(
                    title=item.get("title", "No title"),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google",
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Google search failed: {e}")
        
        if results:
            self.cache.set("google", query, [r.__dict__ for r in results])
        
        return results
    
    async def search_stackoverflow(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search Stack Overflow using Stack Exchange API.
        
        No API key required for basic usage (rate limited).
        """
        # Check cache
        cached = self.cache.get("stackoverflow", query)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        if not self._check_rate_limit("stackoverflow"):
            return []
        
        results = []
        
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available for Stack Overflow search")
            return []
        
        try:
            # Stack Exchange API endpoint
            search_url = "https://api.stackexchange.com/2.3/search"
            params = {
                "order": "desc",
                "sort": "relevance",
                "intitle": query,
                "site": "stackoverflow",
                "pagesize": min(max_results, 30),
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get("items", []):
                            # Get question details
                            title = item.get("title", "No title")
                            # Clean HTML tags
                            title = re.sub(r'<[^>]+>', '', title)
                            
                            result = SearchResult(
                                title=title,
                                url=item.get("link", ""),
                                snippet=f"Score: {item.get('score', 0)}, Answers: {item.get('answer_count', 0)}, Views: {item.get('view_count', 0)}",
                                source="stackoverflow",
                                published_date=datetime.fromtimestamp(item.get("creation_date", 0)) if item.get("creation_date") else None,
                            )
                            results.append(result)
                    else:
                        logger.warning(f"Stack Overflow API returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Stack Overflow search failed: {e}")
        
        if results:
            self.cache.set("stackoverflow", query, [r.__dict__ for r in results])
        
        return results
    
    async def search_twitter(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search Twitter/X using tweepy.
        
        Requires Twitter API credentials.
        """
        if not TWEEPY_AVAILABLE or not self._twitter_client:
            logger.info("Twitter not configured")
            return []
        
        # Check cache
        cached = self.cache.get("twitter", query)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        if not self._check_rate_limit("twitter"):
            return []
        
        results = []
        try:
            # Recent search (requires Elevated access)
            tweets = tweepy.Paginator(
                self._twitter_client.search_recent_tweets,
                query=query,
                tweet_fields=["created_at", "author_id", "public_metrics"],
                max_results=min(max_results, 100),
            ).flatten(limit=max_results)
            
            async for tweet in tweets:
                result = SearchResult(
                    title=f"Tweet by @{tweet.author_id}",
                    url=f"https://twitter.com/i/web/status/{tweet.id}",
                    snippet=tweet.text[:280],
                    source="twitter",
                    published_date=tweet.created_at,
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Twitter search failed: {e}")
        
        if results:
            self.cache.set("twitter", query, [r.__dict__ for r in results])
        
        return results
    
    async def search_wikipedia(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search Wikipedia.
        
        No API key required.
        """
        if not WIKIPEDIA_AVAILABLE:
            logger.warning("wikipedia library not installed")
            return []
        
        # Check cache
        cached = self.cache.get("wikipedia", query)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        if not self._check_rate_limit("wikipedia"):
            return []
        
        results = []
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=max_results)
            
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    result = SearchResult(
                        title=page.title,
                        url=page.url,
                        snippet=page.summary[:500],
                        source="wikipedia",
                    )
                    results.append(result)
                except wikipedia.exceptions.DisambiguationError:
                    # Skip disambiguation pages
                    continue
                except wikipedia.exceptions.PageError:
                    continue
                    
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
        
        if results:
            self.cache.set("wikipedia", query, [r.__dict__ for r in results])
        
        return results
    
    async def search_news(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """
        Search news using NewsAPI.
        
        Requires NEWSAPI_KEY.
        """
        if not NEWSAPI_AVAILABLE or not self._newsapi_client:
            logger.info("NewsAPI not configured")
            return []
        
        # Check cache
        cached = self.cache.get("newsapi", query)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        if not self._check_rate_limit("newsapi"):
            return []
        
        results = []
        try:
            response = self._newsapi_client.get_everything(
                q=query,
                language="en",
                sort_by="relevancy",
                page_size=min(max_results, 100),
            )
            
            for article in response.get("articles", []):
                result = SearchResult(
                    title=article.get("title", "No title"),
                    url=article.get("url", ""),
                    snippet=article.get("description", ""),
                    source=f"newsapi:{article.get('source', {}).get('name', 'unknown')}",
                    published_date=datetime.fromisoformat(article.get("publishedAt", "").replace("Z", "+00:00")) if article.get("publishedAt") else None,
                    author=article.get("author"),
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
        
        if results:
            self.cache.set("newsapi", query, [r.__dict__ for r in results])
        
        return results
    
    async def search_reddit(self, query: str, max_results: int = 10, subreddit: Optional[str] = None) -> List[SearchResult]:
        """
        Search Reddit.
        
        Requires Reddit API credentials.
        """
        if not PRAW_AVAILABLE or not self._reddit_client:
            logger.info("Reddit not configured")
            return []
        
        # Check cache
        cache_key = f"reddit:{subreddit or 'all'}:{query}"
        cached = self.cache.get("reddit", cache_key)
        if cached:
            return [SearchResult(**r) for r in cached]
        
        if not self._check_rate_limit("reddit"):
            return []
        
        results = []
        try:
            if subreddit:
                # Search within specific subreddit
                sub = self._reddit_client.subreddit(subreddit)
                posts = sub.search(query, limit=max_results, sort="relevance")
            else:
                # Search all of Reddit
                posts = self._reddit_client.subreddit("all").search(query, limit=max_results, sort="relevance")
            
            for post in posts:
                result = SearchResult(
                    title=post.title,
                    url=f"https://reddit.com{post.permalink}",
                    snippet=post.selftext[:500] if post.selftext else f"Score: {post.score}, Comments: {post.num_comments}",
                    source=f"reddit:r/{post.subreddit.display_name}",
                    published_date=datetime.utcfromtimestamp(post.created_utc),
                    author=str(post.author),
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
        
        if results:
            self.cache.set("reddit", cache_key, [r.__dict__ for r in results])
        
        return results
    
    async def unified_search(
        self,
        query: str,
        providers: Optional[List[str]] = None,
        max_results_per_provider: int = 5,
    ) -> Dict[str, List[SearchResult]]:
        """
        Search across multiple providers and return aggregated results.
        
        Args:
            query: Search query
            providers: List of providers to use, or None for all available
            max_results_per_provider: Max results per provider
        
        Returns:
            Dict mapping provider name to list of results
        """
        available_providers = {
            "duckduckgo": self.search_duckduckgo,
            "google": self.search_google,
            "stackoverflow": self.search_stackoverflow,
            "wikipedia": self.search_wikipedia,
            "twitter": self.search_twitter,
            "news": self.search_news,
            "reddit": self.search_reddit,
        }
        
        if providers is None:
            # Use all available (configured) providers
            providers = ["duckduckgo"]  # Always available
            if self._google_client:
                providers.append("google")
            if self._twitter_client:
                providers.append("twitter")
            if self._newsapi_client:
                providers.append("news")
            if self._reddit_client:
                providers.append("reddit")
            providers.extend(["stackoverflow", "wikipedia"])  # No API key needed
        
        results = {}
        
        for provider_name in providers:
            if provider_name in available_providers:
                try:
                    search_func = available_providers[provider_name]
                    provider_results = await search_func(query, max_results_per_provider)
                    if provider_results:
                        results[provider_name] = provider_results
                        logger.info(f"✅ {provider_name}: {len(provider_results)} results")
                except Exception as e:
                    logger.error(f"{provider_name} search failed: {e}")
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of search providers."""
        return {
            "duckduckgo": DDGS_AVAILABLE,
            "google": self._google_client is not None,
            "stackoverflow": AIOHTTP_AVAILABLE,
            "wikipedia": WIKIPEDIA_AVAILABLE,
            "twitter": self._twitter_client is not None,
            "newsapi": self._newsapi_client is not None,
            "reddit": self._reddit_client is not None,
        }


class InternetSearchTool(Tool):
    """
    Unified internet search tool.
    
    Searches across multiple providers and returns formatted results.
    Can search specific providers or use unified search.
    """
    
    name: str = "internet_search"
    description: str = (
        "Search the internet for information. Supports multiple search providers: "
        "DuckDuckGo (always available), Google, Stack Overflow, Wikipedia, Twitter/X, "
        "News API, and Reddit. Use 'unified' provider to search all configured sources."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "provider": {
                "type": "string",
                "enum": ["unified", "duckduckgo", "google", "stackoverflow", "wikipedia", "twitter", "news", "reddit"],
                "description": "Search provider to use. 'unified' searches all configured providers.",
                "default": "unified",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (per provider for unified)",
                "default": 5,
            },
            "subreddit": {
                "type": "string",
                "description": "For Reddit search: specific subreddit to search (e.g., 'python')",
            },
        },
        "required": ["query"],
    }
    
    risk_level: str = "low"
    
    def __init__(self, search_manager: Optional[SearchManager] = None):
        self._search_manager = search_manager
    
    def set_search_manager(self, manager: SearchManager) -> None:
        self._search_manager = manager
    
    async def execute(
        self,
        query: str,
        provider: str = "unified",
        max_results: int = 5,
        subreddit: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """Execute search."""
        if self._search_manager is None:
            return ToolResult(
                success=False,
                output=None,
                error="Search manager not initialized",
            )
        
        try:
            if provider == "unified":
                results = await self._search_manager.unified_search(
                    query=query,
                    max_results_per_provider=max_results,
                )
                
                # Format results
                formatted = self._format_unified_results(results)
                
                # Store in memory for future reference
                await self._store_in_memory(query, results)
                
                return ToolResult(
                    success=True,
                    output={
                        "query": query,
                        "providers_searched": list(results.keys()),
                        "total_results": sum(len(r) for r in results.values()),
                        "results": formatted,
                    },
                )
            
            elif provider == "duckduckgo":
                results = await self._search_manager.search_duckduckgo(query, max_results)
            
            elif provider == "google":
                results = await self._search_manager.search_google(query, max_results)
            
            elif provider == "stackoverflow":
                results = await self._search_manager.search_stackoverflow(query, max_results)
            
            elif provider == "wikipedia":
                results = await self._search_manager.search_wikipedia(query, max_results)
            
            elif provider == "twitter":
                results = await self._search_manager.search_twitter(query, max_results)
            
            elif provider == "news":
                results = await self._search_manager.search_news(query, max_results)
            
            elif provider == "reddit":
                results = await self._search_manager.search_reddit(query, max_results, subreddit)
            
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown provider: {provider}",
                )
            
            # Store in memory
            await self._store_in_memory(query, {provider: results})
            
            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "provider": provider,
                    "results": [r.__dict__ for r in results],
                    "formatted": self._format_results(results),
                },
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ToolResult(
                success=False,
                output=None,
                error=f"Search failed: {str(e)}",
            )
    
    def _format_unified_results(self, results: Dict[str, List[SearchResult]]) -> str:
        """Format unified search results for display."""
        lines = [
            "",
            "=" * 60,
            "🔍 INTERNET SEARCH RESULTS",
            "=" * 60,
            "",
        ]
        
        for provider, provider_results in results.items():
            if not provider_results:
                continue
            
            lines.append(f"📌 {provider.upper()} ({len(provider_results)} results)")
            lines.append("-" * 40)
            
            for i, r in enumerate(provider_results[:5], 1):
                lines.append(f"  {i}. {r.title}")
                lines.append(f"     URL: {r.url}")
                if r.snippet:
                    snippet = r.snippet[:150] + "..." if len(r.snippet) > 150 else r.snippet
                    lines.append(f"     {snippet}")
                lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def _format_results(self, results: List[SearchResult]) -> str:
        """Format single provider results."""
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.title}")
            lines.append(f"   {r.url}")
            if r.snippet:
                lines.append(f"   {r.snippet[:200]}")
            lines.append("")
        return "\n".join(lines)
    
    async def _store_in_memory(self, query: str, results: Dict[str, List[SearchResult]]) -> None:
        """Store search results in RLM memory."""
        try:
            # Get agent's memory manager
            if hasattr(self, '_agent') and self._agent and self._agent._memory:
                memory = self._agent._memory
                
                # Format for storage
                result_text = f"Search query: {query}\n\n"
                for provider, provider_results in results.items():
                    result_text += f"=== {provider.upper()} ===\n"
                    for r in provider_results[:3]:  # Store top 3 per provider
                        result_text += f"- {r.title}: {r.url}\n"
                    result_text += "\n"
                
                await memory.store_in_ems(
                    content=result_text,
                    chunk_type=MemoryChunkType.WEB_CONTENT,
                    importance=6.0,  # Medium-high importance for search results
                    tags=["search", "internet", "web"],
                    summary=f"Search: '{query}' ({sum(len(r) for r in results.values())} results)",
                    load_hints=[query, "search", "web"],
                    source=f"search_tool:{query[:50]}",
                )
        except Exception as e:
            logger.debug(f"Failed to store search in memory: {e}")


class QuickSearchTool(Tool):
    """
    Quick search tool - always uses DuckDuckGo, no configuration needed.
    
    This is the "just works" search tool that requires no setup.
    """
    
    name: str = "quick_search"
    description: str = (
        "Quick internet search using DuckDuckGo. No API key required. "
        "Returns top results with titles, URLs, and snippets."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for",
            },
            "results": {
                "type": "integer",
                "description": "Number of results (1-10)",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    
    risk_level: str = "low"
    
    def __init__(self, search_manager: Optional[SearchManager] = None):
        self._search_manager = search_manager
    
    def set_search_manager(self, manager: SearchManager) -> None:
        self._search_manager = manager
    
    async def execute(self, query: str, results: int = 5, **kwargs) -> ToolResult:
        """Execute quick search."""
        if self._search_manager is None:
            return ToolResult(
                success=False,
                output=None,
                error="Search manager not initialized",
            )
        
        try:
            # Clamp results
            results = max(1, min(results, 10))
            
            search_results = await self._search_manager.search_duckduckgo(query, results)
            
            if not search_results:
                return ToolResult(
                    success=True,
                    output={
                        "query": query,
                        "results": [],
                        "message": "No results found",
                    },
                )
            
            # Format nicely
            formatted = []
            for r in search_results:
                formatted.append({
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                })
            
            # Store in memory
            await self._store_in_memory(query, search_results)
            
            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "count": len(formatted),
                    "results": formatted,
                },
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Quick search failed: {str(e)}",
            )
    
    async def _store_in_memory(self, query: str, results: List[SearchResult]) -> None:
        """Store in RLM memory."""
        try:
            if hasattr(self, '_agent') and self._agent and self._agent._memory:
                memory = self._agent._memory
                
                content = f"Quick search: {query}\n\n"
                for r in results[:3]:
                    content += f"- {r.title}: {r.url}\n"
                
                await memory.store_in_ems(
                    content=content,
                    chunk_type=MemoryChunkType.WEB_CONTENT,
                    importance=5.0,
                    tags=["search", "quick"],
                    summary=f"Quick search: {query}",
                    source="quick_search",
                )
        except Exception:
            pass


def get_search_tools(config: Optional[Dict[str, Any]] = None) -> List[Tool]:
    """
    Factory function to create search tools.
    
    Returns list of configured search tools.
    """
    search_manager = SearchManager(config)
    
    # Always return quick_search (works without config)
    tools: List[Tool] = [QuickSearchTool(search_manager)]
    
    # Add full internet_search if any providers are configured
    if any([
        search_manager._google_client,
        search_manager._twitter_client,
        search_manager._newsapi_client,
        search_manager._reddit_client,
        DDGS_AVAILABLE,  # DuckDuckGo always works
    ]):
        tools.append(InternetSearchTool(search_manager))
    
    return tools
```