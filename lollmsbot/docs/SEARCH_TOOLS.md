# Internet Search Tools

LollmsBot includes comprehensive internet search capabilities with multiple providers.

## Quick Start

The simplest search requires **no configuration**:

```
<tool>quick_search</tool>
<query>python asyncio best practices</query>
```

This uses DuckDuckGo and works immediately.

## Available Providers

| Provider | API Key Required | Description |
|----------|---------------|-------------|
| **DuckDuckGo** | ❌ No | Always available, privacy-focused |
| **Google** | ✅ Yes | Custom Search JSON API |
| **Stack Overflow** | ❌ No | Programming Q&A search |
| **Wikipedia** | ❌ No | Encyclopedia search |
| **Twitter/X** | ✅ Yes | Social media search |
| **News API** | ✅ Yes | News article search |
| **Reddit** | ✅ Yes | Forum/community search |

## Configuration

Run the wizard to configure search:

```bash
lollmsbot wizard
# Select "🔍 Internet Search"
```

### Required Libraries

Install search dependencies:

```bash
# All providers
pip install duckduckgo-search google-api-python-client tweepy wikipedia newsapi-python praw aiohttp

# Minimal (DuckDuckGo only)
pip install duckduckgo-search
```

### API Keys

#### Google Search
1. Get API key: https://developers.google.com/custom-search/v1/overview
2. Create search engine: https://programmablesearchengine.google.com/
3. Enter both in wizard

#### Twitter/X
1. Apply at: https://developer.twitter.com/en/portal/dashboard
2. Get Bearer Token (Elevated access for search)

#### News API
1. Sign up: https://newsapi.org/
2. Copy API key

#### Reddit
1. Create app: https://www.reddit.com/prefs/apps
2. Get Client ID and Secret

## Tool Usage

### quick_search
Simple DuckDuckGo search (always works):

```xml
<tool>quick_search</tool>
<query>machine learning tutorials</query>
<results>5</results>
```

### internet_search
Full-featured search with provider selection:

```xml
<!-- Search all configured providers -->
<tool>internet_search</tool>
<query>latest AI developments 2024</query>
<provider>unified</provider>
<max_results>5</max_results>

<!-- Search specific provider -->
<tool>internet_search</tool>
<query>python decorators</query>
<provider>stackoverflow</provider>
<max_results>10</max_results>

<!-- Reddit search with subreddit -->
<tool>internet_search</tool>
<query>best python IDE</query>
<provider>reddit</provider>
<subreddit>python</subreddit>
```

## Caching

Search results are cached for 24 hours by default to:
- Reduce API usage
- Improve response times
- Avoid rate limits

Configure in wizard: **Cache TTL (hours)**

## Memory Integration

Search results are automatically stored in RLM memory with:
- Source URL preservation
- Importance scoring (6/10)
- Cross-reference hints

This means:
- "Remember when we searched for..." works
- Results appear in memory searches
- Context persists across sessions

## Provider Status Check

To see which providers are configured:

```xml
<tool>internet_search</tool>
<query>status</query>
<provider>unified</provider>
```

Or check `/health` endpoint on the gateway.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "duckduckgo-search not installed" | Run `pip install duckduckgo-search` |
| "Google client not initialized" | Configure API key in wizard |
| Rate limit errors | Increase cache TTL, wait between searches |
| No results | Try different query, check internet connection |

## Examples

### Research Task
```
User: Research the latest Python 3.12 features

Assistant: I'll search for the latest Python 3.12 features.

<tool>internet_search</tool>
<query>Python 3.12 new features changelog</query>
<provider>unified</provider>

[Results formatted and presented]
```

### Programming Help
```
User: How do I use asyncio.gather?

Assistant: Let me search Stack Overflow for examples.

<tool>internet_search</tool>
<query>python asyncio.gather examples</query>
<provider>stackoverflow</provider>

[Top answers with code examples]
```

### Current Events
```
User: What's happening in AI today?

Assistant: Searching recent news...

<tool>internet_search</tool>
<query>artificial intelligence news today</query>
<provider>news</provider>

[Latest news articles]
```

## Security Notes

- API keys stored in `~/.lollmsbot/config.json` (encrypted at rest if OS supports)
- Search queries logged for debugging (not content)
- No personal data sent to search providers except query
- DuckDuckGo provides privacy-respecting search without tracking
```

I've created a comprehensive search tools module for LollmsBot with the following features:

## Summary

**1. Search Tools Module (`lollmsbot/tools/search.py`)**:
- **7 search providers**: DuckDuckGo, Google, Stack Overflow, Wikipedia, Twitter/X, News API, Reddit
- **2 tool classes**: `quick_search` (simple, always works) and `internet_search` (full-featured)
- **Caching system**: File-based cache with configurable TTL to reduce API usage
- **Rate limiting**: Prevents hitting provider limits
- **RLM integration**: Results automatically stored in memory for future reference

**2. Wizard Integration**:
- New "🔍 Internet Search" menu option
- Step-by-step configuration for each provider
- Visual status indicators in the configuration tree
- Test functionality to verify setup
- API key masking for security

**3. Auto-registration**:
- Search tools auto-register on agent initialization
- Reads configuration from wizard's config.json
- Graceful fallback if libraries not installed
- Status reporting in health checks

## Key Features

- **DuckDuckGo**: Works immediately with zero configuration
- **Unified search**: Searches all configured providers at once
- **Provider-specific**: Can target specific sources (Stack Overflow for code, News for current events)
- **Memory-aware**: Results stored in RLM for "remember when we searched..." functionality
- **Configurable caching**: Reduces API costs and improves speed

## Usage Examples

```xml
<!-- Simple search (always works) -->
<tool>quick_search</tool>
<query>python asyncio</query>

<!-- Full internet search -->
<tool>internet_search</tool>
<query>machine learning news</query>
<provider>unified</provider>

<!-- Stack Overflow specifically -->
<tool>internet_search</tool>
<query>python decorators</query>
<provider>stackoverflow</provider>

<!-- Reddit with subreddit -->
<tool>internet_search</tool>
<query>best python IDE</query>
<provider>reddit</provider>
<subreddit>python</subreddit>
```