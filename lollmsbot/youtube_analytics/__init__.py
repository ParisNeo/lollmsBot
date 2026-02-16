"""
YouTube Analytics Module - Channel tracking and competitor analysis

Provides daily analytics ingestion, growth tracking, competitor monitoring,
and content performance insights with automated reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of YouTube metrics."""
    VIEWS = "views"
    LIKES = "likes"
    COMMENTS = "comments"
    SUBSCRIBERS = "subscribers"
    WATCH_TIME = "watch_time"
    CTR = "ctr"  # Click-through rate
    AVD = "avd"  # Average view duration


@dataclass
class VideoMetrics:
    """Metrics for a single video."""
    video_id: str
    title: str
    published_at: datetime
    thumbnail_url: Optional[str] = None
    
    # Current stats
    views: int = 0
    likes: int = 0
    comments: int = 0
    
    # Derived metrics
    ctr: Optional[float] = None  # Percentage
    avd_seconds: Optional[int] = None
    avd_percentage: Optional[float] = None  # Percentage of video length
    
    # Historical data (date -> metric value)
    views_history: Dict[str, int] = field(default_factory=dict)
    
    # Analysis
    performance_tier: str = "unknown"  # "viral", "high", "average", "low"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "published_at": self.published_at.isoformat(),
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "ctr": self.ctr,
            "avd_seconds": self.avd_seconds,
            "performance_tier": self.performance_tier,
        }


@dataclass
class ChannelSnapshot:
    """Snapshot of channel metrics at a point in time."""
    timestamp: datetime
    total_views: int
    total_subscribers: int
    total_videos: int
    estimated_revenue: Optional[float] = None
    
    # Growth rates (percentage)
    subscriber_growth_7d: Optional[float] = None
    view_growth_7d: Optional[float] = None
    
    # Top performing videos (last 28 days)
    top_videos: List[str] = field(default_factory=list)


@dataclass
class CompetitorChannel:
    """Tracked competitor channel."""
    channel_id: str
    channel_name: str
    tracked_since: datetime
    
    # Latest snapshot
    last_snapshot: Optional[ChannelSnapshot] = None
    
    # Upload cadence
    uploads_last_30d: int = 0
    avg_upload_frequency: Optional[str] = None  # "daily", "2-3x weekly", "weekly", etc.
    
    # Content themes
    common_topics: List[str] = field(default_factory=list)


class YouTubeAnalyticsManager:
    """
    Manages YouTube channel analytics and competitor tracking.
    
    Key features from SimplifiedAgant:
    - Daily API ingestion
    - Historical tracking with snapshots
    - Competitor monitoring (upload cadence, topics)
    - Performance tier classification
    - Automated insights and recommendations
    """
    
    def __init__(self, memory_manager: Any) -> None:
        self._memory = memory_manager
        self._api_key: Optional[str] = None
        self._channel_id: Optional[str] = None
        
        # Data storage
        self._videos: Dict[str, VideoMetrics] = {}  # video_id -> metrics
        self._channel_history: List[ChannelSnapshot] = []
        self._competitors: Dict[str, CompetitorChannel] = {}  # channel_id -> competitor
        
        self._initialized = False
    
    def set_api_credentials(self, api_key: str, channel_id: str) -> None:
        """Set YouTube Data API credentials."""
        self._api_key = api_key
        self._channel_id = channel_id
        logger.info(f"YouTube API configured for channel: {channel_id}")
    
    async def initialize(self) -> None:
        """Load existing analytics data from memory."""
        if self._initialized:
            return
        
        if self._memory:
            # Load video metrics
            results = await self._memory.search_ems(
                query="youtube_analytics video",
                chunk_types=[MemoryChunkType.FACT],
                limit=200,
            )
            
            for chunk, score in results:
                if "yt_video:" in (chunk.source or ""):
                    try:
                        data = json.loads(chunk.decompress_content())
                        video = self._dict_to_video(data)
                        self._videos[video.video_id] = video
                    except Exception as e:
                        logger.warning(f"Failed to load video metrics: {e}")
            
            # Load channel snapshots
            snapshot_results = await self._memory.search_ems(
                query="youtube_analytics channel_snapshot",
                limit=100,
            )
            
            for chunk, score in snapshot_results:
                if "yt_channel_snapshot:" in (chunk.source or ""):
                    try:
                        data = json.loads(chunk.decompress_content())
                        snapshot = self._dict_to_snapshot(data)
                        self._channel_history.append(snapshot)
                    except Exception as e:
                        logger.warning(f"Failed to load channel snapshot: {e}")
            
            # Sort by timestamp
            self._channel_history.sort(key=lambda x: x.timestamp)
        
        self._initialized = True
        logger.info(f"YouTube analytics initialized: {len(self._videos)} videos, {len(self._channel_history)} snapshots")
    
    def _dict_to_video(self, data: Dict[str, Any]) -> VideoMetrics:
        """Reconstruct video metrics from dictionary."""
        return VideoMetrics(
            video_id=data["video_id"],
            title=data["title"],
            published_at=datetime.fromisoformat(data["published_at"]),
            thumbnail_url=data.get("thumbnail_url"),
            views=data.get("views", 0),
            likes=data.get("likes", 0),
            comments=data.get("comments", 0),
            ctr=data.get("ctr"),
            avd_seconds=data.get("avd_seconds"),
            avd_percentage=data.get("avd_percentage"),
            views_history=data.get("views_history", {}),
            performance_tier=data.get("performance_tier", "unknown"),
            tags=data.get("tags", []),
        )
    
    def _dict_to_snapshot(self, data: Dict[str, Any]) -> ChannelSnapshot:
        """Reconstruct snapshot from dictionary."""
        return ChannelSnapshot(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            total_views=data.get("total_views", 0),
            total_subscribers=data.get("total_subscribers", 0),
            total_videos=data.get("total_videos", 0),
            estimated_revenue=data.get("estimated_revenue"),
            subscriber_growth_7d=data.get("subscriber_growth_7d"),
            view_growth_7d=data.get("view_growth_7d"),
            top_videos=data.get("top_videos", []),
        )
    
    async def ingest_daily_analytics(self, api_response: Optional[Dict[str, Any]] = None) -> ChannelSnapshot:
        """
        Ingest daily analytics from YouTube API.
        
        In production, this would call the YouTube Data API.
        For now, accepts mock data or uses placeholder.
        """
        # Create snapshot
        now = datetime.now()
        
        if api_response:
            # Parse real API response
            snapshot = ChannelSnapshot(
                timestamp=now,
                total_views=api_response.get("statistics", {}).get("viewCount", 0),
                total_subscribers=api_response.get("statistics", {}).get("subscriberCount", 0),
                total_videos=api_response.get("statistics", {}).get("videoCount", 0),
            )
            
            # Parse video data
            for video_data in api_response.get("videos", []):
                video_id = video_data["id"]
                
                metrics = VideoMetrics(
                    video_id=video_id,
                    title=video_data["snippet"]["title"],
                    published_at=datetime.fromisoformat(video_data["snippet"]["publishedAt"].replace("Z", "+00:00")),
                    thumbnail_url=video_data["snippet"]["thumbnails"]["high"]["url"] if "high" in video_data["snippet"]["thumbnails"] else None,
                    views=int(video_data["statistics"].get("viewCount", 0)),
                    likes=int(video_data["statistics"].get("likeCount", 0)),
                    comments=int(video_data["statistics"].get("commentCount", 0)),
                )
                
                # Calculate performance tier
                metrics.performance_tier = self._calculate_performance_tier(metrics)
                
                # Store
                self._videos[video_id] = metrics
                await self._persist_video(metrics)
        else:
            # Placeholder snapshot
            snapshot = ChannelSnapshot(
                timestamp=now,
                total_views=sum(v.views for v in self._videos.values()),
                total_subscribers=0,  # Would come from API
                total_videos=len(self._videos),
            )
        
        # Calculate growth rates
        if len(self._channel_history) >= 7:
            week_ago = self._channel_history[-7]
            if week_ago.total_subscribers > 0:
                snapshot.subscriber_growth_7d = (
                    (snapshot.total_subscribers - week_ago.total_subscribers) 
                    / week_ago.total_subscribers * 100
                )
            if week_ago.total_views > 0:
                snapshot.view_growth_7d = (
                    (snapshot.total_views - week_ago.total_views)
                    / week_ago.total_views * 100
                )
        
        # Get top videos (last 28 days)
        recent_videos = [
            v for v in self._videos.values()
            if (now - v.published_at).days <= 28
        ]
        recent_videos.sort(key=lambda x: -x.views)
        snapshot.top_videos = [v.video_id for v in recent_videos[:5]]
        
        # Store snapshot
        self._channel_history.append(snapshot)
        await self._persist_snapshot(snapshot)
        
        logger.info(f"Daily analytics ingested: {snapshot.total_views} total views, {len(self._videos)} videos")
        
        return snapshot
    
    def _calculate_performance_tier(self, video: VideoMetrics) -> str:
        """Classify video performance based on metrics."""
        # Get average views for recent videos
        recent = [
            v for v in self._videos.values()
            if (datetime.now() - v.published_at).days <= 30 and v.video_id != video.video_id
        ]
        
        if not recent:
            return "unknown"
        
        avg_views = sum(v.views for v in recent) / len(recent)
        
        # Classify
        ratio = video.views / avg_views if avg_views > 0 else 0
        
        if ratio > 5:
            return "viral"
        elif ratio > 2:
            return "high"
        elif ratio > 0.5:
            return "average"
        else:
            return "low"
    
    async def _persist_video(self, video: VideoMetrics) -> None:
        """Save video metrics to memory."""
        if not self._memory:
            return
        
        await self._memory.store_in_ems(
            content=json.dumps(video.to_dict()),
            chunk_type=MemoryChunkType.FACT,
            importance=2.0 if video.performance_tier in ["viral", "high"] else 1.5,
            tags=["youtube", "video", video.performance_tier, f"video_{video.video_id}"],
            summary=f"YouTube: {video.title[:60]}... ({video.views} views, {video.performance_tier})",
            load_hints=[video.title, video.video_id] + video.tags,
            source=f"yt_video:{video.video_id}",
        )
    
    async def _persist_snapshot(self, snapshot: ChannelSnapshot) -> None:
        """Save channel snapshot to memory."""
        if not self._memory:
            return
        
        await self._memory.store_in_ems(
            content=json.dumps({
                "timestamp": snapshot.timestamp.isoformat(),
                "total_views": snapshot.total_views,
                "total_subscribers": snapshot.total_subscribers,
                "total_videos": snapshot.total_videos,
                "estimated_revenue": snapshot.estimated_revenue,
                "subscriber_growth_7d": snapshot.subscriber_growth_7d,
                "view_growth_7d": snapshot.view_growth_7d,
                "top_videos": snapshot.top_videos,
            }),
            chunk_type=MemoryChunkType.FACT,
            importance=2.0,  # Channel snapshots are important
            tags=["youtube", "channel", "analytics", "snapshot"],
            summary=f"Channel snapshot {snapshot.timestamp.strftime('%Y-%m-%d')}: {snapshot.total_views} views",
            load_hints=["youtube", "analytics", "channel", "growth", "metrics"],
            source=f"yt_channel_snapshot:{snapshot.timestamp.isoformat()}",
        )
    
    async def add_competitor(self, channel_id: str, channel_name: str) -> CompetitorChannel:
        """Add a competitor channel to track."""
        competitor = CompetitorChannel(
            channel_id=channel_id,
            channel_name=channel_name,
            tracked_since=datetime.now(),
        )
        
        self._competitors[channel_id] = competitor
        
        # Would fetch initial data from API here
        
        logger.info(f"Added competitor: {channel_name} ({channel_id})")
        
        return competitor
    
    async def update_competitor_data(self, channel_id: str, 
                                     api_data: Optional[Dict[str, Any]] = None) -> CompetitorChannel:
        """Update competitor tracking data."""
        competitor = self._competitors.get(channel_id)
        if not competitor:
            logger.warning(f"Competitor not found: {channel_id}")
            return None
        
        if api_data:
            # Parse API data
            recent_uploads = api_data.get("recent_uploads", [])
            competitor.uploads_last_30d = len(recent_uploads)
            
            # Calculate frequency
            if competitor.uploads_last_30d >= 20:
                competitor.avg_upload_frequency = "daily"
            elif competitor.uploads_last_30d >= 8:
                competitor.avg_upload_frequency = "2-3x weekly"
            elif competitor.uploads_last_30d >= 4:
                competitor.avg_upload_frequency = "weekly"
            else:
                competitor.avg_upload_frequency = "sporadic"
            
            # Extract common topics from titles
            all_titles = " ".join([v.get("title", "") for v in recent_uploads])
            competitor.common_topics = self._extract_topics(all_titles)
            
            # Create snapshot
            competitor.last_snapshot = ChannelSnapshot(
                timestamp=datetime.now(),
                total_views=api_data.get("statistics", {}).get("viewCount", 0),
                total_subscribers=api_data.get("statistics", {}).get("subscriberCount", 0),
                total_videos=api_data.get("statistics", {}).get("videoCount", 0),
            )
        
        return competitor
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract common topics from text."""
        words = text.lower().split()
        word_freq = {}
        
        for w in words:
            if len(w) > 3 and w.isalpha():
                word_freq[w] = word_freq.get(w, 0) + 1
        
        # Return most common
        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:10]]
    
    async def generate_insights(self) -> Dict[str, Any]:
        """
        Generate content insights and recommendations.
        
        This is the key feature that feeds into business meta-analysis.
        """
        if not self._channel_history:
            return {"error": "No analytics data available"}
        
        latest = self._channel_history[-1]
        
        # Analyze performance patterns
        viral_videos = [v for v in self._videos.values() if v.performance_tier == "viral"]
        high_videos = [v for v in self._videos.values() if v.performance_tier == "high"]
        
        # Title/thumbnail analysis (would be more sophisticated in production)
        viral_themes = []
        for v in viral_videos[:3]:
            viral_themes.extend(v.tags)
        
        # Competitor insights
        competitor_activity = []
        for comp in self._competitors.values():
            if comp.last_snapshot:
                competitor_activity.append({
                    "name": comp.channel_name,
                    "upload_frequency": comp.avg_upload_frequency,
                    "recent_topics": comp.common_topics[:5],
                })
        
        return {
            "generated_at": datetime.now().isoformat(),
            "channel_health": {
                "total_views": latest.total_views,
                "total_subscribers": latest.total_subscribers,
                "growth_7d": {
                    "subscribers": latest.subscriber_growth_7d,
                    "views": latest.view_growth_7d,
                },
            },
            "content_performance": {
                "viral_videos": len(viral_videos),
                "high_performers": len(high_videos),
                "total_tracked": len(self._videos),
            },
            "viral_themes": list(set(viral_themes))[:10],
            "recommendations": self._generate_recommendations(viral_videos, high_videos),
            "competitor_activity": competitor_activity,
        }
    
    def _generate_recommendations(self, viral: List[VideoMetrics], 
                                   high: List[VideoMetrics]) -> List[str]:
        """Generate content recommendations based on performance data."""
        recommendations = []
        
        if viral:
            # Analyze what made videos viral
            common_elements = []
            for v in viral:
                if "tutorial" in v.title.lower() or "how to" in v.title.lower():
                    common_elements.append("tutorial format")
                if "vs" in v.title.lower() or "comparison" in v.title.lower():
                    common_elements.append("comparison format")
            
            if common_elements:
                recommendations.append(
                    f"Your viral videos often use: {', '.join(set(common_elements))}. "
                    "Consider creating more content in these formats."
                )
        
        # Upload frequency recommendation
        recent_uploads = len([
            v for v in self._videos.values()
            if (datetime.now() - v.published_at).days <= 30
        ])
        
        if recent_uploads < 4:
            recommendations.append(
                "Your upload frequency is low (less than weekly). "
                "Consider increasing to 2-3 videos per week for better algorithm favor."
            )
        elif recent_uploads > 20:
            recommendations.append(
                "You're uploading very frequently. Ensure quality doesn't suffer - "
                "consider A/B testing thumbnails more carefully."
            )
        
        # Competitor-based recommendations
        for comp in self._competitors.values():
            if comp.avg_upload_frequency == "daily" and recent_uploads < 8:
                recommendations.append(
                    f"Competitor '{comp.channel_name}' uploads daily. "
                    "Consider increasing your frequency to stay competitive."
                )
        
        return recommendations
    
    async def get_daily_report(self) -> str:
        """Generate daily analytics report (for cron/heartbeat)."""
        if not self._channel_history:
            return "No analytics data available yet."
        
        latest = self._channel_history[-1]
        insights = await self.generate_insights()
        
        lines = [
            "# 📊 YouTube Analytics Daily Report",
            "",
            f"**Date:** {latest.timestamp.strftime('%Y-%m-%d')}",
            f"**Channel Views:** {latest.total_views:,}",
            f"**Subscribers:** {latest.total_subscribers:,}",
            f"**Videos:** {latest.total_videos}",
            "",
        ]
        
        if latest.subscriber_growth_7d is not None:
            emoji = "📈" if latest.subscriber_growth_7d > 0 else "📉"
            lines.append(f"**7-Day Growth:** {emoji} {latest.subscriber_growth_7d:+.1f}% subscribers")
        
        if latest.view_growth_7d is not None:
            emoji = "📈" if latest.view_growth_7d > 0 else "📉"
            lines.append(f"**7-Day Growth:** {emoji} {latest.view_growth_7d:+.1f}% views")
        
        lines.extend([
            "",
            "## Top Performing Videos (Last 28 Days)",
            "",
        ])
        
        for vid_id in latest.top_videos[:5]:
            video = self._videos.get(vid_id)
            if video:
                lines.append(f"- **{video.title[:50]}...** - {video.views:,} views ({video.performance_tier})")
        
        lines.extend([
            "",
            "## Recommendations",
            "",
        ])
        
        for rec in insights.get("recommendations", [])[:3]:
            lines.append(f"- {rec}")
        
        return "\n".join(lines)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get analytics system statistics."""
        return {
            "videos_tracked": len(self._videos),
            "snapshots_stored": len(self._channel_history),
            "competitors_tracked": len(self._competitors),
            "date_range": {
                "first_snapshot": self._channel_history[0].timestamp.isoformat() if self._channel_history else None,
                "latest_snapshot": self._channel_history[-1].timestamp.isoformat() if self._channel_history else None,
            },
        }


# Import for type hints
from lollmsbot.agent.rlm.models import MemoryChunkType


__all__ = [
    "YouTubeAnalyticsManager",
    "VideoMetrics",
    "ChannelSnapshot",
    "CompetitorChannel",
    "MetricType",
]
