"""
YouTube Analytics Tools for SimplifiedAgant integration
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lollmsbot.agent import Tool, ToolResult


class YouTubeReportTool(Tool):
    """Get YouTube analytics report."""
    
    name: str = "youtube_report"
    description: str = (
        "Get comprehensive YouTube analytics report including channel metrics, "
        "video performance tiers, growth trends, and content recommendations. "
        "Also tracks competitor activity if configured."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "report_type": {
                "type": "string",
                "enum": ["daily", "insights", "competitors", "full"],
                "default": "daily",
            },
        },
    }
    
    risk_level: str = "low"
    
    def __init__(self, yt_manager: Optional[Any] = None):
        self._yt = yt_manager
    
    async def execute(self, report_type: str = "daily", **kwargs) -> ToolResult:
        if not self._yt:
            return ToolResult(
                success=False,
                output=None,
                error="YouTube analytics not initialized",
            )
        
        try:
            if report_type == "daily":
                report = await self._yt.get_daily_report()
                return ToolResult(
                    success=True,
                    output={
                        "report_type": "daily",
                        "report": report,
                    },
                )
            
            elif report_type == "insights":
                insights = await self._yt.generate_insights()
                return ToolResult(
                    success=True,
                    output={
                        "report_type": "insights",
                        "insights": insights,
                    },
                )
            
            elif report_type == "competitors":
                competitors = []
                for comp in self._yt._competitors.values():
                    competitors.append({
                        "name": comp.channel_name,
                        "upload_frequency": comp.avg_upload_frequency,
                        "topics": comp.common_topics[:5],
                    })
                
                return ToolResult(
                    success=True,
                    output={
                        "report_type": "competitors",
                        "competitors": competitors,
                    },
                )
            
            else:  # full
                daily = await self._yt.get_daily_report()
                insights = await self._yt.generate_insights()
                
                return ToolResult(
                    success=True,
                    output={
                        "report_type": "full",
                        "daily_report": daily,
                        "insights": insights,
                    },
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"YouTube report failed: {str(e)}",
            )


__all__ = ["YouTubeReportTool"]
