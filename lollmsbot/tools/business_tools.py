"""
Business Analysis Tools for SimplifiedAgant integration
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lollmsbot.agent import Tool, ToolResult


class BusinessReportTool(Tool):
    """Get AI Council business analysis report."""
    
    name: str = "business_report"
    description: str = (
        "Get the latest AI Council business analysis report. "
        "The council analyzes signals from all connected systems (YouTube, CRM, "
        "tasks, analytics) and provides strategic recommendations from multiple "
        "perspectives: Growth, Revenue, Operations, and Team Dynamics."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "run_analysis": {
                "type": "boolean",
                "description": "Run new analysis (true) or get latest (false)",
                "default": False,
            },
        },
    }
    
    risk_level: str = "low"
    
    def __init__(self, council: Optional[Any] = None):
        self._council = council
    
    async def execute(self, run_analysis: bool = False, **kwargs) -> ToolResult:
        if not self._council:
            return ToolResult(
                success=False,
                output=None,
                error="Business analysis council not initialized",
            )
        
        try:
            if run_analysis:
                # Run new analysis
                report = await self._council.run_council_analysis(period_days=1)
                formatted = await self._council.format_report_for_display(report)
            else:
                # Get latest
                report = await self._council.get_latest_report()
                if not report:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="No business reports available. Run with run_analysis=true first.",
                    )
                formatted = await self._council.format_report_for_display(report)
            
            return ToolResult(
                success=True,
                output={
                    "report": formatted,
                    "generated_at": report.generated_at.isoformat() if report else None,
                    "consensus": report.consensus_score if report else 0,
                    "confidence": report.confidence_level if report else "unknown",
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Business report failed: {str(e)}",
            )


__all__ = ["BusinessReportTool"]
