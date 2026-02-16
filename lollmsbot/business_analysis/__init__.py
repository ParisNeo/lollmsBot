"""
Business Meta-Analysis Module - AI Council for strategic insights

Provides multi-agent analysis of business signals, generating
daily reports with actionable recommendations from multiple perspectives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class AnalystRole(Enum):
    """Different analyst perspectives in the council."""
    GROWTH_STRATEGIST = "growth_strategist"
    REVENUE_GUARDIAN = "revenue_guardian"
    SKEPTICAL_OPERATOR = "skeptical_operator"
    TEAM_DYNAMICS_ARCHITECT = "team_dynamics_architect"
    COUNCIL_MODERATOR = "council_moderator"


@dataclass
class BusinessSignal:
    """A single business signal for analysis."""
    signal_id: str
    source: str  # "youtube", "crm", "slack", "email", "calendar", "hubspot", etc.
    signal_type: str  # "metric", "event", "alert", "opportunity", "risk"
    timestamp: datetime
    content: str
    confidence: float  # 0-10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalystPerspective:
    """Analysis from a single council member."""
    analyst: AnalystRole
    key_findings: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class CouncilReport:
    """Final synthesized report from the AI council."""
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    
    # Raw signals processed
    signals_processed: int
    top_signals: List[BusinessSignal] = field(default_factory=list)
    
    # Individual perspectives
    perspectives: List[AnalystPerspective] = field(default_factory=list)
    
    # Synthesized output
    executive_summary: str = ""
    strategic_priorities: List[Dict[str, Any]] = field(default_factory=list)
    immediate_actions: List[str] = field(default_factory=list)
    risks_identified: List[str] = field(default_factory=list)
    opportunities_identified: List[str] = field(default_factory=list)
    
    # Consensus metrics
    consensus_score: float = 0.0  # How much agreement among analysts
    confidence_level: str = "medium"  # "low", "medium", "high"


class BusinessAnalysisCouncil:
    """
    AI Council for business meta-analysis.
    
    Replicates the SimplifiedAgant feature where multiple AI agents
    analyze business signals and collaborate on recommendations.
    """
    
    # Analysis prompts for each role
    ROLE_PROMPTS: Dict[AnalystRole, str] = {
        AnalystRole.GROWTH_STRATEGIST: """
You are a Growth Strategist. Your focus is on identifying opportunities for 
business expansion, audience growth, and strategic partnerships. Look for:
- Untapped market opportunities
- Content themes with high potential
- Partnership and collaboration possibilities
- Growth bottlenecks and how to overcome them

Be optimistic but realistic. Prioritize sustainable growth over quick wins.
""",
        
        AnalystRole.REVENUE_GUARDIAN: """
You are a Revenue Guardian. Your focus is on financial sustainability and 
monetization. Look for:
- Revenue risks and opportunities
- Sponsor/client relationship health
- Pricing and value optimization
- Diversification needs

Be conservative but not pessimistic. Protect existing revenue while seeking new streams.
""",
        
        AnalystRole.SKEPTICAL_OPERATOR: """
You are a Skeptical Operator. Your focus is on operational efficiency and 
risk mitigation. Look for:
- Process inefficiencies and bottlenecks
- Operational risks and failure points
- Resource allocation issues
- Quality control concerns

Be critical and questioning. Challenge assumptions and demand evidence.
""",
        
        AnalystRole.TEAM_DYNAMICS_ARCHITECT: """
You are a Team Dynamics Architect. Your focus is on collaboration, 
communication, and team effectiveness. Look for:
- Communication gaps or silos
- Collaboration opportunities
- Meeting effectiveness
- Information flow issues

Be observant of human dynamics. Suggest improvements to how people work together.
""",
        
        AnalystRole.COUNCIL_MODERATOR: """
You are the Council Moderator. Your role is to synthesize the perspectives
of all analysts, resolve disagreements, and produce actionable final recommendations.

Look for:
- Areas of consensus among analysts
- Important disagreements that need resolution
- Conflicting recommendations that need prioritization
- Gaps in analysis that need addressing

Produce a clear, prioritized action plan with specific next steps.
""",
    }
    
    def __init__(self, memory_manager: Any, llm_client: Optional[Any] = None) -> None:
        self._memory = memory_manager
        self._llm = llm_client
        
        # Signal collection
        self._recent_signals: List[BusinessSignal] = []
        self._max_signals: int = 200  # Keep top 200 by confidence
        
        # Analysis history
        self._report_history: List[CouncilReport] = []
        self._max_history: int = 30  # 30 days of reports
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Load previous reports from memory."""
        if self._initialized:
            return
        
        if self._memory:
            results = await self._memory.search_ems(
                query="business_analysis council_report",
                chunk_types=[MemoryChunkType.FACT],
                limit=30,
            )
            
            for chunk, score in results:
                if "council_report:" in (chunk.source or ""):
                    try:
                        data = json.loads(chunk.decompress_content())
                        report = self._dict_to_report(data)
                        self._report_history.append(report)
                    except Exception as e:
                        logger.warning(f"Failed to load council report: {e}")
            
            # Sort by date
            self._report_history.sort(key=lambda x: x.generated_at)
        
        self._initialized = True
        logger.info(f"Business analysis council initialized with {len(self._report_history)} historical reports")
    
    def _dict_to_report(self, data: Dict[str, Any]) -> CouncilReport:
        """Reconstruct report from dictionary."""
        return CouncilReport(
            generated_at=datetime.fromisoformat(data["generated_at"]),
            period_start=datetime.fromisoformat(data["period_start"]),
            period_end=datetime.fromisoformat(data["period_end"]),
            signals_processed=data.get("signals_processed", 0),
            executive_summary=data.get("executive_summary", ""),
            strategic_priorities=data.get("strategic_priorities", []),
            immediate_actions=data.get("immediate_actions", []),
            risks_identified=data.get("risks_identified", []),
            opportunities_identified=data.get("opportunities_identified", []),
            consensus_score=data.get("consensus_score", 0),
            confidence_level=data.get("confidence_level", "medium"),
        )
    
    async def ingest_signal(self, source: str, signal_type: str, 
                           content: str, confidence: float = 5.0,
                           metadata: Optional[Dict[str, Any]] = None) -> BusinessSignal:
        """
        Ingest a business signal from any source.
        
        Sources: youtube, crm, slack, email, calendar, hubspot, 
                fathom, github, analytics, etc.
        """
        signal = BusinessSignal(
            signal_id=f"sig_{datetime.now().isoformat()}_{source}",
            source=source,
            signal_type=signal_type,
            timestamp=datetime.now(),
            content=content,
            confidence=confidence,
            metadata=metadata or {},
        )
        
        self._recent_signals.append(signal)
        
        # Keep only highest confidence signals
        self._recent_signals.sort(key=lambda x: -x.confidence)
        self._recent_signals = self._recent_signals[:self._max_signals]
        
        logger.debug(f"Business signal ingested: {source}/{signal_type} (confidence: {confidence})")
        
        return signal
    
    async def run_council_analysis(self, 
                                   period_days: int = 1,
                                   force_llm: bool = False) -> CouncilReport:
        """
        Run the full AI council analysis.
        
        This is the core feature - multiple AI perspectives analyzing
        the same data and collaborating on recommendations.
        """
        # Determine analysis period
        period_end = datetime.now()
        period_start = period_end - timedelta(days=period_days)
        
        # Filter signals for period
        period_signals = [
            s for s in self._recent_signals
            if period_start <= s.timestamp <= period_end
        ]
        
        # Compact to top 200 by confidence
        period_signals.sort(key=lambda x: -x.confidence)
        top_signals = period_signals[:200]
        
        logger.info(f"Council analysis: {len(top_signals)} signals from last {period_days} days")
        
        # In production, this would call LLM with each role prompt
        # For now, generate simulated perspectives based on signal patterns
        
        perspectives = []
        
        # Growth Strategist perspective
        growth = await self._analyze_as_role(
            AnalystRole.GROWTH_STRATEGIST,
            top_signals,
            force_llm
        )
        perspectives.append(growth)
        
        # Revenue Guardian perspective
        revenue = await self._analyze_as_role(
            AnalystRole.REVENUE_GUARDIAN,
            top_signals,
            force_llm
        )
        perspectives.append(revenue)
        
        # Skeptical Operator perspective
        operator = await self._analyze_as_role(
            AnalystRole.SKEPTICAL_OPERATOR,
            top_signals,
            force_llm
        )
        perspectives.append(operator)
        
        # Team Dynamics perspective
        team = await self._analyze_as_role(
            AnalystRole.TEAM_DYNAMICS_ARCHITECT,
            top_signals,
            force_llm
        )
        perspectives.append(team)
        
        # Moderator synthesis
        moderator = await self._synthesize_as_moderator(perspectives, top_signals, force_llm)
        
        # Calculate consensus
        consensus = self._calculate_consensus(perspectives)
        
        # Create final report
        report = CouncilReport(
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            signals_processed=len(top_signals),
            top_signals=top_signals[:20],  # Include top 20 in report
            perspectives=perspectives,
            executive_summary=moderator.key_findings[0] if moderator.key_findings else "Analysis complete",
            strategic_priorities=moderator.recommendations,
            immediate_actions=moderator.opportunities[:5],
            risks_identified=moderator.concerns,
            opportunities_identified=moderator.opportunities,
            consensus_score=consensus,
            confidence_level="high" if consensus > 0.7 else "medium" if consensus > 0.4 else "low",
        )
        
        # Store report
        self._report_history.append(report)
        if len(self._report_history) > self._max_history:
            self._report_history = self._report_history[-self._max_history:]
        
        await self._persist_report(report)
        
        logger.info(f"Council report generated: {len(report.strategic_priorities)} priorities, consensus: {consensus:.2f}")
        
        return report
    
    async def _analyze_as_role(self, role: AnalystRole, 
                                signals: List[BusinessSignal],
                                use_llm: bool) -> AnalystPerspective:
        """
        Generate analysis from a specific role's perspective.
        
        In production, this calls an LLM with the role prompt.
        For now, uses rule-based analysis.
        """
        perspective = AnalystPerspective(analyst=role)
        
        # Filter signals relevant to this role
        role_signals = self._filter_signals_for_role(role, signals)
        
        if use_llm and self._llm:
            # Would call LLM here
            pass
        
        # Rule-based analysis (simulated)
        if role == AnalystRole.GROWTH_STRATEGIST:
            # Look for growth signals
            youtube_signals = [s for s in role_signals if s.source == "youtube"]
            if youtube_signals:
                top_video = max(youtube_signals, key=lambda x: x.metadata.get("views", 0))
                perspective.opportunities.append(
                    f"Video '{top_video.metadata.get('title', 'Unknown')}' performing well - "
                    f"consider similar content themes"
                )
            
            # Check for viral potential
            viral_signals = [s for s in role_signals if "viral" in s.content.lower()]
            if viral_signals:
                perspective.key_findings.append(
                    f"Identified {len(viral_signals)} content pieces with viral potential"
                )
            
            perspective.recommendations.append({
                "priority": "high",
                "action": "Double down on top-performing content themes",
                "expected_impact": "15-20% view growth",
            })
            
        elif role == AnalystRole.REVENUE_GUARDIAN:
            # Look for revenue signals
            sponsor_signals = [s for s in role_signals if s.source == "hubspot"]
            if sponsor_signals:
                perspective.key_findings.append(
                    f"Active sponsor pipeline: {len(sponsor_signals)} deals in progress"
                )
            
            # Check for risks
            risk_signals = [s for s in role_signals if s.signal_type == "risk"]
            if risk_signals:
                perspective.concerns.extend([s.content for s in risk_signals[:3]])
            
            perspective.recommendations.append({
                "priority": "medium",
                "action": "Diversify sponsor portfolio - reduce single-client dependency",
                "expected_impact": "Risk mitigation",
            })
            
        elif role == AnalystRole.SKEPTICAL_OPERATOR:
            # Look for operational issues
            error_signals = [s for s in role_signals if "error" in s.content.lower() or "failed" in s.content.lower()]
            if error_signals:
                perspective.concerns.append(
                    f"Detected {len(error_signals)} operational issues requiring attention"
                )
            
            # Check resource utilization
            cron_signals = [s for s in role_signals if s.source == "cron"]
            failed_crons = [s for s in cron_signals if "failed" in s.content.lower()]
            if failed_crons:
                perspective.concerns.append(
                    f"{len(failed_crons)} scheduled tasks failed - review automation health"
                )
            
            perspective.recommendations.append({
                "priority": "high",
                "action": "Implement monitoring alerts for failed automations",
                "expected_impact": "Prevent silent failures",
            })
            
        elif role == AnalystRole.TEAM_DYNAMICS_ARCHITECT:
            # Look for communication patterns
            slack_signals = [s for s in role_signals if s.source == "slack"]
            if len(slack_signals) > 50:
                perspective.key_findings.append(
                    "High Slack activity - team communication healthy"
                )
            
            # Check for meeting load
            meeting_signals = [s for s in role_signals if s.source == "calendar"]
            if len(meeting_signals) > 20:
                perspective.opportunities.append(
                    "Consider async alternatives to reduce meeting load"
                )
            
            perspective.recommendations.append({
                "priority": "low",
                "action": "Schedule team retrospective to align on priorities",
                "expected_impact": "Improved alignment",
            })
        
        # Calculate confidence based on signal quality
        perspective.confidence_score = sum(s.confidence for s in role_signals) / max(len(role_signals), 1)
        
        return perspective
    
    def _filter_signals_for_role(self, role: AnalystRole, 
                                  signals: List[BusinessSignal]) -> List[BusinessSignal]:
        """Filter signals most relevant to a specific role."""
        role_source_weights = {
            AnalystRole.GROWTH_STRATEGIST: {
                "youtube": 2.0, "analytics": 1.5, "twitter": 1.5, "competitor": 1.5,
            },
            AnalystRole.REVENUE_GUARDIAN: {
                "hubspot": 2.0, "crm": 1.5, "sponsor": 2.0, "analytics": 1.0,
            },
            AnalystRole.SKEPTICAL_OPERATOR: {
                "cron": 2.0, "error": 2.0, "github": 1.5, "system": 1.5,
            },
            AnalystRole.TEAM_DYNAMICS_ARCHITECT: {
                "slack": 2.0, "calendar": 1.5, "meeting": 1.5, "fathom": 1.5,
            },
        }
        
        weights = role_source_weights.get(role, {})
        
        # Score and filter signals
        scored = []
        for signal in signals:
            weight = weights.get(signal.source, 1.0)
            # Boost for high-confidence signals
            effective_score = signal.confidence * weight
            scored.append((signal, effective_score))
        
        # Sort by weighted score and return top 50
        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:50]]
    
    async def _synthesize_as_moderator(self, perspectives: List[AnalystPerspective],
                                        signals: List[BusinessSignal],
                                        use_llm: bool) -> AnalystPerspective:
        """Synthesize all perspectives into unified recommendations."""
        moderator = AnalystPerspective(analyst=AnalystRole.COUNCIL_MODERATOR)
        
        # Collect all findings
        all_findings = []
        all_concerns = []
        all_opportunities = []
        all_recommendations = []
        
        for p in perspectives:
            all_findings.extend(p.key_findings)
            all_concerns.extend(p.concerns)
            all_opportunities.extend(p.opportunities)
            all_recommendations.extend(p.recommendations)
        
        # Deduplicate and prioritize
        moderator.key_findings = list(set(all_findings))[:10]
        moderator.concerns = list(set(all_concerns))[:5]
        moderator.opportunities = list(set(all_opportunities))[:5]
        
        # Prioritize recommendations by frequency and priority
        priority_scores = {"high": 3, "medium": 2, "low": 1}
        
        # Count occurrences and score
        rec_scores = {}
        for rec in all_recommendations:
            action = rec.get("action", "")
            if action not in rec_scores:
                rec_scores[action] = {
                    "count": 0,
                    "priority_score": 0,
                    "original": rec,
                }
            rec_scores[action]["count"] += 1
            rec_scores[action]["priority_score"] += priority_scores.get(rec.get("priority", "low"), 1)
        
        # Sort by combined score (count * priority)
        sorted_recs = sorted(
            rec_scores.values(),
            key=lambda x: -(x["count"] * x["priority_score"])
        )
        
        moderator.recommendations = [x["original"] for x in sorted_recs[:10]]
        moderator.confidence_score = sum(p.confidence_score for p in perspectives) / len(perspectives)
        
        return moderator
    
    def _calculate_consensus(self, perspectives: List[AnalystPerspective]) -> float:
        """Calculate how much agreement exists among analysts."""
        # Simple measure: overlap in recommendations
        if len(perspectives) < 2:
            return 1.0
        
        # Get all recommendation actions
        all_actions = set()
        perspective_actions = []
        
        for p in perspectives:
            actions = set(r.get("action", "") for r in p.recommendations)
            perspective_actions.append(actions)
            all_actions.update(actions)
        
        if not all_actions:
            return 0.0
        
        # Calculate average overlap
        overlaps = []
        for i, actions_i in enumerate(perspective_actions):
            for j, actions_j in enumerate(perspective_actions[i+1:], i+1):
                if actions_i and actions_j:
                    overlap = len(actions_i & actions_j) / len(actions_i | actions_j)
                    overlaps.append(overlap)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0
    
    async def _persist_report(self, report: CouncilReport) -> None:
        """Save report to memory."""
        if not self._memory:
            return
        
        await self._memory.store_in_ems(
            content=json.dumps({
                "generated_at": report.generated_at.isoformat(),
                "period_start": report.period_start.isoformat(),
                "period_end": report.period_end.isoformat(),
                "signals_processed": report.signals_processed,
                "executive_summary": report.executive_summary,
                "strategic_priorities": report.strategic_priorities,
                "immediate_actions": report.immediate_actions,
                "risks_identified": report.risks_identified,
                "opportunities_identified": report.opportunities_identified,
                "consensus_score": report.consensus_score,
                "confidence_level": report.confidence_level,
            }),
            chunk_type=MemoryChunkType.FACT,
            importance=3.0,  # Business reports are important
            tags=["business_analysis", "council_report", "strategy"],
            summary=f"Business Council Report {report.generated_at.strftime('%Y-%m-%d')}: "
                   f"{len(report.strategic_priorities)} priorities, {report.confidence_level} confidence",
            load_hints=["business", "strategy", "analysis", "council", "recommendations"],
            source=f"council_report:{report.generated_at.isoformat()}",
        )
    
    async def get_latest_report(self) -> Optional[CouncilReport]:
        """Get most recent council report."""
        if self._report_history:
            return self._report_history[-1]
        return None
    
    async def format_report_for_display(self, report: CouncilReport) -> str:
        """Format report as readable markdown."""
        lines = [
            "# 🤖 AI Council Business Report",
            "",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Period:** {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}",
            f"**Signals Analyzed:** {report.signals_processed}",
            f"**Consensus Score:** {report.consensus_score:.1%}",
            f"**Confidence Level:** {report.confidence_level.upper()}",
            "",
            "## 📋 Executive Summary",
            "",
            report.executive_summary,
            "",
            "## 🎯 Strategic Priorities",
            "",
        ]
        
        for i, priority in enumerate(report.strategic_priorities[:5], 1):
            p_level = priority.get("priority", "medium").upper()
            action = priority.get("action", "No action specified")
            impact = priority.get("expected_impact", "Unknown")
            
            lines.append(f"{i}. **[{p_level}]** {action}")
            lines.append(f"   *Expected Impact:* {impact}")
            lines.append("")
        
        if report.immediate_actions:
            lines.extend([
                "## ⚡ Immediate Actions",
                "",
            ])
            for action in report.immediate_actions[:5]:
                lines.append(f"- {action}")
            lines.append("")
        
        if report.risks_identified:
            lines.extend([
                "## ⚠️ Risks Identified",
                "",
            ])
            for risk in report.risks_identified[:5]:
                lines.append(f"- {risk}")
            lines.append("")
        
        if report.opportunities_identified:
            lines.extend([
                "## 💡 Opportunities",
                "",
            ])
            for opp in report.opportunities_identified[:5]:
                lines.append(f"- {opp}")
            lines.append("")
        
        # Add analyst perspectives summary
        lines.extend([
            "## 🎭 Analyst Perspectives",
            "",
        ])
        
        for p in report.perspectives:
            lines.append(f"### {p.analyst.value.replace('_', ' ').title()}")
            lines.append(f"*Confidence: {p.confidence_score:.1f}/10*")
            
            if p.key_findings:
                lines.append("**Key Findings:**")
                for f in p.key_findings[:3]:
                    lines.append(f"- {f}")
            
            if p.concerns:
                lines.append("**Concerns:**")
                for c in p.concerns[:2]:
                    lines.append(f"- {c}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get analysis system statistics."""
        return {
            "signals_collected": len(self._recent_signals),
            "reports_generated": len(self._report_history),
            "last_report": self._report_history[-1].generated_at.isoformat() if self._report_history else None,
            "avg_consensus": sum(r.consensus_score for r in self._report_history) / len(self._report_history) if self._report_history else 0,
        }


# Import for type hints
from lollmsbot.agent.rlm.models import MemoryChunkType
from datetime import timedelta


__all__ = [
    "BusinessAnalysisCouncil",
    "CouncilReport",
    "AnalystPerspective",
    "BusinessSignal",
    "AnalystRole",
]
