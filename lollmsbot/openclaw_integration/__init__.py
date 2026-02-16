"""
SimplifiedAgant Integration Module - Main orchestrator for advanced features

Integrates CRM, Knowledge Base, Task Manager, YouTube Analytics,
and Business Analysis into the core lollmsBot agent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from lollmsbot.crm import CRMManager
from lollmsbot.knowledge_base import KnowledgeBaseManager
from lollmsbot.task_manager import TaskManager
from lollmsbot.youtube_analytics import YouTubeAnalyticsManager
from lollmsbot.business_analysis import BusinessAnalysisCouncil, BusinessSignal

import logging

logger = logging.getLogger(__name__)


class SimplifiedAgantIntegration:
    """
    Main integration class that brings SimplifiedAgant-like capabilities to lollmsBot.
    
    This orchestrates:
    - CRM with Gmail/calendar ingestion
    - Knowledge base with web content
    - Task management with Todoist
    - YouTube analytics
    - Business meta-analysis (AI Council)
    - Automated workflows via heartbeat
    """
    
    def __init__(self, agent: Any) -> None:
        self._agent = agent
        self._memory = None  # Set during initialization
        
        # Subsystems
        self.crm: Optional[CRMManager] = None
        self.knowledge_base: Optional[KnowledgeBaseManager] = None
        self.task_manager: Optional[TaskManager] = None
        self.youtube_analytics: Optional[YouTubeAnalyticsManager] = None
        self.business_council: Optional[BusinessAnalysisCouncil] = None
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all subsystems."""
        if self._initialized:
            return
        
        # Get memory manager from agent
        if hasattr(self._agent, '_memory'):
            self._memory = self._agent._memory
        
        if not self._memory:
            logger.error("Cannot initialize SimplifiedAgant features - no memory manager available")
            return
        
        logger.info("🚀 Initializing SimplifiedAgant integration...")
        
        # Initialize CRM
        self.crm = CRMManager(self._memory)
        await self.crm.initialize()
        
        # Initialize Knowledge Base
        self.knowledge_base = KnowledgeBaseManager(self._memory)
        await self.knowledge_base.initialize()
        
        # Initialize Task Manager (with CRM integration)
        self.task_manager = TaskManager(self._memory, self.crm)
        await self.task_manager.initialize()
        
        # Initialize YouTube Analytics
        self.youtube_analytics = YouTubeAnalyticsManager(self._memory)
        await self.youtube_analytics.initialize()
        
        # Initialize Business Analysis Council
        llm_client = getattr(self._agent, '_lollms_client', None)
        self.business_council = BusinessAnalysisCouncil(self._memory, llm_client)
        await self.business_council.initialize()
        
        self._initialized = True
        logger.info("✅ SimplifiedAgant integration fully initialized")
    
    def configure_apis(self, config: Dict[str, str]) -> None:
        """Configure external API keys."""
        # Todoist
        if "todoist_api_key" in config:
            if self.task_manager:
                self.task_manager.set_todoist_api_key(config["todoist_api_key"])
        
        # YouTube
        if "youtube_api_key" in config and "youtube_channel_id" in config:
            if self.youtube_analytics:
                self.youtube_analytics.set_api_credentials(
                    config["youtube_api_key"],
                    config["youtube_channel_id"]
                )
    
    async def daily_workflow(self) -> Dict[str, Any]:
        """
        Run the daily automated workflow (called by heartbeat).
        
        This replicates SimplifiedAgant's cron-based automation:
        1. Ingest emails and calendar
        2. Update CRM
        3. Collect YouTube analytics
        4. Run business analysis
        5. Generate reports
        """
        if not self._initialized:
            return {"error": "Not initialized"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "steps_completed": [],
        }
        
        try:
            # Step 1: CRM daily ingestion (would connect to Gmail API in production)
            # For now, this is a placeholder
            logger.info("Daily workflow: CRM ingestion")
            # crm_digest = await self.crm.get_daily_digest()
            results["steps_completed"].append("crm_ready")
            
        except Exception as e:
            logger.error(f"CRM daily workflow failed: {e}")
            results["crm_error"] = str(e)
        
        try:
            # Step 2: YouTube analytics
            if self.youtube_analytics:
                snapshot = await self.youtube_analytics.ingest_daily_analytics()
                results["youtube_snapshot"] = {
                    "views": snapshot.total_views,
                    "subscribers": snapshot.total_subscribers,
                }
                results["steps_completed"].append("youtube_analytics")
                
                # Ingest as business signal
                if self.business_council:
                    await self.business_council.ingest_signal(
                        source="youtube",
                        signal_type="metric",
                        content=f"Channel views: {snapshot.total_views:,}, subscribers: {snapshot.total_subscribers:,}",
                        confidence=8.0,
                        metadata={
                            "views": snapshot.total_views,
                            "subscribers": snapshot.total_subscribers,
                            "growth_7d": snapshot.subscriber_growth_7d,
                        }
                    )
                
        except Exception as e:
            logger.error(f"YouTube analytics failed: {e}")
            results["youtube_error"] = str(e)
        
        try:
            # Step 3: Business council analysis (runs nightly)
            if self.business_council:
                report = await self.business_council.run_council_analysis(period_days=1)
                results["business_analysis"] = {
                    "priorities": len(report.strategic_priorities),
                    "consensus": report.consensus_score,
                    "confidence": report.confidence_level,
                }
                results["steps_completed"].append("business_analysis")
                
        except Exception as e:
            logger.error(f"Business analysis failed: {e}")
            results["analysis_error"] = str(e)
        
        return results
    
    async def process_meeting_transcript(self, transcript: str, 
                                          participants: List[Dict[str, Any]],
                                          meeting_title: str,
                                          meeting_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process meeting transcript - core SimplifiedAgant feature.
        
        Extracts tasks, updates CRM, creates follow-ups.
        """
        if not self._initialized:
            return {"error": "Not initialized"}
        
        results = {
            "tasks_extracted": 0,
            "contacts_updated": 0,
            "tasks": [],
        }
        
        # Extract tasks
        if self.task_manager:
            tasks = await self.task_manager.extract_tasks_from_meeting(
                transcript, participants, meeting_title, meeting_id
            )
            results["tasks_extracted"] = len(tasks)
            results["tasks"] = [t.to_dict() for t in tasks]
        
        # Update CRM with meeting interaction
        if self.crm:
            for participant in participants:
                # Would add meeting interaction to contact
                pass
        
        # Ingest as business signal
        if self.business_council:
            await self.business_council.ingest_signal(
                source="meeting",
                signal_type="event",
                content=f"Meeting: {meeting_title} with {len(participants)} participants",
                confidence=7.0,
                metadata={
                    "title": meeting_title,
                    "participant_count": len(participants),
                    "tasks_extracted": results["tasks_extracted"],
                }
            )
        
        return results
    
    async def ingest_web_content(self, url: str, title: Optional[str] = None,
                                  content: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest web content into knowledge base.
        
        Also triggers video idea extraction if relevant.
        """
        if not self._initialized or not self.knowledge_base:
            return {"error": "Knowledge base not initialized"}
        
        entry = await self.knowledge_base.ingest_url(url, title, content)
        
        if not entry:
            return {"error": "Failed to ingest content"}
        
        # Check for video ideas
        ideas = []
        if "video" in entry.title.lower() or "youtube" in entry.source_type:
            ideas = await self.knowledge_base.get_video_ideas(entry.title)
        
        # Ingest as business signal
        if self.business_council:
            await self.business_council.ingest_signal(
                source="knowledge_base",
                signal_type="opportunity" if ideas else "metric",
                content=f"New knowledge entry: {entry.title}",
                confidence=6.0,
                metadata={
                    "entry_id": entry.entry_id,
                    "source_url": entry.source_url,
                    "video_ideas": len(ideas),
                }
            )
        
        return {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "video_ideas_found": len(ideas),
            "ideas": ideas[:3],
        }
    
    async def get_meeting_prep(self, contact_email: str, 
                               meeting_topic: Optional[str] = None) -> str:
        """Get meeting preparation for a contact."""
        if not self._initialized or not self.crm:
            return "CRM not available"
        
        return await self.crm.get_meeting_prep(contact_email, meeting_topic)
    
    async def query_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Query knowledge base."""
        if not self._initialized or not self.knowledge_base:
            return []
        
        return await self.knowledge_base.query(query)
    
    async def get_business_report(self) -> str:
        """Get latest business analysis report."""
        if not self._initialized or not self.business_council:
            return "Business analysis not available"
        
        report = await self.business_council.get_latest_report()
        if not report:
            return "No business reports available yet"
        
        return await self.business_council.format_report_for_display(report)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "initialized": self._initialized,
            "crm_contacts": len(self.crm._contacts) if self.crm else 0,
            "knowledge_entries": len(self.knowledge_base._entries) if self.knowledge_base else 0,
            "tasks_tracked": len(self.task_manager._tasks) if self.task_manager else 0,
            "youtube_videos": len(self.youtube_analytics._videos) if self.youtube_analytics else 0,
            "business_reports": len(self.business_council._report_history) if self.business_council else 0,
        }


from datetime import datetime

__all__ = [
    "SimplifiedAgantIntegration",
]
