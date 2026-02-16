"""
Task Management Module - Todoist integration and intelligent task extraction

Provides meeting transcript analysis, automatic task extraction,
task assignment with CRM context, and Todoist synchronization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    P1 = 1  # Urgent
    P2 = 2  # High
    P3 = 3  # Medium
    P4 = 4  # Low


@dataclass
class Task:
    """A task with metadata."""
    task_id: str
    content: str
    description: Optional[str] = None
    priority: TaskPriority = TaskPriority.P4
    due_date: Optional[datetime] = None
    assignee: Optional[str] = None  # "me" or contact email
    source: str = "manual"  # "meeting", "manual", "crm", "ai_suggested"
    source_id: Optional[str] = None  # Meeting ID, email ID, etc.
    project_id: Optional[str] = None  # Todoist project
    labels: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completed_at: Optional[datetime] = None
    
    # CRM context
    related_contact: Optional[str] = None  # Email of related contact
    related_company: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "content": self.content,
            "description": self.description,
            "priority": self.priority.value,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "assignee": self.assignee,
            "source": self.source,
            "source_id": self.source_id,
            "project_id": self.project_id,
            "labels": list(self.labels),
            "created_at": self.created_at.isoformat(),
            "completed": self.completed,
            "related_contact": self.related_contact,
            "related_company": self.related_company,
        }
    
    def to_todoist_format(self) -> Dict[str, Any]:
        """Convert to Todoist API format."""
        return {
            "content": self.content,
            "description": self.description or "",
            "priority": self.priority.value,
            "due_string": self._format_due_date(),
            "labels": list(self.labels),
            "project_id": self.project_id,
        }
    
    def _format_due_date(self) -> Optional[str]:
        """Format due date for Todoist."""
        if not self.due_date:
            return None
        
        # If it's today, use "today"
        if self.due_date.date() == datetime.now().date():
            return "today"
        
        # If it's tomorrow
        if self.due_date.date() == (datetime.now() + timedelta(days=1)).date():
            return "tomorrow"
        
        # Otherwise use ISO format
        return self.due_date.isoformat()


class TaskManager:
    """
    Manages tasks with intelligent extraction and Todoist integration.
    
    Key features from SimplifiedAgant:
    - Meeting transcript analysis for task extraction
    - Automatic task assignment (me vs. attendee)
    - CRM cross-referencing for context
    - Todoist synchronization
    - Natural language task creation
    """
    
    def __init__(self, memory_manager: Any, crm_manager: Optional[Any] = None) -> None:
        self._memory = memory_manager
        self._crm = crm_manager
        self._tasks: Dict[str, Task] = {}
        self._todoist_api_key: Optional[str] = None
        self._initialized = False
    
    def set_todoist_api_key(self, api_key: str) -> None:
        """Set Todoist API key for synchronization."""
        self._todoist_api_key = api_key
        logger.info("Todoist API key configured")
    
    async def initialize(self) -> None:
        """Load existing tasks from memory."""
        if self._initialized:
            return
        
        if self._memory:
            results = await self._memory.search_ems(
                query="task_manager task",
                chunk_types=[MemoryChunkType.FACT],
                limit=100,
            )
            
            for chunk, score in results:
                if "task:" in (chunk.source or ""):
                    try:
                        import json
                        data = json.loads(chunk.decompress_content())
                        task = self._dict_to_task(data)
                        self._tasks[task.task_id] = task
                    except Exception as e:
                        logger.warning(f"Failed to load task: {e}")
        
        self._initialized = True
        logger.info(f"Task manager initialized with {len(self._tasks)} tasks")
    
    def _dict_to_task(self, data: Dict[str, Any]) -> Task:
        """Reconstruct task from dictionary."""
        return Task(
            task_id=data["task_id"],
            content=data["content"],
            description=data.get("description"),
            priority=TaskPriority(data.get("priority", 4)),
            due_date=datetime.fromisoformat(data["due_date"]) if data.get("due_date") else None,
            assignee=data.get("assignee"),
            source=data.get("source", "manual"),
            source_id=data.get("source_id"),
            project_id=data.get("project_id"),
            labels=set(data.get("labels", [])),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed=data.get("completed", False),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            related_contact=data.get("related_contact"),
            related_company=data.get("related_company"),
        )
    
    async def extract_tasks_from_meeting(self, transcript: str, 
                                        participants: List[Dict[str, Any]],
                                        meeting_title: str,
                                        meeting_id: Optional[str] = None) -> List[Task]:
        """
        Extract tasks from meeting transcript using AI analysis.
        
        This is the core feature from SimplifiedAgant - automatic task extraction
        from Fathom/AI notetaker transcripts.
        """
        # In production, this would call an LLM to analyze the transcript
        # For now, use pattern matching
        
        tasks = []
        
        # Look for action item patterns
        action_patterns = [
            r'(?:I|we|you) (?:will|need to|should|must|have to) ([^.]+)',
            r'(?:action item|todo|task|follow up)(?:[:\s]+)([^.]+)',
            r'(?:remind|schedule|set up|prepare|review|send|call|email)(?:\s+me\s+to\s+)?([^.]+)',
        ]
        
        extracted_items = []
        for pattern in action_patterns:
            for match in re.finditer(pattern, transcript, re.IGNORECASE):
                action = match.group(1).strip()
                if len(action) > 10:  # Filter out very short matches
                    extracted_items.append(action)
        
        # Deduplicate
        seen = set()
        unique_items = []
        for item in extracted_items:
            normalized = item.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_items.append(item)
        
        # Create tasks with context
        for i, item in enumerate(unique_items[:5], 1):  # Limit to top 5
            # Determine assignee based on context
            assignee = "me"  # Default
            
            # Check for attendee references
            for participant in participants:
                name = participant.get("name", "").lower()
                email = participant.get("email", "").lower()
                
                if name in item.lower() or email.split("@")[0] in item.lower():
                    assignee = email
                    break
            
            # Determine priority based on urgency words
            priority = TaskPriority.P3
            urgent_words = ["urgent", "asap", "immediately", "critical", "deadline"]
            if any(w in item.lower() for w in urgent_words):
                priority = TaskPriority.P1
            
            # Try to extract due date
            due_date = None
            date_patterns = [
                (r'by (tomorrow|next week|Monday|Tuesday|Wednesday|Thursday|Friday)', 
                 lambda m: self._parse_relative_date(m.group(1))),
                (r'by (\d{1,2}/\d{1,2}(?:/\d{2,4})?)',
                 lambda m: self._parse_date(m.group(1))),
            ]
            
            for pattern, parser in date_patterns:
                match = re.search(pattern, item, re.IGNORECASE)
                if match:
                    try:
                        due_date = parser(match)
                        break
                    except:
                        pass
            
            # Cross-reference with CRM
            related_contact = None
            related_company = None
            
            if self._crm:
                for participant in participants:
                    email = participant.get("email", "").lower()
                    contact = await self._crm.query_contacts(email)
                    if contact:
                        related_contact = email
                        related_company = contact[0].get("company")
                        break
            
            task = Task(
                task_id=f"task_meeting_{meeting_id or 'unknown'}_{i}",
                content=item,
                description=f"From meeting: {meeting_title}\n\nContext: Extracted from transcript",
                priority=priority,
                due_date=due_date,
                assignee=assignee,
                source="meeting",
                source_id=meeting_id,
                labels={"meeting", "auto_extracted"},
                related_contact=related_contact,
                related_company=related_company,
            )
            
            tasks.append(task)
            self._tasks[task.task_id] = task
            await self._persist_task(task)
        
        # Sync to Todoist if configured
        if self._todoist_api_key:
            for task in tasks:
                await self._sync_to_todoist(task)
        
        return tasks
    
    def _parse_relative_date(self, phrase: str) -> Optional[datetime]:
        """Parse relative date phrases."""
        phrase = phrase.lower()
        now = datetime.now()
        
        if phrase == "tomorrow":
            return now + timedelta(days=1)
        elif phrase == "next week":
            return now + timedelta(days=7)
        elif phrase in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            # Find next occurrence of that weekday
            weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            target_day = weekdays.index(phrase)
            days_ahead = (target_day - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # Next week if today
            return now + timedelta(days=days_ahead)
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string."""
        # Simple parsing - would use dateutil in production
        formats = ["%m/%d/%Y", "%m/%d/%y", "%m/%d"]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        return None
    
    async def create_task(self, content: str, assignee: str = "me",
                         due_date: Optional[datetime] = None,
                         priority: TaskPriority = TaskPriority.P4,
                         related_contact: Optional[str] = None) -> Task:
        """Create a task manually or from natural language."""
        
        # Generate task ID
        import hashlib
        task_id = f"task_manual_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        
        # Cross-reference with CRM if contact provided
        company = None
        if related_contact and self._crm:
            contacts = await self._crm.query_contacts(related_contact)
            if contacts:
                company = contacts[0].get("company")
        
        task = Task(
            task_id=task_id,
            content=content,
            assignee=assignee,
            due_date=due_date,
            priority=priority,
            source="manual",
            related_contact=related_contact,
            related_company=company,
            labels={"manual"},
        )
        
        self._tasks[task.task_id] = task
        await self._persist_task(task)
        
        # Sync to Todoist
        if self._todoist_api_key:
            await self._sync_to_todoist(task)
        
        return task
    
    async def _persist_task(self, task: Task) -> None:
        """Save task to memory."""
        if not self._memory:
            return
        
        import json
        
        await self._memory.store_in_ems(
            content=json.dumps(task.to_dict()),
            chunk_type=MemoryChunkType.FACT,
            importance=task.priority.value / 2,  # Higher priority = more important
            tags=["task", "task_manager", task.source, f"priority_{task.priority.name}"],
            summary=f"Task: {task.content[:60]}... ({task.assignee})",
            load_hints=[task.content, task.assignee or "", task.related_contact or ""],
            source=f"task:{task.task_id}",
        )
    
    async def _sync_to_todoist(self, task: Task) -> bool:
        """Sync task to Todoist."""
        if not self._todoist_api_key:
            return False
        
        # Would call Todoist API here
        # For now, just log
        logger.info(f"Would sync to Todoist: {task.content[:50]}...")
        return True
    
    async def get_tasks_for_review(self) -> List[Task]:
        """Get pending tasks for user review."""
        pending = [t for t in self._tasks.values() if not t.completed]
        
        # Sort by priority and due date
        def sort_key(t: Task):
            due = t.due_date or datetime.max
            return (t.priority.value, due)
        
        pending.sort(key=sort_key)
        return pending
    
    async def get_contact_tasks(self, contact_email: str) -> List[Task]:
        """Get all tasks related to a specific contact."""
        return [
            t for t in self._tasks.values()
            if t.related_contact and t.related_contact.lower() == contact_email.lower()
        ]
    
    async def complete_task(self, task_id: str) -> bool:
        """Mark task as complete."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task.completed = True
        task.completed_at = datetime.now()
        
        await self._persist_task(task)
        
        # Sync to Todoist
        if self._todoist_api_key:
            # Would call Todoist API to complete
            pass
        
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        total = len(self._tasks)
        completed = sum(1 for t in self._tasks.values() if t.completed)
        pending = total - completed
        
        by_source = {}
        for t in self._tasks.values():
            by_source[t.source] = by_source.get(t.source, 0) + 1
        
        overdue = len([
            t for t in self._tasks.values()
            if not t.completed and t.due_date and t.due_date < datetime.now()
        ])
        
        return {
            "total_tasks": total,
            "completed": completed,
            "pending": pending,
            "overdue": overdue,
            "by_source": by_source,
        }


# Import for type hints
from lollmsbot.agent.rlm.models import MemoryChunkType


__all__ = [
    "TaskManager",
    "Task",
    "TaskPriority",
]
