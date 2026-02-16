"""
Task Management Tools for SimplifiedAgant integration
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lollmsbot.agent import Tool, ToolResult


class CreateTaskTool(Tool):
    """Create a new task."""
    
    name: str = "create_task"
    description: str = (
        "Create a new task with optional due date, priority, and CRM contact linking. "
        "Tasks are automatically synced to Todoist if configured. "
        "Can extract tasks from meeting transcripts automatically."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Task description",
            },
            "assignee": {
                "type": "string",
                "description": "Who should do this (email or 'me')",
                "default": "me",
            },
            "due_date": {
                "type": "string",
                "description": "Due date (ISO format or 'tomorrow', 'next week')",
            },
            "priority": {
                "type": "integer",
                "description": "Priority 1-4 (1=urgent, 4=low)",
                "default": 4,
            },
            "related_contact": {
                "type": "string",
                "description": "Email of related contact for CRM context",
            },
        },
        "required": ["content"],
    }
    
    risk_level: str = "low"
    
    def __init__(self, task_manager: Optional[Any] = None):
        self._tasks = task_manager
    
    async def execute(self, content: str, assignee: str = "me",
                     due_date: Optional[str] = None, priority: int = 4,
                     related_contact: Optional[str] = None, **kwargs) -> ToolResult:
        if not self._tasks:
            return ToolResult(
                success=False,
                output=None,
                error="Task manager not initialized",
            )
        
        try:
            # Parse due date
            parsed_date = None
            if due_date:
                from datetime import datetime, timedelta
                if due_date.lower() == "tomorrow":
                    parsed_date = datetime.now() + timedelta(days=1)
                elif due_date.lower() == "next week":
                    parsed_date = datetime.now() + timedelta(days=7)
                else:
                    try:
                        parsed_date = datetime.fromisoformat(due_date)
                    except:
                        pass
            
            from lollmsbot.task_manager import TaskPriority
            task_priority = TaskPriority(priority)
            
            task = await self._tasks.create_task(
                content=content,
                assignee=assignee,
                due_date=parsed_date,
                priority=task_priority,
                related_contact=related_contact,
            )
            
            return ToolResult(
                success=True,
                output={
                    "task_id": task.task_id,
                    "content": task.content,
                    "assignee": task.assignee,
                    "priority": task.priority.name,
                    "created": True,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Task creation failed: {str(e)}",
            )


class GetTasksTool(Tool):
    """Get tasks for review."""
    
    name: str = "get_tasks"
    description: str = (
        "Get pending tasks for review. "
        "Returns tasks sorted by priority and due date, "
        "with optional filtering by assignee or related contact."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "enum": ["all", "mine", "overdue", "contact"],
                "default": "mine",
            },
            "contact_email": {
                "type": "string",
                "description": "Filter by related contact (if filter=contact)",
            },
        },
    }
    
    risk_level: str = "low"
    
    def __init__(self, task_manager: Optional[Any] = None):
        self._tasks = task_manager
    
    async def execute(self, filter: str = "mine", contact_email: Optional[str] = None, **kwargs) -> ToolResult:
        if not self._tasks:
            return ToolResult(
                success=False,
                output=None,
                error="Task manager not initialized",
            )
        
        try:
            if filter == "contact" and contact_email:
                tasks = await self._tasks.get_contact_tasks(contact_email)
            else:
                tasks = await self._tasks.get_tasks_for_review()
            
            return ToolResult(
                success=True,
                output={
                    "filter": filter,
                    "tasks": [t.to_dict() for t in tasks[:20]],  # Top 20
                    "count": len(tasks),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Task retrieval failed: {str(e)}",
            )


__all__ = ["CreateTaskTool", "GetTasksTool"]
