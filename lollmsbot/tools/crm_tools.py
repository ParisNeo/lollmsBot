"""
CRM Tools for SimplifiedAgant integration
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lollmsbot.agent import Tool, ToolResult


class CRMQueryTool(Tool):
    """Query the CRM for contacts and interactions."""
    
    name: str = "crm_query"
    description: str = (
        "Query the CRM system to find contacts, view interaction history, "
        "and get information about people in your network. "
        "Use natural language queries like 'who do I know at Google' or "
        "'what did I last discuss with John'."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query about contacts",
            },
        },
        "required": ["query"],
    }
    
    risk_level: str = "low"
    
    def __init__(self, crm_manager: Optional[Any] = None):
        self._crm = crm_manager
    
    async def execute(self, query: str, **kwargs) -> ToolResult:
        if not self._crm:
            return ToolResult(
                success=False,
                output=None,
                error="CRM not initialized",
            )
        
        try:
            results = await self._crm.query_contacts(query)
            
            return ToolResult(
                success=True,
                output={
                    "query": query,
                    "results": results,
                    "count": len(results),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"CRM query failed: {str(e)}",
            )


class MeetingPrepTool(Tool):
    """Get meeting preparation for a contact."""
    
    name: str = "meeting_prep"
    description: str = (
        "Generate meeting preparation for a specific contact. "
        "Provides conversation history, context, and suggested talking points. "
        "Essential for important meetings."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "contact_email": {
                "type": "string",
                "description": "Email address of the contact",
            },
            "meeting_topic": {
                "type": "string",
                "description": "Optional topic of the meeting",
            },
        },
        "required": ["contact_email"],
    }
    
    risk_level: str = "low"
    
    def __init__(self, crm_manager: Optional[Any] = None):
        self._crm = crm_manager
    
    async def execute(self, contact_email: str, meeting_topic: Optional[str] = None, **kwargs) -> ToolResult:
        if not self._crm:
            return ToolResult(
                success=False,
                output=None,
                error="CRM not initialized",
            )
        
        try:
            prep = await self._crm.get_meeting_prep(contact_email, meeting_topic)
            
            return ToolResult(
                success=True,
                output={
                    "contact": contact_email,
                    "meeting_topic": meeting_topic,
                    "preparation": prep,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Meeting prep failed: {str(e)}",
            )


__all__ = ["CRMQueryTool", "MeetingPrepTool"]
