"""
Calendar tool for LollmsBot.

This module provides the CalendarTool class for managing calendar events
with in-memory storage and optional ICS file export/import support.
All datetime operations are timezone-aware.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lollmsbot.agent import Tool, ToolResult, ToolError


@dataclass
class Event:
    """Represents a calendar event.
    
    Attributes:
        id: Unique identifier for the event (UUID string).
        title: Event title/summary.
        start: Start datetime (timezone-aware).
        end: End datetime (timezone-aware).
        description: Optional event description.
        created_at: Timestamp when the event was created.
        updated_at: Timestamp when the event was last updated.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    start: datetime = field(default_factory=datetime.now)
    end: datetime = field(default_factory=datetime.now)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> None:
        """Ensure datetime fields are timezone-aware."""
        if self.start.tzinfo is None:
            from datetime import timezone
            self.start = self.start.replace(tzinfo=timezone.utc)
        if self.end.tzinfo is None:
            from datetime import timezone
            self.end = self.end.replace(tzinfo=timezone.utc)
        if self.created_at.tzinfo is None:
            from datetime import timezone
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at.tzinfo is None:
            from datetime import timezone
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation.
        
        Returns:
            Dictionary with event data.
        """
        return {
            "id": self.id,
            "title": self.title,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary representation.
        
        Args:
            data: Dictionary with event data.
            
        Returns:
            New Event instance.
        """
        def parse_datetime(value: Any) -> datetime:
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            return value
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", ""),
            start=parse_datetime(data.get("start", datetime.now())),
            end=parse_datetime(data.get("end", datetime.now())),
            description=data.get("description", ""),
            created_at=parse_datetime(data.get("created_at", datetime.now())),
            updated_at=parse_datetime(data.get("updated_at", datetime.now())),
        )
    
    def to_ics(self) -> str:
        """Convert event to ICS format string.
        
        Returns:
            ICS VEVENT component as string.
        """
        from datetime import timezone
        
        def format_datetime(dt: datetime) -> str:
            """Format datetime for ICS (UTC or local with Z suffix)."""
            utc_dt = dt.astimezone(timezone.utc)
            return utc_dt.strftime("%Y%m%dT%H%M%SZ")
        
        ics_lines = [
            "BEGIN:VEVENT",
            f"UID:{self.id}",
            f"SUMMARY:{self.title}",
            f"DTSTART:{format_datetime(self.start)}",
            f"DTEND:{format_datetime(self.end)}",
        ]
        
        if self.description:
            # Escape special characters in description
            escaped_desc = self.description.replace("\\", "\\\\").replace("\n", "\\n").replace(",", "\\,").replace(";", "\\;")
            ics_lines.append(f"DESCRIPTION:{escaped_desc}")
        
        ics_lines.append(f"DTSTAMP:{format_datetime(datetime.now(timezone.utc))}")
        ics_lines.append("END:VEVENT")
        
        return "\r\n".join(ics_lines)
    
    @classmethod
    def from_ics(cls, ics_data: str) -> "Event":
        """Parse event from ICS format string.
        
        Args:
            ics_data: ICS VEVENT component as string.
            
        Returns:
            New Event instance.
        """
        from datetime import timezone
        
        lines = ics_data.replace("\r\n", "\n").split("\n")
        data: Dict[str, str] = {}
        
        for line in lines:
            if ":" in line and not line.startswith("BEGIN:") and not line.startswith("END:"):
                key, value = line.split(":", 1)
                # Handle property parameters (e.g., DTSTART;TZID=...)
                if ";" in key:
                    key = key.split(";")[0]
                data[key] = value
        
        def parse_ics_datetime(value: str) -> datetime:
            """Parse ICS datetime format."""
            value = value.strip()
            if value.endswith("Z"):
                # UTC datetime
                return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            elif "T" in value:
                # Local datetime (assume UTC if no tz specified)
                dt = datetime.strptime(value, "%Y%m%dT%H%M%S")
                return dt.replace(tzinfo=timezone.utc)
            else:
                # Date only
                dt = datetime.strptime(value, "%Y%m%d")
                return dt.replace(tzinfo=timezone.utc)
        
        # Unescape description
        description = data.get("DESCRIPTION", "")
        description = description.replace("\\n", "\n").replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\")
        
        return cls(
            id=data.get("UID", str(uuid.uuid4())),
            title=data.get("SUMMARY", ""),
            start=parse_ics_datetime(data.get("DTSTART", "")),
            end=parse_ics_datetime(data.get("DTEND", "")),
            description=description,
        )


class CalendarTool(Tool):
    """Tool for managing calendar events with in-memory storage.
    
    This tool provides CRUD operations for calendar events with optional
    ICS file import/export support. All datetime operations are timezone-aware.
    
    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema describing expected parameters for each method.
        events: In-memory storage of events indexed by ID.
        ics_file_path: Optional path for ICS file persistence.
    """
    
    name: str = "calendar"
    description: str = (
        "Manage calendar events including listing, adding, and deleting events. "
        "Supports filtering by date range and optional ICS file export/import. "
        "All datetimes are handled with timezone awareness."
    )
    
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["get_events", "add_event", "delete_event", "export_ics", "import_ics"],
                "description": "The calendar operation to perform",
            },
            "start": {
                "type": "string",
                "description": "Start datetime in ISO format (required for get_events, add_event)",
            },
            "end": {
                "type": "string",
                "description": "End datetime in ISO format (required for get_events, add_event)",
            },
            "title": {
                "type": "string",
                "description": "Event title (required for add_event)",
            },
            "description": {
                "type": "string",
                "description": "Event description (optional for add_event)",
            },
            "event_id": {
                "type": "string",
                "description": "Event ID (required for delete_event)",
            },
            "ics_path": {
                "type": "string",
                "description": "Path to ICS file (required for export_ics, import_ics)",
            },
        },
        "required": ["operation"],
    }
    
    def __init__(
        self,
        ics_file_path: Optional[str] = None,
        default_timezone: str = "UTC",
    ) -> None:
        """Initialize the CalendarTool.
        
        Args:
            ics_file_path: Optional path for ICS file persistence.
            default_timezone: Default timezone for datetime operations.
        """
        self.events: Dict[str, Event] = {}
        self.ics_file_path: Optional[Path] = Path(ics_file_path) if ics_file_path else None
        self.default_timezone: str = default_timezone
        
        # Load existing events from ICS if available
        if self.ics_file_path and self.ics_file_path.exists():
            asyncio.create_task(self._load_from_ics_async())
    
    def _get_timezone(self, tz_name: Optional[str] = None) -> Any:
        """Get timezone object by name.
        
        Args:
            tz_name: Timezone name (e.g., 'UTC', 'America/New_York').
                    Defaults to default_timezone if not specified.
            
        Returns:
            Timezone object.
        """
        import zoneinfo
        
        tz = tz_name or self.default_timezone
        try:
            return zoneinfo.ZoneInfo(tz)
        except zoneinfo.ZoneInfoNotFoundError:
            # Fallback to UTC if timezone not found
            return zoneinfo.ZoneInfo("UTC")
    
    def _parse_datetime(self, value: str | datetime) -> datetime:
        """Parse datetime from string or return datetime object.
        
        Args:
            value: ISO format string or datetime object.
            
        Returns:
            Timezone-aware datetime object.
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                from datetime import timezone
                return value.replace(tzinfo=timezone.utc)
            return value
        
        # Try ISO format parsing
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            # Try common formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d/%m/%Y %H:%M:%S",
                "%d/%m/%Y",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    from datetime import timezone
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            raise ValueError(f"Unable to parse datetime: {value}")
    
    async def get_events(
        self,
        start: Optional[str | datetime] = None,
        end: Optional[str | datetime] = None,
    ) -> ToolResult:
        """Get events within a date range.
        
        Args:
            start: Start of date range (inclusive). If None, no lower bound.
            end: End of date range (inclusive). If None, no upper bound.
            
        Returns:
            ToolResult with list of matching events.
        """
        try:
            parsed_start: Optional[datetime] = None
            parsed_end: Optional[datetime] = None
            
            if start:
                parsed_start = self._parse_datetime(start)
            if end:
                parsed_end = self._parse_datetime(end)
            
            matching_events: List[Event] = []
            
            for event in self.events.values():
                # Check if event overlaps with range
                event_in_range = True
                
                if parsed_start and event.end < parsed_start:
                    event_in_range = False
                if parsed_end and event.start > parsed_end:
                    event_in_range = False
                
                if event_in_range:
                    matching_events.append(event)
            
            # Sort by start time
            matching_events.sort(key=lambda e: e.start)
            
            return ToolResult(
                success=True,
                output={
                    "count": len(matching_events),
                    "events": [e.to_dict() for e in matching_events],
                },
                error=None,
            )
            
        except ValueError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid datetime format: {str(exc)}",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error retrieving events: {str(exc)}",
            )
    
    async def add_event(
        self,
        title: str,
        start: str | datetime,
        end: str | datetime,
        description: str = "",
    ) -> ToolResult:
        """Add a new calendar event.
        
        Args:
            title: Event title/summary.
            start: Event start datetime.
            end: Event end datetime.
            description: Optional event description.
            
        Returns:
            ToolResult with the created event data.
        """
        try:
            if not title:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Event title is required",
                )
            
            parsed_start = self._parse_datetime(start)
            parsed_end = self._parse_datetime(end)
            
            if parsed_end <= parsed_start:
                return ToolResult(
                    success=False,
                    output=None,
                    error="End time must be after start time",
                )
            
            event = Event(
                title=title,
                start=parsed_start,
                end=parsed_end,
                description=description,
            )
            
            self.events[event.id] = event
            
            # Persist to ICS if path is configured
            if self.ics_file_path:
                await self._save_to_ics_async()
            
            return ToolResult(
                success=True,
                output={
                    "message": f"Event '{title}' created successfully",
                    "event": event.to_dict(),
                },
                error=None,
            )
            
        except ValueError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid datetime format: {str(exc)}",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error creating event: {str(exc)}",
            )
    
    async def delete_event(self, event_id: str) -> ToolResult:
        """Delete a calendar event by ID.
        
        Args:
            event_id: UUID of the event to delete.
            
        Returns:
            ToolResult indicating success or failure.
        """
        try:
            if event_id not in self.events:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Event with ID '{event_id}' not found",
                )
            
            event = self.events.pop(event_id)
            
            # Persist to ICS if path is configured
            if self.ics_file_path:
                await self._save_to_ics_async()
            
            return ToolResult(
                success=True,
                output={
                    "message": f"Event '{event.title}' deleted successfully",
                    "deleted_event": event.to_dict(),
                },
                error=None,
            )
            
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error deleting event: {str(exc)}",
            )
    
    async def export_ics(self, path: Optional[str] = None) -> ToolResult:
        """Export all events to ICS file.
        
        Args:
            path: Output file path. Uses configured ics_file_path if not specified.
            
        Returns:
            ToolResult with export status and file path.
        """
        try:
            output_path = Path(path) if path else self.ics_file_path
            
            if not output_path:
                return ToolResult(
                    success=False,
                    output=None,
                    error="No output path specified. Provide path parameter or configure ics_file_path.",
                )
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build ICS content
            ics_lines = [
                "BEGIN:VCALENDAR",
                "VERSION:2.0",
                "PRODID:-//LollmsBot//Calendar Tool//EN",
                "CALSCALE:GREGORIAN",
                "METHOD:PUBLISH",
            ]
            
            for event in sorted(self.events.values(), key=lambda e: e.start):
                ics_lines.append(event.to_ics())
            
            ics_lines.append("END:VCALENDAR")
            
            # Write file
            ics_content = "\r\n".join(ics_lines)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: output_path.write_text(ics_content, encoding="utf-8"),
            )
            
            return ToolResult(
                success=True,
                output={
                    "message": f"Exported {len(self.events)} events to {output_path}",
                    "path": str(output_path),
                    "event_count": len(self.events),
                },
                error=None,
            )
            
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error exporting ICS file: {str(exc)}",
            )
    
    async def import_ics(self, path: Optional[str] = None) -> ToolResult:
        """Import events from ICS file.
        
        Args:
            path: Input file path. Uses configured ics_file_path if not specified.
            
        Returns:
            ToolResult with import status and count of imported events.
        """
        try:
            input_path = Path(path) if path else self.ics_file_path
            
            if not input_path:
                return ToolResult(
                    success=False,
                    output=None,
                    error="No input path specified. Provide path parameter or configure ics_file_path.",
                )
            
            if not input_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"ICS file not found: {input_path}",
                )
            
            # Read file
            loop = asyncio.get_event_loop()
            ics_content = await loop.run_in_executor(
                None,
                lambda: input_path.read_text(encoding="utf-8"),
            )
            
            # Parse VEVENT blocks
            imported_count = 0
            in_event = False
            event_lines: List[str] = []
            
            for line in ics_content.replace("\r\n", "\n").split("\n"):
                if line.startswith("BEGIN:VEVENT"):
                    in_event = True
                    event_lines = [line]
                elif line.startswith("END:VEVENT"):
                    in_event = False
                    event_lines.append(line)
                    try:
                        event = Event.from_ics("\n".join(event_lines))
                        self.events[event.id] = event
                        imported_count += 1
                    except Exception:
                        # Skip malformed events
                        pass
                elif in_event:
                    event_lines.append(line)
            
            return ToolResult(
                success=True,
                output={
                    "message": f"Imported {imported_count} events from {input_path}",
                    "path": str(input_path),
                    "imported_count": imported_count,
                    "total_events": len(self.events),
                },
                error=None,
            )
            
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error importing ICS file: {str(exc)}",
            )
    
    async def _save_to_ics_async(self) -> None:
        """Persist events to ICS file asynchronously."""
        if self.ics_file_path:
            await self.export_ics(str(self.ics_file_path))
    
    async def _load_from_ics_async(self) -> None:
        """Load events from ICS file asynchronously."""
        if self.ics_file_path and self.ics_file_path.exists():
            await self.import_ics(str(self.ics_file_path))
    
    async def execute(self, **params: Any) -> ToolResult:
        """Execute a calendar operation based on parameters.
        
        This is the main entry point for the Tool base class. It dispatches
        to the appropriate method based on the 'operation' parameter.
        
        Args:
            **params: Parameters must include:
                - operation: One of 'get_events', 'add_event', 'delete_event',
                           'export_ics', 'import_ics'
                - Additional parameters depend on the operation
                
        Returns:
            ToolResult from the executed operation.
            
        Raises:
            ToolError: If the operation is unknown or parameters are invalid.
        """
        operation = params.get("operation")
        
        if not operation:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: 'operation'",
            )
        
        # Dispatch to appropriate method
        if operation == "get_events":
            return await self.get_events(
                start=params.get("start"),
                end=params.get("end"),
            )
        
        elif operation == "add_event":
            title = params.get("title")
            start = params.get("start")
            end = params.get("end")
            
            if not title:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Missing required parameter: 'title'",
                )
            if not start:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Missing required parameter: 'start'",
                )
            if not end:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Missing required parameter: 'end'",
                )
            
            return await self.add_event(
                title=title,
                start=start,
                end=end,
                description=params.get("description", ""),
            )
        
        elif operation == "delete_event":
            event_id = params.get("event_id")
            if not event_id:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Missing required parameter: 'event_id'",
                )
            return await self.delete_event(event_id)
        
        elif operation == "export_ics":
            return await self.export_ics(params.get("ics_path"))
        
        elif operation == "import_ics":
            return await self.import_ics(params.get("ics_path"))
        
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown operation: '{operation}'. "
                      f"Valid operations are: get_events, add_event, delete_event, export_ics, import_ics",
            )