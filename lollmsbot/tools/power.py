"""
Power Management Tool for LollmsBot Agent

Allows the agent to control system power state - preventing or allowing
sleep/shutdown. This is critical for maintaining long-running connections
(Discord bot, WebSocket clients, etc.).

SECURITY: This tool requires ADMIN permission level due to system-level impact.
"""

from __future__ import annotations

from typing import Any, Dict

from lollmsbot.agent import Tool, ToolResult
from lollmsbot.power_management import get_power_manager, PowerManager


class PowerTool(Tool):
    """
    Tool for controlling system power management.
    
    Allows preventing sleep/shutdown (for maintaining connections) or
    restoring normal power behavior.
    
    SECURITY NOTE: This tool requires ADMIN permission level as it affects
    system-wide power state. Use with caution on battery-powered devices.
    """
    
    name: str = "power"
    description: str = (
        "Control system power management to prevent or allow sleep/shutdown. "
        "Use 'prevent_sleep' when maintaining long-running connections "
        "(Discord bot, WebSocket clients, background tasks). "
        "Use 'allow_sleep' to restore normal power behavior. "
        "REQUIRES ADMIN PERMISSION - affects system-wide power state."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["prevent_sleep", "allow_sleep", "get_status"],
                "description": "Power operation to perform",
            },
            "keep_display_on": {
                "type": "boolean",
                "description": "For prevent_sleep: keep display awake (default: true)",
                "default": True,
            },
            "use_away_mode": {
                "type": "boolean",
                "description": "For prevent_sleep: use away mode instead of full wake (default: false)",
                "default": False,
            },
        },
        "required": ["operation"],
    }
    
    # High risk level - requires ADMIN permission
    risk_level: str = "critical"
    
    def __init__(self) -> None:
        self._power_mgr: PowerManager = get_power_manager()
    
    async def execute(self, **params: Any) -> ToolResult:
        """
        Execute power management operation.
        
        Args:
            operation: 'prevent_sleep', 'allow_sleep', or 'get_status'
            keep_display_on: Whether to keep display awake (prevent_sleep only)
            use_away_mode: Use away mode instead of full wake (prevent_sleep only)
            
        Returns:
            ToolResult with operation status and current power state
        """
        operation = params.get("operation")
        
        if not operation:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: 'operation'",
            )
        
        try:
            if operation == "prevent_sleep":
                return await self._prevent_sleep(params)
            elif operation == "allow_sleep":
                return await self._allow_sleep()
            elif operation == "get_status":
                return await self._get_status()
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown operation: {operation}",
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Power management error: {str(e)}",
            )
    
    async def _prevent_sleep(self, params: Dict[str, Any]) -> ToolResult:
        """Enable sleep prevention."""
        keep_display = params.get("keep_display_on", True)
        away_mode = params.get("use_away_mode", False)
        
        # Check platform support
        if not self._power_mgr.is_windows:
            return ToolResult(
                success=False,
                output=None,
                error="Power management only available on Windows. "
                      "Linux/Mac would require different implementation "
                      "(systemd-inhibit / caffeinate).",
            )
        
        success = self._power_mgr.prevent_sleep(
            require_display=keep_display,
            use_away_mode=away_mode,
        )
        
        if success:
            state = self._power_mgr.get_state()
            return ToolResult(
                success=True,
                output={
                    "status": "sleep_prevented",
                    "display_kept_awake": state.display_required,
                    "away_mode": state.away_mode,
                    "message": (
                        f"System will not sleep. "
                        f"Display: {'on' if state.display_required else 'can sleep'}. "
                        f"Mode: {'away' if state.away_mode else 'full wake'}."
                    ),
                    "warning": (
                        "Power management active. System will stay awake "
                        "until explicitly allowed to sleep or gateway stops."
                    ),
                },
                error=None,
            )
        else:
            return ToolResult(
                success=False,
                output=None,
                error="Failed to enable sleep prevention. May require elevated privileges.",
            )
    
    async def _allow_sleep(self) -> ToolResult:
        """Restore normal sleep behavior."""
        if not self._power_mgr.is_windows:
            return ToolResult(
                success=False,
                output=None,
                error="Power management only available on Windows.",
            )
        
        success = self._power_mgr.allow_sleep()
        
        if success:
            return ToolResult(
                success=True,
                output={
                    "status": "sleep_allowed",
                    "message": "System can now sleep normally. Power management disabled.",
                },
                error=None,
            )
        else:
            return ToolResult(
                success=False,
                output=None,
                error="Failed to restore normal power management.",
            )
    
    async def _get_status(self) -> ToolResult:
        """Get current power management status."""
        state = self._power_mgr.get_state()
        
        return ToolResult(
            success=True,
            output={
                "is_preventing_sleep": state.is_preventing_sleep,
                "display_required": state.display_required,
                "away_mode": state.away_mode,
                "platform_supported": self._power_mgr.is_windows,
                "message": (
                    f"Power management: {'ACTIVE' if state.is_preventing_sleep else 'inactive'}. "
                    f"System sleep: {'BLOCKED' if state.is_preventing_sleep else 'allowed'}."
                ),
            },
            error=None,
        )
