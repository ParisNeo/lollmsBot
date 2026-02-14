"""
Power Management Module for LollmsBot Gateway

Prevents Windows PC from sleeping or shutting down while the gateway is active.
Provides both automatic gateway-level protection and agent-controllable tool.

Uses Windows SetThreadExecutionState API via ctypes.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from dataclasses import dataclass
from enum import IntFlag
from typing import Optional

logger = logging.getLogger("lollmsbot.power")


class ExecutionState(IntFlag):
    """Windows execution state flags."""
    # Prevent idle sleep
    ES_AWAYMODE_REQUIRED = 0x00000040
    ES_CONTINUOUS = 0x80000000
    ES_DISPLAY_REQUIRED = 0x00000002
    ES_SYSTEM_REQUIRED = 0x00000001
    
    # Common combinations
    AWAKE = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    AWAKE_NO_DISPLAY = ES_CONTINUOUS | ES_SYSTEM_REQUIRED


@dataclass
class PowerState:
    """Current power management state."""
    is_preventing_sleep: bool = False
    display_required: bool = True
    away_mode: bool = False


class PowerManager:
    """
    Manages system power state to prevent sleep/shutdown.
    
    Windows-only implementation using SetThreadExecutionState.
    On non-Windows platforms, logs warnings but doesn't fail.
    """
    
    _instance: Optional[PowerManager] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._state = PowerState()
        self._is_windows = platform.system() == "Windows"
        
        if not self._is_windows:
            logger.warning("Power management only available on Windows. Linux/Mac support would use different APIs.")
    
    def _set_execution_state(self, flags: int) -> bool:
        """Call Windows SetThreadExecutionState API."""
        if not self._is_windows:
            return False
            
        try:
            # kernel32.SetThreadExecutionState
            result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
            return result != 0
        except Exception as e:
            logger.error(f"Failed to set execution state: {e}")
            return False
    
    def prevent_sleep(
        self,
        require_display: bool = True,
        use_away_mode: bool = False,
    ) -> bool:
        """
        Prevent system from sleeping or idle-shutting down.
        
        Args:
            require_display: Keep display on (prevents screen off)
            use_away_mode: Use away mode (allows sleep but processes continue)
            
        Returns:
            True if successful
        """
        if not self._is_windows:
            logger.warning("Cannot prevent sleep: not on Windows")
            return False
        
        # Build flags
        flags = ExecutionState.ES_CONTINUOUS | ExecutionState.ES_SYSTEM_REQUIRED
        
        if require_display:
            flags |= ExecutionState.ES_DISPLAY_REQUIRED
        
        if use_away_mode:
            flags |= ExecutionState.ES_AWAYMODE_REQUIRED
        
        success = self._set_execution_state(flags)
        
        if success:
            self._state.is_preventing_sleep = True
            self._state.display_required = require_display
            self._state.away_mode = use_away_mode
            
            mode_desc = "away mode" if use_away_mode else "full wake"
            display_desc = "display on" if require_display else "display can sleep"
            logger.info(f"🔌 Power management enabled: {mode_desc}, {display_desc}")
        else:
            logger.error("Failed to enable power management")
        
        return success
    
    def allow_sleep(self) -> bool:
        """
        Restore normal power management (allow sleep).
        
        Returns:
            True if successful
        """
        if not self._is_windows:
            return False
        
        # Clear continuous flag to restore default behavior
        success = self._set_execution_state(ExecutionState.ES_CONTINUOUS)
        
        if success:
            self._state.is_preventing_sleep = False
            logger.info("🔌 Power management disabled - system can sleep normally")
        else:
            logger.error("Failed to disable power management")
        
        return success
    
    def get_state(self) -> PowerState:
        """Get current power management state."""
        return PowerState(
            is_preventing_sleep=self._state.is_preventing_sleep,
            display_required=self._state.display_required,
            away_mode=self._state.away_mode,
        )
    
    @property
    def is_windows(self) -> bool:
        """Check if running on Windows."""
        return self._is_windows


def get_power_manager() -> PowerManager:
    """Get or create the singleton PowerManager instance."""
    return PowerManager()
