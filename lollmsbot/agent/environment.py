"""
Environment detection module for LollmsBot.

Gathers runtime environment information including:
- Operating system details (platform, version, release)
- Container/virtualization detection (Docker, WSL, VM)
- Python environment (version, venv, packages)
- File system context (working directories, paths)
- Network configuration (bindings, ports)
- Hardware hints (CPU, memory if available)

This information is stored as self-knowledge and made available
to the Soul for accurate environmental awareness.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Comprehensive runtime environment information."""
    
    # OS Information
    os_platform: str = ""  # 'windows', 'linux', 'darwin', etc.
    os_system: str = ""  # 'Windows', 'Linux', 'Darwin'
    os_release: str = ""  # '10', '22.04', etc.
    os_version: str = ""  # Detailed version string
    os_machine: str = ""  # 'AMD64', 'x86_64', 'arm64', etc.
    
    # Python Environment
    python_version: str = ""
    python_implementation: str = ""  # 'CPython', 'PyPy', etc.
    python_executable: str = ""
    in_virtualenv: bool = False
    virtualenv_path: Optional[str] = None
    
    # Container/Virtualization
    in_docker: bool = False
    in_wsl: bool = False
    container_id: Optional[str] = None
    
    # File System Context
    working_directory: str = ""
    home_directory: str = ""
    lollmsbot_data_dir: str = ""
    lollmsbot_config_dir: str = ""
    
    # Network/Runtime
    hostname: str = ""
    host_bindings: List[str] = field(default_factory=list)  # e.g., ["127.0.0.1:8800"]
    gateway_mode: str = "unknown"  # 'standalone', 'discord', 'telegram', 'http_api', etc.
    
    # Hardware Hints (if available)
    cpu_count: Optional[int] = None
    memory_info: Optional[str] = None  # May be limited by container
    
    # Process Context
    process_id: int = 0
    process_args: List[str] = field(default_factory=list)
    
    def to_facts(self) -> List[Tuple[str, str, float]]:
        """
        Convert environment info to (fact_id, content, importance) tuples.
        
        Importance: 7-9 for critical env facts, 5-6 for nice-to-know
        """
        facts = []
        
        # Critical: OS platform (affects paths, tools, behavior)
        facts.append((
            "os_platform",
            f"I am running on {self.os_system} {self.os_release} ({self.os_machine})",
            8.0
        ))
        
        # Critical: Python environment (affects capabilities)
        venv_status = "inside a virtual environment" if self.in_virtualenv else "system Python"
        facts.append((
            "python_environment",
            f"Python {self.python_version} ({self.python_implementation}), running {venv_status}",
            7.5
        ))
        
        # Important: Container context (affects file system, networking)
        if self.in_docker:
            container_msg = "I am running inside a Docker container"
            if self.container_id:
                container_msg += f" (ID: {self.container_id[:12]})"
            facts.append(("container_context", container_msg, 8.5))
        elif self.in_wsl:
            facts.append(("container_context", "I am running in Windows Subsystem for Linux (WSL)", 8.0))
        else:
            facts.append(("container_context", "I am running directly on bare metal or a VM", 6.0))
        
        # Important: File system paths (affects where I can read/write)
        facts.append((
            "filesystem_context",
            f"My working directory is {self.working_directory}. "
            f"My data is stored at {self.lollmsbot_data_dir}. "
            f"My configuration is at {self.lollmsbot_config_dir}",
            7.5
        ))
        
        # Network bindings
        if self.host_bindings:
            bindings_str = ", ".join(self.host_bindings)
            facts.append((
                "network_bindings",
                f"I am accessible at: {bindings_str}. My hostname is {self.hostname}",
                7.0
            ))
        
        # Gateway/channel mode
        if self.gateway_mode != "unknown":
            facts.append((
                "runtime_mode",
                f"My current runtime mode is: {self.gateway_mode}",
                6.5
            ))
        
        # Hardware (lower importance - often restricted)
        if self.cpu_count:
            facts.append((
                "hardware_hints",
                f"Available CPUs: {self.cpu_count}",
                5.0
            ))
        
        return facts


class EnvironmentDetector:
    """Detects and gathers runtime environment information."""
    
    def __init__(self) -> None:
        self._info: Optional[EnvironmentInfo] = None
    
    def detect(self, gateway_mode: str = "unknown", host_bindings: Optional[List[str]] = None) -> EnvironmentInfo:
        """
        Perform comprehensive environment detection.
        
        Args:
            gateway_mode: How the agent is running ('discord', 'telegram', 'standalone', etc.)
            host_bindings: List of host:port bindings if known
        """
        info = EnvironmentInfo()
        
        # OS Detection
        info.os_platform = sys.platform  # 'win32', 'linux', 'darwin'
        info.os_system = platform.system()  # 'Windows', 'Linux', 'Darwin'
        info.os_release = platform.release()
        info.os_version = platform.version()
        info.os_machine = platform.machine()
        
        # Python Environment
        info.python_version = platform.python_version()
        info.python_implementation = platform.python_implementation()
        info.python_executable = sys.executable
        info.in_virtualenv = self._detect_virtualenv()
        if info.in_virtualenv:
            info.virtualenv_path = self._get_virtualenv_path()
        
        # Container/Virtualization
        info.in_docker = self._detect_docker()
        info.in_wsl = self._detect_wsl()
        if info.in_docker:
            info.container_id = self._get_container_id()
        
        # File System Context
        info.working_directory = str(Path.cwd())
        info.home_directory = str(Path.home())
        info.lollmsbot_data_dir = str(Path.home() / ".lollmsbot" / "data")
        info.lollmsbot_config_dir = str(Path.home() / ".lollmsbot")
        
        # Network/Runtime
        info.hostname = platform.node()
        info.gateway_mode = gateway_mode
        if host_bindings:
            info.host_bindings = host_bindings
        
        # Hardware Hints
        try:
            import os
            info.cpu_count = os.cpu_count()
        except Exception:
            pass
        
        # Process Context
        info.process_id = os.getpid()
        info.process_args = sys.argv
        
        self._info = info
        logger.info(f"Environment detected: {info.os_system} {info.os_release}, "
                   f"Docker={info.in_docker}, WSL={info.in_wsl}, "
                   f"venv={info.in_virtualenv}, mode={gateway_mode}")
        
        return info
    
    def _detect_virtualenv(self) -> bool:
        """Detect if running in a Python virtual environment."""
        # Check standard venv/virtualenv indicators
        if hasattr(sys, 'real_prefix'):
            return True  # Old virtualenv
        if sys.base_exec_prefix != sys.exec_prefix:
            return True  # Modern venv
        if 'VIRTUAL_ENV' in os.environ:
            return True  # Active venv
        return False
    
    def _get_virtualenv_path(self) -> Optional[str]:
        """Get the virtual environment path if active."""
        if 'VIRTUAL_ENV' in os.environ:
            return os.environ['VIRTUAL_ENV']
        
        # Try to infer from executable path
        exe_path = Path(sys.executable)
        if exe_path.parts[-3:-1] == ('bin', 'Scripts') or '.venv' in str(exe_path):
            return str(exe_path.parent.parent)
        
        return None
    
    def _detect_docker(self) -> bool:
        """Detect if running inside a Docker container."""
        # Method 1: Check for .dockerenv file
        if Path('/.dockerenv').exists():
            return True
        
        # Method 2: Check cgroup for docker references
        try:
            cgroup_path = Path('/proc/self/cgroup')
            if cgroup_path.exists():
                content = cgroup_path.read_text()
                if 'docker' in content or 'containerd' in content:
                    return True
        except Exception:
            pass
        
        # Method 3: Check environment variables
        if 'DOCKER_CONTAINER' in os.environ:
            return True
        
        return False
    
    def _get_container_id(self) -> Optional[str]:
        """Get Docker container ID if in a container."""
        try:
            # Read from cgroup
            cgroup_path = Path('/proc/self/cgroup')
            if cgroup_path.exists():
                content = cgroup_path.read_text()
                # Extract container ID from docker cgroup path
                for line in content.split('\n'):
                    if 'docker' in line:
                        parts = line.split('/')
                        for part in parts:
                            if len(part) == 64:  # Full container ID
                                return part
                            if len(part) == 12:  # Short ID
                                return part
        except Exception:
            pass
        
        # Try hostname (often set to container ID)
        hostname = platform.node()
        if len(hostname) == 12 and all(c in '0123456789abcdef' for c in hostname):
            return hostname
        
        return None
    
    def _detect_wsl(self) -> bool:
        """Detect if running in Windows Subsystem for Linux."""
        # Method 1: Check /proc/version for 'Microsoft' or 'WSL'
        try:
            version_path = Path('/proc/version')
            if version_path.exists():
                version = version_path.read_text().lower()
                if 'microsoft' in version or 'wsl' in version:
                    return True
        except Exception:
            pass
        
        # Method 2: Check environment variables
        if 'WSL_DISTRO_NAME' in os.environ or 'WSLENV' in os.environ:
            return True
        
        # Method 3: Check for Windows interop
        if Path('/mnt/c/Windows').exists():
            return True
        
        return False
    
    def get_summary(self) -> str:
        """Get human-readable environment summary."""
        if not self._info:
            return "Environment not yet detected"
        
        info = self._info
        
        parts = [
            f"Platform: {info.os_system} {info.os_release} ({info.os_machine})",
            f"Python: {info.python_version} ({info.python_implementation})",
        ]
        
        if info.in_virtualenv:
            parts.append(f"Virtualenv: Yes ({info.virtualenv_path or 'unknown path'})")
        
        if info.in_docker:
            container_msg = "Docker container"
            if info.container_id:
                container_msg += f" ({info.container_id[:12]})"
            parts.append(f"Container: {container_msg}")
        elif info.in_wsl:
            parts.append("Container: WSL (Windows Subsystem for Linux)")
        
        parts.append(f"Working directory: {info.working_directory}")
        parts.append(f"Data directory: {info.lollmsbot_data_dir}")
        
        if info.host_bindings:
            parts.append(f"Network bindings: {', '.join(info.host_bindings)}")
        
        parts.append(f"Runtime mode: {info.gateway_mode}")
        
        return " | ".join(parts)


# Singleton instance
_detector_instance: Optional[EnvironmentDetector] = None

def get_environment_detector() -> EnvironmentDetector:
    """Get or create singleton environment detector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EnvironmentDetector()
    return _detector_instance


def detect_environment(gateway_mode: str = "unknown", host_bindings: Optional[List[str]] = None) -> EnvironmentInfo:
    """Convenience function to run environment detection."""
    return get_environment_detector().detect(gateway_mode, host_bindings)
