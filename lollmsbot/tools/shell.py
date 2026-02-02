"""
Shell tool for LollmsBot.

This module provides the ShellTool class for safe shell command execution
with strict security controls including allowlist/denylist filtering,
timeout protection, and comprehensive output capture.

SECURITY WARNING:
This tool executes system shell commands which can be dangerous. It includes
multiple security layers to mitigate risks, but should still be used with
caution. Always review the allowlist/denylist configuration carefully.

Security features:
- Explicit allowlist for permitted commands (opt-in security)
- Denylist for known dangerous commands and patterns
- Timeout protection to prevent hanging processes
- Working directory restriction
- No shell=True to prevent injection attacks
- Command argument validation
"""

import asyncio
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Pattern, Set, Union

from lollmsbot.agent import Tool, ToolResult


@dataclass
class SecurityPolicy:
    """Security configuration for shell command execution.
    
    Attributes:
        allowed_commands: Set of explicitly allowed command names (empty = all allowed).
        denied_commands: Set of denied command names/patterns.
        denied_patterns: List of regex patterns that will reject commands.
        max_timeout: Maximum execution timeout in seconds.
        allowed_working_dirs: Set of directories where commands can execute.
        max_output_size: Maximum output size in bytes to prevent memory issues.
    """
    allowed_commands: Set[str] = field(default_factory=set)
    denied_commands: Set[str] = field(default_factory=lambda: {
        "rm", "del", "format", "mkfs", "dd", "shred", "wipe",
        "chmod", "chown", "sudo", "su", "passwd", "shadow",
        "nc", "netcat", "ncat", "telnet",
        "bash", "sh", "zsh", "fish", "cmd", "powershell", "pwsh",
        "python", "python3", "perl", "ruby", "node", "php",
        "wget", "curl", "fetch", "axel",
        "ssh", "scp", "sftp", "ftp", "rsync",
        "systemctl", "service", "init", "reboot", "shutdown", "halt",
        "kill", "killall", "pkill", "xkill",
        "iptables", "ufw", "firewalld",
        "useradd", "userdel", "groupadd", "groupdel",
        "mount", "umount", "losetup", "modprobe",
    })
    denied_patterns: List[Pattern[str]] = field(default_factory=lambda: [
        re.compile(r"[;&|]\s*(?:rm|del|format|mkfs|dd|chmod|chown|sudo)\b"),  # Command chaining
        re.compile(r"`.*?`"),  # Backtick substitution
        re.compile(r"\$\(.*?\)"),  # Command substitution
        re.compile(r"[><|]\s*/(?:etc|bin|sbin|usr|var|root|home|proc|sys|dev)"),  # Redirection to system paths
        re.compile(r"-[a-zA-Z]*[rf]"),  # Force/recursive flags often used destructively
        re.compile(r"\.\./\.\."),  # Path traversal attempts
        re.compile(r"(?:https?|ftp|file|data):[/\\]{2}"),  # URL-like patterns in commands
    ])
    max_timeout: float = 30.0
    allowed_working_dirs: Set[Path] = field(default_factory=lambda: {Path.cwd()})
    max_output_size: int = 1024 * 1024  # 1 MB


class ShellTool(Tool):
    """Tool for safe shell command execution with strict security controls.
    
    This tool provides controlled shell command execution with multiple
    security layers including allowlist/denylist filtering, timeout
    protection, and output capture. Commands are executed without shell
    interpolation to prevent injection attacks.
    
    SECURITY WARNING:
    - Only commands in the allowlist are permitted (if configured)
    - Commands in the denylist are always rejected
    - Commands matching denied patterns are rejected
    - Timeout prevents runaway processes
    - Working directory is restricted
    
    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema describing expected parameters.
        security: SecurityPolicy instance controlling command validation.
    """
    
    name: str = "shell"
    description: str = (
        "Execute safe shell commands with strict security controls. "
        "Commands are validated against allowlist/denylist, executed "
        "with timeout protection, and return stdout, stderr, and return code. "
        "Use with caution - only pre-approved commands are allowed."
    )
    
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["execute", "check_allowed"],
                "description": "Operation to perform",
            },
            "command": {
                "type": "string",
                "description": "Shell command to execute (for execute operation)",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, max 300)",
                "minimum": 1,
                "maximum": 300,
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for command execution (optional)",
            },
            "env_vars": {
                "type": "object",
                "description": "Environment variables to set (optional)",
            },
        },
        "required": ["operation"],
    }
    
    def __init__(
        self,
        security: Optional[SecurityPolicy] = None,
        default_timeout: float = 30.0,
    ) -> None:
        """Initialize the ShellTool.
        
        Args:
            security: SecurityPolicy for command validation. Uses defaults if None.
            default_timeout: Default timeout for command execution in seconds.
        """
        self.security: SecurityPolicy = security or SecurityPolicy()
        self.default_timeout: float = min(default_timeout, 300.0)  # Cap at 5 minutes
        
        # Compile any additional patterns if provided as strings
        self._ensure_patterns_compiled()
    
    def _ensure_patterns_compiled(self) -> None:
        """Ensure all denial patterns are compiled regex objects."""
        compiled_patterns: List[Pattern[str]] = []
        for pattern in self.security.denied_patterns:
            if isinstance(pattern, str):
                compiled_patterns.append(re.compile(pattern))
            else:
                compiled_patterns.append(pattern)
        self.security.denied_patterns = compiled_patterns
    
    def check_command_allowed(self, command: str) -> tuple[bool, Optional[str]]:
        """Check if a command is allowed under current security policy.
        
        Performs multiple security checks:
        1. Empty/whitespace check
        2. Denied pattern matching
        3. Explicit denylist check
        4. Explicit allowlist check (if configured)
        5. Basic injection attempt detection
        
        Args:
            command: The command string to validate.
            
        Returns:
            Tuple of (is_allowed, reason_if_denied). reason is None if allowed.
        """
        # Check for empty command
        stripped = command.strip()
        if not stripped:
            return False, "Empty command not allowed"
        
        # Check for denied patterns (injection attempts, dangerous sequences)
        for pattern in self.security.denied_patterns:
            if pattern.search(stripped):
                return False, f"Command matches denied security pattern: {pattern.pattern}"
        
        # Parse command to get base command name
        try:
            # Use shlex to properly parse without executing
            parsed = shlex.split(stripped)
            if not parsed:
                return False, "Could not parse command"
            
            base_command = parsed[0]
            
            # Remove path prefix if present to get command name
            base_name = Path(base_command).name
            
        except ValueError as exc:
            return False, f"Command parsing error: {str(exc)}"
        
        # Check explicit denylist
        if base_name in self.security.denied_commands:
            return False, f"Command '{base_name}' is in denylist"
        
        # Check if base command is in allowed list (if allowlist is configured)
        if self.security.allowed_commands:
            # Check both full path and base name
            allowed = base_name in self.security.allowed_commands or base_command in self.security.allowed_commands
            if not allowed:
                allowed_list = ", ".join(sorted(self.security.allowed_commands))
                return False, f"Command '{base_name}' not in allowlist. Allowed: {allowed_list}"
        
        # Check for suspicious characters that might indicate injection
        dangerous_chars = [";", "|", "&", "$", "`", "<", ">"]
        for char in dangerous_chars:
            if char in stripped:
                # These are allowed if properly quoted, but flag for review
                # Actually reject for maximum safety
                return False, f"Command contains potentially dangerous character: '{char}'. Use tool parameters instead of shell operators."
        
        return True, None
    
    def _validate_working_directory(self, working_dir: Optional[str]) -> tuple[Path, Optional[str]]:
        """Validate and resolve working directory.
        
        Args:
            working_dir: Requested working directory or None for default.
            
        Returns:
            Tuple of (resolved_path, error_message). error is None if valid.
        """
        if working_dir is None:
            # Use first allowed directory as default
            return next(iter(self.security.allowed_working_dirs)), None
        
        try:
            requested = Path(working_dir).resolve()
            
            # Check if within allowed directories
            for allowed in self.security.allowed_working_dirs:
                try:
                    requested.relative_to(allowed)
                    return requested, None
                except ValueError:
                    continue
            
            allowed_strs = [str(d) for d in self.security.allowed_working_dirs]
            return Path.cwd(), f"Working directory '{working_dir}' outside allowed paths: {', '.join(allowed_strs)}"
            
        except (OSError, ValueError) as exc:
            return Path.cwd(), f"Invalid working directory '{working_dir}': {str(exc)}"
    
    async def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
    ) -> ToolResult:
        """Execute a shell command with security checks and timeout protection.
        
        SECURITY WARNING: This method executes system commands. All inputs
        are validated against the security policy before execution.
        
        Args:
            command: The command string to execute.
            timeout: Maximum execution time in seconds. Uses default if None.
            working_dir: Working directory for execution. Must be in allowed list.
            env_vars: Additional environment variables for the process.
            
        Returns:
            ToolResult with stdout, stderr, return code, and execution metadata.
        """
        # Validate command against security policy
        allowed, reason = self.check_command_allowed(command)
        if not allowed:
            return ToolResult(
                success=False,
                output=None,
                error=f"Security check failed: {reason}",
            )
        
        # Validate working directory
        work_dir, dir_error = self._validate_working_directory(working_dir)
        if dir_error:
            return ToolResult(
                success=False,
                output=None,
                error=f"Directory validation failed: {dir_error}",
            )
        
        # Validate timeout
        exec_timeout = min(timeout or self.default_timeout, 300.0)
        
        # Prepare environment
        process_env: Optional[dict[str, str]] = None
        if env_vars:
            import os
            process_env = {**os.environ, **env_vars}
        
        # Parse command safely using shlex
        try:
            cmd_args = shlex.split(command)
        except ValueError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to parse command: {str(exc)}",
            )
        
        # Execute with timeout using asyncio
        import time
        start_time = time.time()
        
        try:
            # Create subprocess without shell=True for security
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=process_env,
            )
            
            # Wait for completion with timeout
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=exec_timeout,
                )
            except asyncio.TimeoutError:
                # Kill the process if it times out
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass  # Already exited
                
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Command timed out after {exec_timeout} seconds",
                    execution_time=time.time() - start_time,
                )
            
            execution_time = time.time() - start_time
            
            # Decode output with size limit
            def decode_limited(data: bytes, max_size: int) -> str:
                if len(data) > max_size:
                    truncated = data[:max_size]
                    decoded = truncated.decode("utf-8", errors="replace")
                    return decoded + f"\n[TRUNCATED: {len(data) - max_size} bytes omitted]"
                return data.decode("utf-8", errors="replace")
            
            max_size = self.security.max_output_size
            stdout = decode_limited(stdout_bytes, max_size // 2)
            stderr = decode_limited(stderr_bytes, max_size // 2)
            
            # Build result
            result_data = {
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_bytes": len(stdout_bytes),
                "stderr_bytes": len(stderr_bytes),
                "execution_time": execution_time,
                "working_directory": str(work_dir),
            }
            
            # Consider non-zero return code as failure
            success = process.returncode == 0
            
            return ToolResult(
                success=success,
                output=result_data,
                error=stderr if not success and stderr else None,
                execution_time=execution_time,
            )
            
        except FileNotFoundError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Command not found: {exc.filename}",
                execution_time=time.time() - start_time,
            )
        except PermissionError as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied executing command: {str(exc)}",
                execution_time=time.time() - start_time,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                error=f"Execution error: {str(exc)}",
                execution_time=time.time() - start_time,
            )
    
    async def check_allowed(self, command: str) -> ToolResult:
        """Check if a command would be allowed without executing it.
        
        Args:
            command: The command string to check.
            
        Returns:
            ToolResult with check results and security policy info.
        """
        allowed, reason = self.check_command_allowed(command)
        
        result = {
            "command": command,
            "allowed": allowed,
            "reason": reason,
            "security_policy": {
                "allowlist_enabled": bool(self.security.allowed_commands),
                "allowed_commands_count": len(self.security.allowed_commands),
                "denied_commands_count": len(self.security.denied_commands),
                "denied_patterns_count": len(self.security.denied_patterns),
                "max_timeout": self.security.max_timeout,
                "allowed_working_dirs": [str(d) for d in self.security.allowed_working_dirs],
            },
        }
        
        return ToolResult(
            success=allowed,
            output=result,
            error=reason if not allowed else None,
        )
    
    async def execute_tool(self, **params: Any) -> ToolResult:
        """Execute shell tool operation based on parameters.
        
        Main entry point for Tool base class. Dispatches to appropriate
        method based on the 'operation' parameter.
        
        Args:
            **params: Parameters must include:
                - operation: 'execute' or 'check_allowed'
                - command: Required for both operations
                - timeout: Optional for execute
                - working_dir: Optional for execute
                - env_vars: Optional for execute
                
        Returns:
            ToolResult from the executed operation.
        """
        operation = params.get("operation")
        
        if not operation:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: 'operation'",
            )
        
        command = params.get("command")
        if not command:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: 'command'",
            )
        
        if operation == "execute":
            return await self.execute(
                command=command,
                timeout=params.get("timeout"),
                working_dir=params.get("working_dir"),
                env_vars=params.get("env_vars"),
            )
        
        elif operation == "check_allowed":
            return await self.check_allowed(command)
        
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown operation: '{operation}'. Valid operations: execute, check_allowed",
            )
    
    # Alias for Tool base class compatibility
    async def execute(self, **params: Any) -> ToolResult:
        """Compatibility method for Tool base class.
        
        Delegates to execute_tool for actual implementation.
        """
        return await self.execute_tool(**params)
    
    def __repr__(self) -> str:
        return (
            f"ShellTool(allowed={len(self.security.allowed_commands)}, "
            f"denied={len(self.security.denied_commands)}, "
            f"timeout={self.default_timeout})"
        )