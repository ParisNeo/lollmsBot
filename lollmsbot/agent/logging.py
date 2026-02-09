"""
Rich console logging and visualization for the Agent.

Provides colored logging, banners, and visual feedback for agent operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style

import logging


class AgentLogger:
    """Rich console logger for agent operations with visual feedback."""
    
    def __init__(self, agent_name: str, agent_id: str, verbose: bool = True) -> None:
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.verbose = verbose
        self._console = Console()
        self._logger = logging.getLogger(__name__)
        
        # Style mapping
        self._styles = {
            "blue": "bold blue",
            "cyan": "bold cyan",
            "green": "bold green",
            "yellow": "bold yellow",
            "orange": "bold rgb(255,165,0)",
            "red": "bold red",
            "magenta": "bold magenta",
            "purple": "bold purple",
            "dim": "dim",
            "white": "white",
            "gold": "bold rgb(255,215,0)",
        }
        
        if verbose:
            self._print_startup()
    
    def _print_startup(self) -> None:
        """Print agent startup banner."""
        panel = Panel.fit(
            f"[bold cyan]🤖 {self.agent_name}[/bold cyan]\n"
            f"[dim]Agent ID: {self.agent_id}[/dim]",
            title="[bold blue]Agent Initialized[/bold blue]",
            border_style="blue"
        )
        self._console.print(panel)
    
    def log(
        self,
        message: str,
        style: str = "white",
        emoji: str = "",
        level: str = "info",
    ) -> None:
        """Log a message with styling."""
        if not self.verbose:
            return
        
        rich_style = self._styles.get(style, style)
        prefix = f"{emoji} " if emoji else ""
        formatted = f"{prefix}[{rich_style}]{message}[/{rich_style}]"
        
        self._console.print(formatted)
        
        # Also log to standard logger
        log_method = getattr(self._logger, level, self._logger.info)
        log_method(message)
    
    def log_command_received(
        self,
        user_id: str,
        message: str,
        context: Optional[Dict[str, Any]],
        tool_count: int,
    ) -> None:
        """Log when a command is received."""
        channel = context.get("channel", "unknown") if context else "unknown"
        msg_preview = message[:100] + "..." if len(message) > 100 else message
        
        panel = Panel(
            f"[bold white]User:[/bold white] [cyan]{user_id}[/cyan]\n"
            f"[bold white]Channel:[/bold white] [yellow]{channel}[/yellow]\n"
            f"[bold white]Message:[/bold white] [white]{msg_preview}[/white]\n"
            f"[dim]Length: {len(message)} chars | Tools available: {tool_count}[/dim]",
            title="[bold blue]📥 COMMAND RECEIVED[/bold blue]",
            border_style="blue"
        )
        self._console.print(panel)
    
    def log_security_check(self, is_safe: bool, event: Optional[Any]) -> None:
        """Log security screening results."""
        if is_safe:
            self.log("✅ Input passed security screening", "green", "🛡️")
        else:
            desc = getattr(event, 'description', 'Unknown violation') if event else 'Unknown'
            self.log(f"🚫 SECURITY BLOCK: {desc}", "red", "🛡️", "warning")
    
    def log_tool_detection(self, tool_count: int, tools_found: List[str]) -> None:
        """Log tool detection in LLM response."""
        if tool_count > 0:
            panel = Panel(
                f"[bold white]Found {tool_count} tool call(s):[/bold white]\n" +
                "\n".join([f"  [purple]• {t}[/purple]" for t in tools_found]),
                title="[bold purple]🔧 TOOLS DETECTED[/bold purple]",
                border_style="purple"
            )
            self._console.print(panel)
        else:
            self.log("No tool calls detected in response", "dim", "🔧")
    
    def log_tool_execution(
        self,
        tool_name: str,
        params: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Log tool execution with parameters."""
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])[:200]
        
        if success:
            self.log(f"✅ Tool '{tool_name}' executed successfully", "green", "🔧")
            self.log(f"   Parameters: {params_str}", "dim")
        else:
            self.log(f"❌ Tool '{tool_name}' failed: {error}", "red", "🔧", "error")
    
    def log_llm_call(self, prompt_length: int, system_prompt: str) -> None:
        """Log LLM invocation."""
        sys_preview = system_prompt[:80] + "..." if len(system_prompt) > 80 else system_prompt
        
        panel = Panel(
            f"[bold white]Prompt length:[/bold white] [cyan]{prompt_length}[/cyan] chars\n"
            f"[bold white]System prompt:[/bold white] [dim]{sys_preview}[/dim]",
            title="[bold orange]🧠 LLM CALL[/bold orange]",
            border_style="rgb(255,165,0)"
        )
        self._console.print(panel)
    
    def log_llm_response(self, response_length: int, has_tools: bool) -> None:
        """Log LLM response received."""
        tool_status = "🟢 contains tools" if has_tools else "🔵 text only"
        self.log(f"📤 LLM response: {response_length} chars ({tool_status})", "orange")
    
    def log_file_generation(self, file_count: int, filenames: List[str]) -> None:
        """Log file generation events."""
        if file_count > 0:
            panel = Panel(
                f"[bold white]Generated {file_count} file(s):[/bold white]\n" +
                "\n".join([f"  [green]📄 {name}[/green]" for name in filenames]),
                title="[bold green]📦 FILES CREATED[/bold green]",
                border_style="green"
            )
            self._console.print(panel)
    
    def log_state_change(self, old_state: str, new_state: str, reason: str = "") -> None:
        """Log agent state transitions."""
        reason_str = f" ({reason})" if reason else ""
        self.log(f"🔄 State: {old_state} → {new_state}{reason_str}", "yellow", "ℹ️")
    
    def log_response_sent(
        self,
        user_id: str,
        response_length: int,
        tools_used: List[str],
    ) -> None:
        """Log final response delivery."""
        tools_str = f" | Tools: {', '.join(tools_used)}" if tools_used else " | No tools"
        self.log(f"📤 Response sent to {user_id}: {response_length} chars{tools_str}", "green", "✅")
    
    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log errors with optional exception details."""
        exc_str = f": {str(exception)}" if exception else ""
        self.log(f"💥 ERROR: {message}{exc_str}", "red", "❌", "error")
    
    def log_critical(self, message: str) -> None:
        """Log critical security/system events."""
        self.log(message, "red", "🚨", "critical")
    
    def log_skill_execution(self, skill_name: str, success: bool) -> None:
        """Log skill execution events."""
        if success:
            self.log(f"🎯 Skill '{skill_name}' executed successfully", "magenta", "📚")
        else:
            self.log(f"❌ Skill '{skill_name}' execution failed", "red", "📚", "error")
    
    def log_important_memory(self, category: str, facts: Dict[str, Any], user_id: str) -> None:
        """Log important memory detection and storage."""
        facts_str = ", ".join([f"{k}={v}" for k, v in facts.items() if v])
        
        panel = Panel(
            f"[bold gold]🧠 IMPORTANT MEMORY DETECTED[/bold gold]\n"
            f"[bold white]Category:[/bold white] [cyan]{category}[/cyan]\n"
            f"[bold white]User:[/bold white] [yellow]{user_id}[/yellow]\n"
            f"[bold white]Extracted:[/bold white] [green]{facts_str}[/green]",
            title="[bold gold]⭐ HIGH-VALUE INFORMATION STORED[/bold gold]",
            border_style="gold"
        )
        self._console.print(panel)
        
        # Also log to audit trail
        self.log(f"IMPORTANT MEMORY [{category}] from {user_id}: {facts_str}", "gold", "🧠", "audit")
