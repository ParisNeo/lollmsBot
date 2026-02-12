#!/usr/bin/env python
"""
Console Chat Interface for LollmsBot

A rich, interactive terminal interface for chatting with the AI agent directly.
Uses rich for beautiful formatting and questionary for interactive prompts.

Features:
- Beautiful markdown rendering of responses
- Real-time tool execution visualization
- File download notifications
- Conversation history
- Session management
- Command palette for special actions
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    import questionary
    from questionary import Choice
except ImportError:
    print("❌ Install required deps: pip install rich questionary")
    sys.exit(1)

from lollmsbot.agent import Agent, PermissionLevel
from lollmsbot.config import BotConfig, LollmsSettings


@dataclass
class ChatSession:
    """Represents a console chat session."""
    session_id: str
    user_id: str
    started_at: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    tools_used: Set[str] = field(default_factory=set)
    files_generated: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "message_count": self.message_count,
            "tools_used": list(self.tools_used),
            "files_generated": self.files_generated,
        }


class ConsoleChat:
    """
    Rich console interface for direct agent interaction.
    
    Provides a beautiful terminal UI with:
    - Markdown rendering of responses
    - Tool execution visualization
    - File delivery notifications
    - Command palette for special actions
    - Session management
    """
    
    def __init__(
        self,
        agent: Optional[Agent] = None,
        config: Optional[BotConfig] = None,
        verbose: bool = True,
    ) -> None:
        self.console = Console()
        self.config = config or BotConfig.from_env()
        self.agent = agent
        self.verbose = verbose
        
        # Session management
        self.session: Optional[ChatSession] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_history_display = 10
        
        # UI state
        self.show_tool_details = True
        self.show_timestamps = False
        self.markdown_rendering = True
        self.code_theme = "monokai"
        
        # Pending files for download
        self._pending_files: List[Dict[str, Any]] = []
        
        # Command registry
        self._commands: Dict[str, callable] = {
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/history": self._cmd_history,
            "/tools": self._cmd_tools,
            "/skills": self._cmd_skills,
            "/status": self._cmd_status,
            "/settings": self._cmd_settings,
            "/save": self._cmd_save,
            "/exit": self._cmd_exit,
            "/quit": self._cmd_exit,
        }
    
    async def initialize(self) -> bool:
        """Initialize the console chat interface."""
        self.console.print()
        
        # Startup banner
        banner = Panel(
            Text.assemble(
                ("🤖 ", "bold cyan"),
                ("lollmsBot", "bold blue"),
                (" Console Chat\n", "bold white"),
                ("Direct terminal interface to your AI agent", "dim"),
            ),
            box=box.DOUBLE_EDGE,
            border_style="bright_blue",
            padding=(1, 4),
        )
        self.console.print(banner)
        
        # Initialize agent if not provided
        if self.agent is None:
            self.console.print("[yellow]Initializing agent...[/]")
            try:
                self.agent = Agent(
                    config=self.config,
                    name="LollmsBot",
                    default_permissions=PermissionLevel.TOOLS,  # Full tool access in console
                    verbose_logging=self.verbose,
                )
                await self.agent.initialize(gateway_mode="console", host_bindings=["console"])
                self.console.print(f"[green]✅ Agent initialized: {self.agent.name}[/]")
            except Exception as e:
                self.console.print(f"[red]❌ Failed to initialize agent: {e}[/]")
                return False
        
        # Setup file delivery callback
        self.agent.set_file_delivery_callback(self._handle_file_delivery)
        
        # Create session
        import uuid
        session_id = str(uuid.uuid4())[:8]
        self.session = ChatSession(
            session_id=session_id,
            user_id=f"console:{session_id}",
        )
        
        # Welcome message
        self._print_welcome()
        
        return True
    
    def _print_welcome(self) -> None:
        """Print welcome information."""
        welcome_panel = Panel(
            Text.assemble(
                ("Welcome to the lollmsBot console interface!\n\n", "bold"),
                ("• Type your message and press Enter to chat\n", "dim"),
                ("• Use ", "dim"), ("/help", "cyan"), (" for available commands\n", "dim"),
                ("• Files you generate will be saved for download\n", "dim"),
                ("• Press ", "dim"), ("Ctrl+C", "yellow"), (" or type ", "dim"), ("/exit", "cyan"), (" to quit\n", "dim"),
            ),
            title="[bold green]Getting Started[/]",
            border_style="green",
            box=box.ROUNDED,
        )
        self.console.print(welcome_panel)
        self.console.print()
    
    def _print_status_bar(self) -> None:
        """Print a compact status bar."""
        if not self.session:
            return
        
        # Build status components
        status_parts = []
        
        # Agent status
        status_parts.append(f"[cyan]{self.agent.name}[/]")
        
        # Session info
        status_parts.append(f"[dim]msgs:{self.session.message_count}[/]")
        
        # Tools available
        tool_count = len(self.agent.tools) if self.agent else 0
        status_parts.append(f"[dim]tools:{tool_count}[/]")
        
        # Files pending
        if self._pending_files:
            status_parts.append(f"[yellow]📎 {len(self._pending_files)} files[/]")
        
        # Status line
        status_line = " | ".join(status_parts)
        self.console.print(
            Panel(
                status_line,
                box=box.SIMPLE,
                border_style="bright_black",
                padding=(0, 1),
            ),
            style="dim"
        )
    
    async def run(self) -> None:
        """Main chat loop."""
        if not await self.initialize():
            return
        
        try:
            while True:
                self._print_status_bar()
                
                # Get user input with questionary for enhanced experience
                try:
                    user_input = await self._get_input()
                except (EOFError, KeyboardInterrupt):
                    break
                
                if not user_input.strip():
                    continue
                
                # Check for commands
                if user_input.startswith("/"):
                    handled = await self._handle_command(user_input)
                    if handled:
                        continue
                
                # Process message through agent
                await self._process_message(user_input)
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]👋 Interrupted[/]")
        finally:
            await self._cleanup()
    
    async def _get_input(self) -> str:
        """Get user input with rich prompt."""
        # Use questionary for the input to get nice styling
        # But we need to handle this carefully to not break the flow
        
        # For now, use simple prompt with rich styling
        self.console.print("[bold green]You:[/] ", end="")
        try:
            # Read from stdin
            import asyncio
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, input)
            return user_input
        except EOFError:
            raise
    
    async def _handle_command(self, cmd_input: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        parts = cmd_input.strip().split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        handler = self._commands.get(cmd)
        if handler:
            await handler(args)
            return True
        
        self.console.print(f"[red]Unknown command: {cmd}. Type /help for available commands.[/]")
        return True
    
    async def _process_message(self, message: str) -> None:
        """Process a user message through the agent."""
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Update session
        self.session.message_count += 1
        
        # Show thinking indicator
        with self.console.status("[bold cyan]Thinking...[/]", spinner="dots") as status:
            try:
                result = await self.agent.chat(
                    user_id=self.session.user_id,
                    message=message,
                    context={"channel": "console", "session_id": self.session.session_id},
                )
            except Exception as e:
                self.console.print(f"[red]❌ Error: {e}[/]")
                return
        # Display response
        self._display_response(result)
        
        # Update session stats
        if result.get("tools_used"):
            self.session.tools_used.update(result["tools_used"])
        
        if result.get("files_to_send"):
            self.session.files_generated.extend(result["files_to_send"])
    
    def _display_response(self, result: Dict[str, Any]) -> None:
        """Display the agent response beautifully."""
        response = result.get("response", "")
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "tools_used": result.get("tools_used", []),
        })
        
        # Print agent header
        self.console.print()
        self.console.print(f"[bold blue]{self.agent.name}:[/]")
        
        # Display tools used
        tools_used = result.get("tools_used", [])
        if tools_used and self.show_tool_details:
            tools_str = ", ".join(f"[cyan]{t}[/]" for t in tools_used)
            self.console.print(f"[dim]🔧 Used: {tools_str}[/]")
        
        # Display the response with markdown rendering
        if self.markdown_rendering and not self._is_simple_text(response):
            # Use markdown rendering
            md = Markdown(response, code_theme=self.code_theme)
            panel = Panel(
                md,
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2),
            )
            self.console.print(panel)
        else:
            # Simple text display
            self.console.print(Panel(
                response,
                border_style="blue",
                box=box.ROUNDED,
            ))
        
        # Display file notifications
        files = result.get("files_to_send", [])
        if files:
            self._display_file_notification(files)
        
        self.console.print()
    
    def _is_simple_text(self, text: str) -> bool:
        """Check if text is simple (no markdown formatting)."""
        # Quick heuristics
        markdown_indicators = [
            "```", "**", "__", "# ", "## ", "### ",
            "- ", "* ", "1. ", "[", "](", "|",
        ]
        return not any(ind in text for ind in markdown_indicators)
    
    def _display_file_notification(self, files: List[Dict[str, Any]]) -> None:
        """Display file generation notification."""
        self.console.print()
        
        for file_info in files:
            filename = file_info.get("filename", "unnamed")
            file_path = file_info.get("path", "")
            
            # Add to pending files
            file_id = f"file_{len(self._pending_files)}"
            self._pending_files.append({
                "id": file_id,
                "filename": filename,
                "path": file_path,
                "description": file_info.get("description", ""),
            })
            
            # Display notification
            file_panel = Panel(
                Text.assemble(
                    ("📄 ", "yellow"),
                    (filename, "bold white"),
                    ("\n", ""),
                    (file_path, "dim"),
                ),
                title="[bold green]File Generated[/]",
                border_style="green",
                box=box.ROUNDED,
            )
            self.console.print(file_panel)
    
    async def _handle_file_delivery(self, user_id: str, files: List[Dict[str, Any]]) -> bool:
        """Handle file delivery callback from agent."""
        # Store files and they'll be displayed with next response
        # or we can show a notification immediately
        for file_info in files:
            self._pending_files.append({
                "id": f"pending_{len(self._pending_files)}",
                "filename": file_info.get("filename", "unnamed"),
                "path": file_info.get("path", ""),
                "description": file_info.get("description", ""),
            })
        return True
    
    # === Command Handlers ===
    
    async def _cmd_help(self, args: List[str]) -> None:
        """Show help information."""
        help_text = """
[bold cyan]Available Commands:[/]

[bold]/help[/]          Show this help message
[bold]/clear[/]         Clear the screen and history display
[bold]/history[/]       Show conversation history
[bold]/tools[/]         List available tools
[bold]/skills[/]        List and manage skills
[bold]/status[/]        Show agent and session status
[bold]/settings[/]      Configure display settings
[bold]/save[/]          Save conversation to file
[bold]/exit[/] or [bold]/quit[/]  Exit the console chat

[dim]Tip: You can use natural language to ask the agent to perform tasks using tools.[/]
"""
        self.console.print(Panel(help_text, title="Help", border_style="cyan"))
    
    async def _cmd_clear(self, args: List[str]) -> None:
        """Clear the screen."""
        self.console.clear()
        self._print_welcome()
    
    async def _cmd_history(self, args: List[str]) -> None:
        """Show conversation history."""
        if not self.conversation_history:
            self.console.print("[dim]No conversation history yet.[/]")
            return
        
        table = Table(title="Conversation History")
        table.add_column("#", style="dim", justify="right")
        table.add_column("Role", style="cyan")
        table.add_column("Content", style="white", max_width=60)
        table.add_column("Time", style="dim")
        
        # Show last N messages
        start_idx = max(0, len(self.conversation_history) - self.max_history_display)
        for i, msg in enumerate(self.conversation_history[start_idx:], start=start_idx):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
            time_str = msg.get("timestamp", "")[11:16] if self.show_timestamps else ""
            
            role_style = "green" if role == "user" else "blue" if role == "assistant" else "dim"
            table.add_row(str(i + 1), f"[{role_style}]{role}[/{role_style}]", content, time_str)
        
        self.console.print(table)
        self.console.print(f"[dim]Showing {start_idx + 1}-{len(self.conversation_history)} of {len(self.conversation_history)} messages[/]")
    
    async def _cmd_tools(self, args: List[str]) -> None:
        """List available tools."""
        if not self.agent or not self.agent.tools:
            self.console.print("[yellow]No tools available.[/]")
            return
        
        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white", max_width=50)
        table.add_column("Risk", style="yellow")
        
        for name, tool in sorted(self.agent.tools.items()):
            risk_color = {
                "low": "green",
                "medium": "yellow",
                "high": "red",
                "critical": "red bold",
            }.get(tool.risk_level, "white")
            
            desc = tool.description[:45] + "..." if len(tool.description) > 45 else tool.description
            table.add_row(name, desc, f"[{risk_color}]{tool.risk_level}[/{risk_color}]")
        
        self.console.print(table)
        self.console.print(f"[dim]Total: {len(self.agent.tools)} tools[/]")
    
    async def _cmd_skills(self, args: List[str]) -> None:
        """List and manage skills."""
        self.console.print("[yellow]Skills management not yet implemented in console.[/]")
        self.console.print("[dim]Use the web UI or API for full skill management.[/]")
    
    async def _cmd_status(self, args: List[str]) -> None:
        """Show agent and session status."""
        if not self.session:
            return
        
        # Build status tree
        tree = Tree("[bold]Session Status[/]")
        
        # Agent info
        agent_branch = tree.add("[cyan]Agent[/]")
        agent_branch.add(f"Name: {self.agent.name}")
        agent_branch.add(f"State: {self.agent.state.name}")
        agent_branch.add(f"Tools: {len(self.agent.tools)}")
        
        # Session info
        session_branch = tree.add("[cyan]Session[/]")
        session_branch.add(f"ID: {self.session.session_id}")
        session_branch.add(f"Started: {self.session.started_at.strftime('%H:%M:%S')}")
        session_branch.add(f"Messages: {self.session.message_count}")
        
        if self.session.tools_used:
            tools_branch = session_branch.add("Tools used:")
            for tool in sorted(self.session.tools_used):
                tools_branch.add(f"[dim]{tool}[/]")
        
        if self._pending_files:
            files_branch = session_branch.add(f"Files pending: {len(self._pending_files)}")
            for f in self._pending_files:
                files_branch.add(f"[dim]{f['filename']}[/]")
        
        self.console.print(tree)
    
    async def _cmd_settings(self, args: List[str]) -> None:
        """Configure display settings using questionary."""
        self.console.print("[bold]Display Settings[/]")
        
        # Use questionary for interactive settings
        self.markdown_rendering = questionary.confirm(
            "Enable markdown rendering?",
            default=self.markdown_rendering
        ).ask()
        
        self.show_tool_details = questionary.confirm(
            "Show tool execution details?",
            default=self.show_tool_details
        ).ask()
        
        self.show_timestamps = questionary.confirm(
            "Show timestamps in history?",
            default=self.show_timestamps
        ).ask()
        
        theme_choices = ["monokai", "github-dark", "dracula", "solarized-dark", "vscode"]
        self.code_theme = questionary.select(
            "Code highlighting theme:",
            choices=theme_choices,
            default=self.code_theme
        ).ask()
        
        self.console.print("[green]✅ Settings updated[/]")
    
    async def _cmd_save(self, args: List[str]) -> None:
        """Save conversation to file."""
        if not self.conversation_history:
            self.console.print("[yellow]No conversation to save.[/]")
            return
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"lollmsbot_chat_{timestamp}.md"
        
        filename = questionary.text(
            "Save as:",
            default=str(Path.home() / ".lollmsbot" / "chats" / default_name)
        ).ask()
        
        # Ensure directory exists
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Build markdown content
        lines = [f"# lollmsBot Conversation - {self.session.started_at.isoformat()}\n"]
        lines.append(f"**Session:** {self.session.session_id}\n")
        lines.append(f"**Agent:** {self.agent.name}\n\n")
        lines.append("---\n\n")
        
        for msg in self.conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            time_str = msg.get("timestamp", "")[:19] if self.show_timestamps else ""
            
            if role == "user":
                lines.append(f"## User {time_str}\n\n{content}\n\n")
            elif role == "assistant":
                lines.append(f"## {self.agent.name} {time_str}\n\n{content}\n\n")
                # Add tool info if present
                tools = msg.get("tools_used", [])
                if tools:
                    lines.append(f"*Tools: {', '.join(tools)}*\n\n")
        
        # Write file
        filepath.write_text("\n".join(lines), encoding="utf-8")
        self.console.print(f"[green]✅ Conversation saved to {filepath}[/]")
    
    async def _cmd_exit(self, args: List[str]) -> None:
        """Exit the console chat."""
        # Confirm if there's unsaved content
        if self.conversation_history and len(self.conversation_history) > 2:
            save_first = questionary.confirm(
                "Save conversation before exiting?",
                default=False
            ).ask()
            
            if save_first:
                await self._cmd_save([])
        
        raise KeyboardInterrupt("User requested exit")
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        self.console.print()
        
        # Session summary
        if self.session:
            duration = datetime.now() - self.session.started_at
            summary = Panel(
                Text.assemble(
                    ("Session Summary\n\n", "bold"),
                    (f"Duration: {duration.total_seconds():.0f}s\n", "dim"),
                    (f"Messages: {self.session.message_count}\n", "dim"),
                    (f"Tools used: {len(self.session.tools_used)}\n", "dim"),
                    (f"Files generated: {len(self.session.files_generated)}\n", "dim"),
                ),
                title="[bold cyan]Goodbye![/]",
                border_style="cyan",
            )
            self.console.print(summary)
        
        # Close agent resources
        if self.agent:
            await self.agent.close()
        
        self.console.print("[dim]Console chat closed.[/]")


def run_console_chat(
    agent: Optional[Agent] = None,
    config: Optional[BotConfig] = None,
    verbose: bool = True,
) -> None:
    """Entry point for console chat."""
    chat = ConsoleChat(agent=agent, config=config, verbose=verbose)
    
    try:
        asyncio.run(chat.run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        raise


if __name__ == "__main__":
    run_console_chat()
