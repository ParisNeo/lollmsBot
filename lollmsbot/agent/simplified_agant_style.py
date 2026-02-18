"""
SimplifiedAgant-Style Module - minimal agent architecture for lollmsbot.

Implements key concepts:
- Minimal 4-tool core (read, write, edit, bash)
- Self-written code extensions
- Hot reloading
- Session branching
- Skills vs Tools distinction

Philosophy: "The agent writes its own extensions through code"
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


class ExtensionStatus(Enum):
    """Status of a self-written extension."""
    DRAFT = auto()      # Just created, not tested
    ACTIVE = auto()     # Loaded and working
    DEPRECATED = auto() # Replaced by newer version
    BROKEN = auto()     # Failed to load or execute


@dataclass
class CodeExtension:
    """
    A self-written extension created by the agent.
    
    OpenClaw-style: The agent writes Python code to add functionality.
    Extensions are stored as Python modules and can be hot-reloaded.
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    code: str = ""  # The actual Python code
    dependencies: List[str] = field(default_factory=list)
    
    # Runtime state
    status: ExtensionStatus = ExtensionStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    execution_count: int = 0
    
    # Compiled module (transient)
    _compiled_module: Optional[Any] = field(default=None, repr=False)
    _main_function: Optional[Callable] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "code_preview": self.code[:200] + "..." if len(self.code) > 200 else self.code,
            "dependencies": self.dependencies,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "execution_count": self.execution_count,
        }


@dataclass
class SessionBranch:
    """
    A branch in the session tree for experimentation.
    
    OpenClaw feature: Tree-structured sessions allow trying things
    without polluting the main conversation context.
    """
    branch_id: str
    parent_branch_id: Optional[str] = None
    summary: str = ""  # What this branch is about
    created_at: datetime = field(default_factory=datetime.now)
    
    # Content
    messages: List[Dict[str, Any]] = field(default_factory=list)
    extensions_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    
    # State
    is_active: bool = False
    is_merged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.messages),
            "extensions_created": self.extensions_created,
            "files_modified": self.files_modified,
            "is_active": self.is_active,
            "is_merged": self.is_merged,
        }


class MinimalToolSet:
    """
    OpenClaw's minimal 4-tool philosophy.
    
    Instead of many specialized tools, provide just the essentials
    that can be composed to achieve any task.
    """
    
    TOOLS = {
        "read": {
            "description": "Read content from files or URLs",
            "parameters": {
                "source": "string - file path or URL to read",
                "limit": "optional int - max lines/chars to read",
            },
            "example": '<tool>read</tool><source>/path/to/file.txt</source>',
        },
        "write": {
            "description": "Write content to a file (creates or overwrites)",
            "parameters": {
                "path": "string - file path to write",
                "content": "string - content to write",
            },
            "example": '<tool>write</tool><path>/tmp/output.txt</path><content>Hello World</content>',
        },
        "edit": {
            "description": "Edit existing file content (find/replace)",
            "parameters": {
                "path": "string - file to edit",
                "old_string": "string - text to find",
                "new_string": "string - text to replace with",
            },
            "example": '<tool>edit</tool><path>file.py</path><old_string>print("old")</old_string><new_string>print("new")</new_string>',
        },
        "bash": {
            "description": "Execute shell commands",
            "parameters": {
                "command": "string - shell command to run",
                "working_dir": "optional string - directory to run in",
            },
            "example": '<tool>bash</tool><command>ls -la</command>',
        },
    }
    
    @classmethod
    def get_tool_instructions(cls) -> str:
        """Generate instructions for the minimal tool set."""
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║  MINIMAL TOOL SET - OpenClaw Style                              ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            "",
            "You have exactly 4 tools. All tasks must be accomplished",
            "by composing these 4 primitives:",
            "",
        ]
        
        for name, info in cls.TOOLS.items():
            lines.extend([
                f"  ▶ {name.upper()}",
                f"    {info['description']}",
                f"    Parameters: {', '.join(info['parameters'].keys())}",
                f"    Example: {info['example']}",
                "",
            ])
        
        lines.extend([
            "╔══════════════════════════════════════════════════════════════════╗",
            "║  EXTENSION PRINCIPLE                                             ║",
            "╠══════════════════════════════════════════════════════════════════╣",
            "",
            "Need more capability? Write an extension:",
            "",
            "  <tool>write</tool>",
            "  <path>~/.lollmsbot/extensions/my_tool.py</path>",
            "  <content>",
            "  def execute(args):",
            "      # Your implementation",
            "      return result",
            "  </content>",
            "",
            "Then reload extensions with: /reload",
            "",
            "This is how OpenClaw works - you write your own tools!",
            "",
        ])
        
        return "\n".join(lines)


class SimplifiedAgantStyle:
    """
    Main class implementing OpenClaw-style architecture.
    
    Features:
    - Minimal 4-tool mode
    - Self-written extensions
    - Session branching
    - Hot reloading
    - TUI components
    """
    
    def __init__(
        self,
        agent: Any,
        extensions_dir: Optional[Path] = None,
        enable_tui: bool = True,
    ) -> None:
        self.agent = agent
        self.extensions_dir = extensions_dir or (Path.home() / ".lollmsbot" / "extensions")
        self.extensions_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_tui = enable_tui and RICH_AVAILABLE
        self.console = Console() if self.enable_tui else None
        
        # Extensions registry
        self.extensions: Dict[str, CodeExtension] = {}
        self._extension_lock = threading.Lock()
        
        # Session branching
        self.branches: Dict[str, SessionBranch] = {}
        self._active_branch_id: Optional[str] = None
        self._main_branch_id: Optional[str] = None
        
        # TUI state
        self._progress_bars: Dict[str, Any] = {}
        self._spinners: Dict[str, Any] = {}
        
        # Load existing extensions
        self._load_existing_extensions()
        
        logger.info(f"SimplifiedAgantStyle initialized: extensions_dir={self.extensions_dir}")
    
    def _load_existing_extensions(self) -> None:
        """Load extensions from disk."""
        if not self.extensions_dir.exists():
            return
        
        for ext_file in self.extensions_dir.glob("*.py"):
            try:
                self._load_extension_from_file(ext_file)
            except Exception as e:
                logger.warning(f"Failed to load extension {ext_file}: {e}")
    
    def _load_extension_from_file(self, ext_file: Path) -> Optional[CodeExtension]:
        """Load a single extension from Python file."""
        name = ext_file.stem
        
        try:
            code = ext_file.read_text(encoding='utf-8')
            
            # Parse metadata from docstring/comments
            description = ""
            version = "1.0.0"
            
            # Extract description from docstring
            docstring_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
            if docstring_match:
                description = docstring_match.group(1).strip()[:200]
            
            # Extract version from __version__ or comment
            version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', code)
            if version_match:
                version = version_match.group(1)
            
            ext = CodeExtension(
                name=name,
                version=version,
                description=description,
                code=code,
                status=ExtensionStatus.ACTIVE,
            )
            
            # Try to compile
            if self._compile_extension(ext):
                with self._extension_lock:
                    self.extensions[name] = ext
                logger.info(f"Loaded extension: {name} v{version}")
                return ext
            else:
                ext.status = ExtensionStatus.BROKEN
                with self._extension_lock:
                    self.extensions[name] = ext
                return ext
                
        except Exception as e:
            logger.error(f"Failed to load extension {name}: {e}")
            return None
    
    def _compile_extension(self, ext: CodeExtension) -> bool:
        """Compile extension code and extract main function."""
        try:
            # Compile the code
            compiled = compile(ext.code, f"<extension_{ext.name}>", 'exec')
            
            # Create module namespace
            module_namespace = {
                '__name__': f'extension_{ext.name}',
                '__file__': str(self.extensions_dir / f"{ext.name}.py"),
            }
            
            # Execute to populate namespace
            exec(compiled, module_namespace)
            
            # Look for main function
            main_func = module_namespace.get('execute')
            if main_func is None:
                # Also accept 'main' or any callable starting with ext name
                for key, value in module_namespace.items():
                    if callable(value) and not key.startswith('_'):
                        if key in ('execute', 'main', f'{ext.name}_execute'):
                            main_func = value
                            break
            
            ext._compiled_module = module_namespace
            ext._main_function = main_func
            
            return main_func is not None
            
        except SyntaxError as e:
            logger.error(f"Syntax error in extension {ext.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to compile extension {ext.name}: {e}")
            return False
    
    # ========== EXTENSION MANAGEMENT ==========
    
    def create_extension(
        self,
        name: str,
        code: str,
        description: str = "",
    ) -> CodeExtension:
        """
        Create a new self-written extension.
        
        OpenClaw style: The agent writes Python code that becomes a tool.
        """
        # Validate name
        name = re.sub(r'[^\w]', '_', name.lower())
        if not name or name[0].isdigit():
            name = f"ext_{name}"
        
        # Check for existing
        if name in self.extensions:
            raise ValueError(f"Extension '{name}' already exists")
        
        # Create extension
        ext = CodeExtension(
            name=name,
            description=description or f"Self-written extension: {name}",
            code=code,
        )
        
        # Save to disk
        ext_file = self.extensions_dir / f"{name}.py"
        ext_file.write_text(code, encoding='utf-8')
        
        # Try to compile
        if self._compile_extension(ext):
            ext.status = ExtensionStatus.ACTIVE
            logger.info(f"Created and activated extension: {name}")
        else:
            ext.status = ExtensionStatus.BROKEN
            logger.warning(f"Created extension {name} but compilation failed")
        
        with self._extension_lock:
            self.extensions[name] = ext
        
        # Track in active branch if any
        if self._active_branch_id:
            branch = self.branches.get(self._active_branch_id)
            if branch:
                branch.extensions_created.append(name)
        
        return ext
    
    async def execute_extension(self, name: str, **kwargs) -> Any:
        """Execute a self-written extension."""
        with self._extension_lock:
            ext = self.extensions.get(name)
        
        if not ext:
            raise ValueError(f"Extension '{name}' not found")
        
        if ext.status != ExtensionStatus.ACTIVE:
            raise RuntimeError(f"Extension '{name}' is not active (status: {ext.status.name})")
        
        if not ext._main_function:
            raise RuntimeError(f"Extension '{name}' has no executable function")
        
        # Execute with TUI if enabled
        if self.enable_tui and self.console:
            with self.console.status(f"[bold green]Running {name}...") as status:
                try:
                    result = ext._main_function(**kwargs)
                    ext.execution_count += 1
                    return result
                except Exception as e:
                    logger.error(f"Extension {name} execution failed: {e}")
                    raise
        else:
            result = ext._main_function(**kwargs)
            ext.execution_count += 1
            return result
    
    async def hot_reload(self, ext_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Hot reload extensions without restarting.
        
        OpenClaw feature: Extensions can be reloaded on the fly.
        """
        reloaded = []
        failed = []
        
        if ext_name:
            # Reload specific extension
            targets = [ext_name] if ext_name in self.extensions else []
        else:
            # Reload all
            targets = list(self.extensions.keys())
        
        for name in targets:
            ext = self.extensions.get(name)
            if not ext:
                continue
            
            ext_file = self.extensions_dir / f"{name}.py"
            if not ext_file.exists():
                failed.append(f"{name}: file not found")
                continue
            
            try:
                # Read fresh code
                new_code = ext_file.read_text(encoding='utf-8')
                
                # Update extension
                ext.code = new_code
                ext.last_modified = datetime.now()
                
                # Recompile
                if self._compile_extension(ext):
                    ext.status = ExtensionStatus.ACTIVE
                    reloaded.append(f"{name}: v{ext.version}")
                else:
                    ext.status = ExtensionStatus.BROKEN
                    failed.append(f"{name}: compilation failed")
                    
            except Exception as e:
                failed.append(f"{name}: {str(e)}")
        
        # Also scan for new extensions
        for ext_file in self.extensions_dir.glob("*.py"):
            name = ext_file.stem
            if name not in self.extensions:
                try:
                    self._load_extension_from_file(ext_file)
                    reloaded.append(f"{name}: newly discovered")
                except Exception as e:
                    failed.append(f"{name}: discovery failed - {e}")
        
        return {
            "reloaded": reloaded,
            "failed": failed,
            "total": len(targets),
        }
    
    def list_extensions(self) -> List[Dict[str, Any]]:
        """List all extensions with their status."""
        return [ext.to_dict() for ext in self.extensions.values()]
    
    # ========== SESSION BRANCHING ==========
    
    def create_branch(self, summary: str = "") -> SessionBranch:
        """
        Create a new branch for experimentation.
        
        OpenClaw feature: Tree-structured sessions allow safe experimentation.
        """
        branch_id = f"branch_{hashlib.sha256(f'{summary}{time.time()}'.encode()).hexdigest()[:12]}"
        
        # Parent is currently active branch, or None if first
        parent_id = self._active_branch_id
        
        branch = SessionBranch(
            branch_id=branch_id,
            parent_branch_id=parent_id,
            summary=summary,
            is_active=True,
        )
        
        # Deactivate current branch if any
        if self._active_branch_id and self._active_branch_id in self.branches:
            self.branches[self._active_branch_id].is_active = False
        
        self.branches[branch_id] = branch
        self._active_branch_id = branch_id
        
        if self._main_branch_id is None:
            self._main_branch_id = branch_id
        
        logger.info(f"Created branch: {branch_id} (parent: {parent_id})")
        return branch
    
    def switch_branch(self, branch_id: str) -> Optional[SessionBranch]:
        """Switch to a different branch."""
        if branch_id not in self.branches:
            return None
        
        # Deactivate current
        if self._active_branch_id in self.branches:
            self.branches[self._active_branch_id].is_active = False
        
        # Activate new
        self._active_branch_id = branch_id
        self.branches[branch_id].is_active = True
        
        logger.info(f"Switched to branch: {branch_id}")
        return self.branches[branch_id]
    
    def get_branch_tree(self) -> Dict[str, Any]:
        """Get hierarchical tree of all branches."""
        # Build parent-child relationships
        tree: Dict[str, List[str]] = {}
        for branch in self.branches.values():
            parent = branch.parent_branch_id or "root"
            if parent not in tree:
                tree[parent] = []
            tree[parent].append(branch.branch_id)
        
        def build_node(branch_id: str, depth: int = 0) -> Dict[str, Any]:
            branch = self.branches.get(branch_id)
            if not branch:
                return {"id": branch_id, "missing": True}
            
            children = tree.get(branch_id, [])
            return {
                "id": branch_id,
                "summary": branch.summary,
                "is_active": branch.is_active,
                "depth": depth,
                "children": [build_node(child, depth + 1) for child in children],
            }
        
        # Find root branches (no parent or parent not in tree)
        roots = [bid for bid, b in self.branches.items() 
                if b.parent_branch_id is None or b.parent_branch_id not in self.branches]
        
        return {
            "tree": [build_node(root) for root in roots],
            "active_branch": self._active_branch_id,
            "total_branches": len(self.branches),
        }
    
    def add_message_to_branch(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to the current branch."""
        if not self._active_branch_id:
            return
        
        branch = self.branches.get(self._active_branch_id)
        if not branch:
            return
        
        branch.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })
    
    def get_branch_context(self, branch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get conversation context for a branch."""
        target_id = branch_id or self._active_branch_id
        if not target_id or target_id not in self.branches:
            return []
        
        # Collect messages from this branch and ancestors
        context = []
        current_id = target_id
        
        while current_id:
            branch = self.branches.get(current_id)
            if not branch:
                break
            
            # Add this branch's messages (newest first for prepending)
            for msg in reversed(branch.messages):
                context.insert(0, msg)
            
            current_id = branch.parent_branch_id
        
        return context
    
    # ========== TUI COMPONENTS ==========
    
    def show_progress(
        self,
        task_id: str,
        description: str = "Processing...",
        total: Optional[int] = None,
    ) -> Any:
        """Show a progress bar (if TUI enabled)."""
        if not self.enable_tui or not self.console:
            return None
        
        if task_id in self._progress_bars:
            # Update existing
            progress = self._progress_bars[task_id]
            return progress
        
        # Create new progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )
        
        task = progress.add_task(description, total=total)
        self._progress_bars[task_id] = progress
        
        return progress
    
    def update_progress(
        self,
        task_id: str,
        advance: int = 1,
        description: Optional[str] = None,
    ) -> None:
        """Update progress bar."""
        if not self.enable_tui or task_id not in self._progress_bars:
            return
        
        progress = self._progress_bars[task_id]
        for task in progress.tasks:
            if description:
                progress.update(task.id, advance=advance, description=description)
            else:
                progress.update(task.id, advance=advance)
    
    def finish_progress(self, task_id: str) -> None:
        """Complete and remove progress bar."""
        if task_id in self._progress_bars:
            del self._progress_bars[task_id]
    
    def show_spinner(self, message: str = "Loading...") -> Any:
        """Show a spinner (if TUI enabled)."""
        if not self.enable_tui or not self.console:
            return None
        
        return self.console.status(f"[bold green]{message}")
    
    def display_code(self, code: str, language: str = "python") -> None:
        """Display syntax-highlighted code."""
        if not self.enable_tui or not self.console:
            print(code)
            return
        
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, border_style="green"))
    
    def display_table(self, data: List[Dict[str, Any]], title: str = "") -> None:
        """Display data as a table."""
        if not self.enable_tui or not self.console or not data:
            # Fallback to simple print
            for row in data:
                print(row)
            return
        
        table = Table(title=title, box=box.ROUNDED)
        
        # Add columns
        for key in data[0].keys():
            table.add_column(key, style="cyan")
        
        # Add rows
        for row in data:
            table.add_row(*[str(v) for v in row.values()])
        
        self.console.print(table)
    
    def display_tree(self, data: Dict[str, Any], title: str = "Tree") -> None:
        """Display hierarchical data as a tree."""
        if not self.enable_tui or not self.console:
            self._print_tree_fallback(data, title)
            return
        
        tree = Tree(f"[bold]{title}[/]")
        self._build_rich_tree(tree, data)
        self.console.print(tree)
    
    def _build_rich_tree(self, parent: Any, node: Dict[str, Any]) -> None:
        """Recursively build rich tree."""
        if "children" in node and node["children"]:
            for child in node["children"]:
                child_node = parent.add(f"{child.get('id', 'unknown')}: {child.get('summary', '')[:40]}")
                self._build_rich_tree(child_node, child)
    
    def _print_tree_fallback(self, data: Dict[str, Any], title: str, indent: int = 0) -> None:
        """Fallback tree printing without rich."""
        prefix = "  " * indent
        print(f"{prefix}{title}")
        
        if "tree" in data:
            for node in data["tree"]:
                self._print_tree_node(node, indent + 1)
    
    def _print_tree_node(self, node: Dict[str, Any], indent: int) -> None:
        """Print tree node recursively."""
        prefix = "  " * indent
        marker = "● " if node.get("is_active") else "○ "
        print(f"{prefix}{marker}{node.get('id', 'unknown')}: {node.get('summary', '')[:50]}")
        
        for child in node.get("children", []):
            self._print_tree_node(child, indent + 1)
    
    # ========== SKILL VS TOOL DISTINCTION ==========
    
    def get_skill_instructions(self) -> str:
        """
        Explain Skills vs Tools distinction (OpenClaw concept).
        
        Tools: Loaded into LLM context (for the AI to use)
        Skills: Agent-maintained functionality (no context overhead)
        """
        return """
╔══════════════════════════════════════════════════════════════════╗
║  SKILLS vs TOOLS - OpenClaw Architecture                        ║
╠══════════════════════════════════════════════════════════════════╣

TOOLS (LLM Context):
  • What the AI sees and can directly invoke
  • 4 minimal tools: read, write, edit, bash
  • Extensions you create become tools after /reload

SKILLS (Agent-Maintained):
  • Functionality the agent manages internally
  • No context overhead - not shown to LLM
  • Examples:
    - Browser automation via CDP
    - Code review workflows
    - Session management
    - File change tracking

KEY DIFFERENCE:
  Tools = "I can call this now" (immediate execution)
  Skills = "I know how to do this" (managed capability)

When you need new functionality:
  1. Can it be a Tool? → Write extension code with write/edit
  2. Can it be a Skill? → Agent manages it internally
  3. Complex? → Compose Tools + Skills

The agent writes Skills through code - this is the OpenClaw way!
        """.strip()
    
    # ========== INTEGRATION WITH AGENT ==========
    
    def get_system_prompt_addon(self) -> str:
        """Get system prompt addon for SimplifiedAgant mode."""
        lines = [
            "",
            "=" * 70,
            "SIMPLIFIEDAGANT MODE ACTIVE - OpenClaw Architecture",
            "=" * 70,
            "",
            MinimalToolSet.get_tool_instructions(),
            "",
            self.get_skill_instructions(),
            "",
            "CURRENT EXTENSIONS:",
        ]
        
        if self.extensions:
            for name, ext in self.extensions.items():
                status_icon = "✅" if ext.status == ExtensionStatus.ACTIVE else "⚠️"
                lines.append(f"  {status_icon} {name} v{ext.version} - {ext.description[:50]}...")
        else:
            lines.append("  (No extensions yet - create one with the write tool!)")
        
        lines.extend([
            "",
            "BRANCH INFORMATION:",
            f"  Active branch: {self._active_branch_id or 'none'}",
            f"  Total branches: {len(self.branches)}",
        ])
        
        if self.branches:
            lines.append("  Use /branch to create new experimental branch")
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)
