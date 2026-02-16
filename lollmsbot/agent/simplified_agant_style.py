"""
simplified_agent minimal agent architecture for lollmsbot.

Implements the "Pi" philosophy: 4 core tools that self-extend through code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from lollmsbot.agent import Tool, ToolResult
from lollmsbot.agent.rlm import RLMMemoryManager, MemoryChunkType

import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeExtension:
    """A self-written code extension (Skill in SimplifiedAgant terms)."""
    name: str
    code: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "is_active": self.is_active,
            "dependencies": self.dependencies,
        }


@dataclass
class SessionBranch:
    """A branch in the tree-structured session history."""
    branch_id: str
    parent_branch_id: Optional[str]
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    extensions_created: List[str] = field(default_factory=list)
    is_active: bool = True
    summary: str = ""  # AI-generated summary of what happened in this branch
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_branch_id": self.parent_branch_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.messages),
            "extensions_created": self.extensions_created,
            "is_active": self.is_active,
            "summary": self.summary,
        }


class MinimalToolSet:
    """
    SimplifiedAgant's 4 core tools: Read, Write, Edit, Bash.
    
    These are the ONLY tools loaded into context. Everything else
    is implemented as self-written code extensions (skills).
    """
    
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    async def read(self, path: str, offset: int = 0, limit: int = 100) -> ToolResult:
        """Read file contents."""
        try:
            full_path = self._resolve_path(path)
            if not full_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {path}"
                )
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            start = offset
            end = min(offset + limit, len(lines))
            selected = lines[start:end]
            
            return ToolResult(
                success=True,
                output={
                    "path": str(full_path),
                    "content": ''.join(selected),
                    "total_lines": len(lines),
                    "shown_lines": f"{start}-{end}",
                }
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def write(self, path: str, content: str, append: bool = False) -> ToolResult:
        """Write or append to file."""
        try:
            full_path = self._resolve_path(path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append else 'w'
            with open(full_path, mode, encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                output={
                    "path": str(full_path),
                    "bytes_written": len(content.encode('utf-8')),
                    "operation": "append" if append else "write",
                }
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def edit(self, path: str, old_string: str, new_string: str) -> ToolResult:
        """Edit file by replacing old_string with new_string."""
        try:
            full_path = self._resolve_path(path)
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if old_string not in content:
                # Try fuzzy matching
                similar = self._find_similar(content, old_string)
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"String not found. Did you mean: {similar[:100]}..."
                )
            
            new_content = content.replace(old_string, new_string, 1)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return ToolResult(
                success=True,
                output={
                    "path": str(full_path),
                    "replacements": 1,
                    "old_length": len(old_string),
                    "new_length": len(new_string),
                }
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    async def bash(self, command: str, timeout: int = 60, cwd: Optional[str] = None) -> ToolResult:
        """Execute bash command with safety checks."""
        # Safety: block dangerous commands
        dangerous = ['rm -rf /', 'mkfs', 'dd if=/dev/zero', '> /dev/sda']
        for d in dangerous:
            if d in command:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Dangerous command blocked: {d}"
                )
        
        try:
            working_dir = cwd or str(self.working_dir)
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            return ToolResult(
                success=result.returncode == 0,
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error=f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to working directory."""
        if path.startswith('/'):
            return Path(path)
        return self.working_dir / path
    
    def _find_similar(self, content: str, target: str) -> str:
        """Find similar string in content for error suggestions."""
        # Simple approach: find lines containing similar words
        target_words = set(target.lower().split())
        best_match = ""
        best_score = 0
        
        for line in content.split('\n'):
            line_words = set(line.lower().split())
            score = len(target_words & line_words)
            if score > best_score:
                best_score = score
                best_match = line
        
        return best_match


class SimplifiedAgantStyle:
    """
    simplified_agent-style agent with minimal tools and self-extension capability.
    
    Key features:
    - 4 core tools only (Read, Write, Edit, Bash)
    - Self-written code extensions (skills)
    - Tree-structured sessions with branching
    - Hot reloading of extensions
    """
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        memory_manager: Optional[RLMMemoryManager] = None,
    ):
        self.working_dir = working_dir or Path.home() / ".lollmsbot" / "openclaw_workspace"
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Extensions directory
        self.extensions_dir = self.working_dir / ".extensions"
        self.extensions_dir.mkdir(exist_ok=True)
        
        # Minimal core tools
        self.core_tools = MinimalToolSet(self.working_dir)
        
        # Memory
        self._memory = memory_manager
        
        # Session tree structure
        self.branches: Dict[str, SessionBranch] = {}
        self.active_branch_id: Optional[str] = None
        self._create_main_branch()
        
        # Loaded extensions (skills)
        self.extensions: Dict[str, CodeExtension] = {}
        self._extension_modules: Dict[str, Any] = {}
        
        # Load existing extensions
        self._load_extensions_from_disk()
        
        # Hot reload watcher
        self._reload_callbacks: List[Callable[[str], None]] = []
    
    def _create_main_branch(self) -> None:
        """Create the main trunk branch."""
        main = SessionBranch(
            branch_id="main",
            parent_branch_id=None,
            summary="Main conversation trunk",
        )
        self.branches["main"] = main
        self.active_branch_id = "main"
    
    # ========== Core Tool Interface ==========
    
    async def execute_core_tool(self, tool_name: str, **params) -> ToolResult:
        """Execute one of the 4 core tools."""
        if tool_name == "read":
            return await self.core_tools.read(
                path=params.get("path", ""),
                offset=params.get("offset", 0),
                limit=params.get("limit", 100),
            )
        elif tool_name == "write":
            return await self.core_tools.write(
                path=params.get("path", ""),
                content=params.get("content", ""),
                append=params.get("append", False),
            )
        elif tool_name == "edit":
            return await self.core_tools.edit(
                path=params.get("path", ""),
                old_string=params.get("old_string", ""),
                new_string=params.get("new_string", ""),
            )
        elif tool_name == "bash":
            return await self.core_tools.bash(
                command=params.get("command", ""),
                timeout=params.get("timeout", 60),
                cwd=params.get("cwd"),
            )
        else:
            # Try extension
            return await self._execute_extension(tool_name, params)
    
    # ========== Self-Extension System ==========
    
    async def create_extension(
        self,
        name: str,
        description: str,
        code: str,
        test_command: Optional[str] = None,
    ) -> CodeExtension:
        """
        Create a new code extension (skill).
        
        The agent writes Python code that extends its capabilities.
        """
        # Validate Python syntax
        try:
            compile(code, f"<{name}>", 'exec')
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")
        
        extension = CodeExtension(
            name=name,
            code=code,
            description=description,
        )
        
        # Save to disk
        ext_path = self.extensions_dir / f"{name}.py"
        ext_path.write_text(code, encoding='utf-8')
        
        # Save metadata
        meta_path = self.extensions_dir / f"{name}.json"
        meta_path.write_text(
            json.dumps(extension.to_dict(), indent=2),
            encoding='utf-8'
        )
        
        # Load and test
        await self._load_extension(extension)
        
        if test_command:
            test_result = await self._test_extension(name, test_command)
            extension.test_results.append(test_result)
        
        self.extensions[name] = extension
        
        # Record in active branch
        if self.active_branch_id:
            branch = self.branches[self.active_branch_id]
            branch.extensions_created.append(name)
        
        # Store in memory
        if self._memory:
            await self._memory.store_in_ems(
                content=f"Extension created: {name}\nDescription: {description}\nCode:\n{code[:500]}...",
                chunk_type=MemoryChunkType.PROCEDURAL,
                importance=7.0,
                tags=["extension", "skill", "self_written", name],
                summary=f"Self-written extension: {name}",
                load_hints=[name, "extension", "skill"],
                source=f"extension:{name}",
            )
        
        return extension
    
    async def _load_extension(self, extension: CodeExtension) -> bool:
        """Load extension code into memory."""
        try:
            # Create module namespace
            module_name = f"ext_{extension.name}"
            namespace = {
                '__name__': module_name,
                '__file__': str(self.extensions_dir / f"{extension.name}.py"),
            }
            
            # Execute code in namespace
            exec(extension.code, namespace)
            
            # Store module
            self._extension_modules[extension.name] = namespace
            
            extension.is_active = True
            logger.info(f"Loaded extension: {extension.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load extension {extension.name}: {e}")
            extension.is_active = False
            return False
    
    async def _execute_extension(self, name: str, params: Dict[str, Any]) -> ToolResult:
        """Execute a loaded extension."""
        if name not in self._extension_modules:
            return ToolResult(
                success=False,
                output=None,
                error=f"Extension '{name}' not found. Create it with: create_extension(name='{name}', ...)"
            )
        
        module = self._extension_modules[name]
        
        # Look for execute function
        if 'execute' not in module:
            return ToolResult(
                success=False,
                output=None,
                error=f"Extension '{name}' has no execute() function"
            )
        
        try:
            execute_func = module['execute']
            result = execute_func(**params)
            
            # Handle async
            if asyncio.iscoroutine(result):
                result = await result
            
            # Normalize to ToolResult
            if isinstance(result, dict):
                return ToolResult(success=True, output=result)
            elif isinstance(result, ToolResult):
                return result
            else:
                return ToolResult(success=True, output={"result": str(result)})
                
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Extension '{name}' execution failed: {str(e)}\n{traceback.format_exc()}"
            )
    
    async def _test_extension(self, name: str, test_command: str) -> Dict[str, Any]:
        """Test an extension with a command."""
        start = time.time()
        
        try:
            # Parse test command
            parts = test_command.split()
            tool_name = parts[0]
            params = {}
            
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    params[key] = value
            
            result = await self._execute_extension(name, params)
            
            return {
                "test_command": test_command,
                "success": result.success,
                "duration": time.time() - start,
                "output_preview": str(result.output)[:200] if result.output else None,
                "error": result.error,
            }
            
        except Exception as e:
            return {
                "test_command": test_command,
                "success": False,
                "duration": time.time() - start,
                "error": str(e),
            }
    
    def _load_extensions_from_disk(self) -> None:
        """Load all extensions from disk on startup."""
        if not self.extensions_dir.exists():
            return
        
        for ext_file in self.extensions_dir.glob("*.py"):
            name = ext_file.stem
            
            # Load metadata if exists
            meta_file = self.extensions_dir / f"{name}.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    extension = CodeExtension(
                        name=name,
                        code=ext_file.read_text(),
                        description=meta.get("description", ""),
                        created_at=datetime.fromisoformat(meta.get("created_at", datetime.now().isoformat())),
                        version=meta.get("version", 1),
                        dependencies=meta.get("dependencies", []),
                    )
                except Exception:
                    extension = CodeExtension(
                        name=name,
                        code=ext_file.read_text(),
                        description=f"Extension: {name}",
                    )
            else:
                extension = CodeExtension(
                    name=name,
                    code=ext_file.read_text(),
                    description=f"Extension: {name}",
                )
            
            # Async load will happen later
            self.extensions[name] = extension
    
    # ========== Hot Reload ==========
    
    async def hot_reload(self, extension_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Hot reload extensions.
        
        If extension_name is None, reload all.
        """
        reloaded = []
        failed = []
        
        targets = [extension_name] if extension_name else list(self.extensions.keys())
        
        for name in targets:
            if name not in self.extensions:
                failed.append(f"{name}: not found")
                continue
            
            ext = self.extensions[name]
            # Re-read from disk
            ext_path = self.extensions_dir / f"{name}.py"
            if ext_path.exists():
                ext.code = ext_path.read_text()
                ext.version += 1
            
            success = await self._load_extension(ext)
            if success:
                reloaded.append(name)
            else:
                failed.append(f"{name}: load failed")
        
        # Notify callbacks
        for cb in self._reload_callbacks:
            for name in reloaded:
                try:
                    cb(name)
                except Exception:
                    pass
        
        return {
            "reloaded": reloaded,
            "failed": failed,
            "total": len(targets),
        }
    
    def on_reload(self, callback: Callable[[str], None]) -> None:
        """Register callback for extension reload events."""
        self._reload_callbacks.append(callback)
    
    # ========== Session Tree ==========
    
    def branch_session(self, summary: str = "") -> SessionBranch:
        """
        Create a new branch from current active branch.
        
        This allows experimenting without polluting main context.
        """
        if not self.active_branch_id:
            self._create_main_branch()
        
        parent_id = self.active_branch_id
        branch_id = f"branch_{len(self.branches)}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        
        # Copy parent messages
        parent = self.branches[parent_id]
        
        branch = SessionBranch(
            branch_id=branch_id,
            parent_branch_id=parent_id,
            messages=parent.messages.copy(),  # Inherit history
            summary=summary or f"Branched from {parent_id}",
        )
        
        self.branches[branch_id] = branch
        self.active_branch_id = branch_id
        
        return branch
    
    def switch_branch(self, branch_id: str) -> bool:
        """Switch to a different branch."""
        if branch_id not in self.branches:
            return False
        
        self.active_branch_id = branch_id
        return True
    
    def merge_branch(self, branch_id: str, target_id: Optional[str] = None) -> bool:
        """
        Merge a branch back to its parent or specified target.
        
        Summarize changes and add to target.
        """
        if branch_id not in self.branches:
            return False
        
        branch = self.branches[branch_id]
        target = target_id or branch.parent_branch_id
        
        if not target or target not in self.branches:
            return False
        
        target_branch = self.branches[target]
        
        # Create merge message with summary
        merge_msg = {
            "role": "system",
            "content": f"[BRANCH MERGED] {branch.summary}\nExtensions created: {', '.join(branch.extensions_created)}\nMessages added: {len(branch.messages) - len(target_branch.messages)}",
            "timestamp": datetime.now().isoformat(),
        }
        
        target_branch.messages.append(merge_msg)
        
        # Deactivate source branch
        branch.is_active = False
        
        # Switch to target
        self.active_branch_id = target
        
        return True
    
    def get_branch_tree(self) -> Dict[str, Any]:
        """Get tree structure of all branches."""
        def build_tree(branch_id: str, depth: int = 0) -> Dict[str, Any]:
            branch = self.branches.get(branch_id)
            if not branch:
                return {"error": f"Branch {branch_id} not found"}
            
            children = [
                build_tree(bid, depth + 1)
                for bid, b in self.branches.items()
                if b.parent_branch_id == branch_id
            ]
            
            return {
                "id": branch_id,
                "summary": branch.summary,
                "depth": depth,
                "is_active": branch_id == self.active_branch_id,
                "message_count": len(branch.messages),
                "extensions": branch.extensions_created,
                "children": children,
            }
        
        # Find root (no parent)
        roots = [bid for bid, b in self.branches.items() if b.parent_branch_id is None]
        
        return {
            "active_branch": self.active_branch_id,
            "total_branches": len(self.branches),
            "tree": [build_tree(r) for r in roots],
        }
    
    # ========== Agent Interface ==========
    
    def get_system_prompt(self) -> str:
        """
        Generate system prompt explaining the SimplifiedAgant-style architecture.
        """
        extensions_list = ", ".join(self.extensions.keys()) or "none yet"
        
        return f"""You are an AI agent with simplified_agent-style architecture.

## Your Core Tools (ALWAYS available)
1. **read(path, offset=0, limit=100)** - Read file contents
2. **write(path, content, append=False)** - Write or append to file
3. **edit(path, old_string, new_string)** - Replace text in file
4. **bash(command, timeout=60)** - Execute shell command

## Your Extensions (self-written skills)
Currently loaded: {extensions_list}

You can create new extensions by writing Python code:
```python
# Example extension: calculator
def execute(expression: str):
    import math
    try:
        result = eval(expression, {{"__builtins__": {{}}}}, math.__dict__)
        return {{"result": result}}
    except Exception as e:
        return {{"error": str(e)}}
```

## Session Tree
You can branch sessions to experiment without risk:
- Current branch: {self.active_branch_id}
- Total branches: {len(self.branches)}
- Use: branch_session(summary="experiment with X")

## Hot Reload
Extensions can be modified and reloaded instantly.
Use: hot_reload(extension_name="my_extension")

## Philosophy
"Software building software" - you maintain your own functionality.
Write code to extend yourself. Throw away what doesn't work.
No pre-built extensions; hand-crafted to user specifications.

## Response Format
When you want to use a tool, output:
<tool>TOOL_NAME</tool>
<param1>value1</param1>

Example:
<tool>read</tool>
<path>./main.py</path>
<limit>50</limit>
"""
    
    async def chat(
        self,
        user_message: str,
        llm_client: Any,  # LollmsClient or similar
        streaming_callback: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """
        Process a chat message with SimplifiedAgant-style tool loop.
        """
        # Add to active branch
        if self.active_branch_id:
            branch = self.branches[self.active_branch_id]
            branch.messages.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            })
        
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
        ]
        
        # Add branch history
        if self.active_branch_id:
            branch = self.branches[self.active_branch_id]
            messages.extend(branch.messages[-10:])  # Last 10 messages
        
        # Generate response
        response = llm_client.generate_text(
            prompt=messages[-1]["content"] if messages else user_message,
            temperature=0.7,
        )
        
        # Parse and execute tools
        final_response, tools_used = await self._parse_and_execute_tools(response)
        
        # Add to branch
        if self.active_branch_id:
            branch = self.branches[self.active_branch_id]
            branch.messages.append({
                "role": "assistant",
                "content": final_response,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat(),
            })
        
        return final_response
    
    async def _parse_and_execute_tools(self, llm_response: str) -> Tuple[str, List[str]]:
        """Parse tool calls from LLM response and execute them."""
        tools_used = []
        
        # Find all tool calls
        tool_pattern = r'<tool>(\w+)</tool>'
        param_pattern = r'<(\w+)>(.*?)</\1>'
        
        # Simple iterative approach
        max_iterations = 5
        current_response = llm_response
        
        for _ in range(max_iterations):
            tool_match = re.search(tool_pattern, current_response, re.DOTALL)
            if not tool_match:
                break
            
            tool_name = tool_match.group(1).lower()
            
            # Find params after this tool tag
            remaining = current_response[tool_match.end():]
            
            # Extract params until next tool or end
            params = {}
            for param_match in re.finditer(param_pattern, remaining):
                param_name = param_match.group(1)
                param_value = param_match.group(2)
                
                # Stop if we hit another tool tag
                if param_name == "tool":
                    break
                
                params[param_name] = param_value
            
            # Execute tool
            result = await self.execute_core_tool(tool_name, **params)
            tools_used.append(tool_name)
            
            # Replace tool call with result indicator
            # Find end of this tool block
            block_end = tool_match.end()
            for param_match in re.finditer(param_pattern, remaining):
                if param_match.group(1) == "tool":
                    break
                block_end = param_match.end()
            
            result_indicator = f"\n[Tool {tool_name}: {'success' if result.success else 'failed'}]\n"
            if result.output:
                result_indicator += f"Output: {str(result.output)[:200]}\n"
            
            current_response = (
                current_response[:tool_match.start()] +
                result_indicator +
                current_response[block_end:]
            )
        
        return current_response.strip(), tools_used