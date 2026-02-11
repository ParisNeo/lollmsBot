"""
LLM prompt building and formatting utilities.

Handles system prompt construction, conversation formatting, and
prompt specialization for different task types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lollmsbot.agent.config import ConversationTurn
    from lollmsbot.tools import Tool


class PromptBuilder:
    """Builds system prompts and formats conversations for LLM."""
    
    def __init__(self, agent_name: str = "LollmsBot") -> None:
        self.agent_name = agent_name
    
    def build_system_prompt(
        self,
        tools: Dict[str, Tool],
        context: Optional[Dict[str, Any]] = None,
        important_facts: Optional[Dict[str, Any]] = None,
        soul: Optional[Any] = None,
    ) -> str:
        """Build the standard system prompt for chat."""
        # Start with explicit tool format warning
        tool_instructions = self._build_strict_tool_instructions(tools)
        
        # Try to get Soul's system prompt if available
        if soul:
            prompt_context = dict(context) if context else {}
            if important_facts:
                prompt_context["important_facts"] = important_facts
            base_prompt = soul.generate_system_prompt(prompt_context)
        else:
            base_prompt = f"You are {self.agent_name}, a helpful AI assistant with access to tools."
        
        # Combine with tool instructions first (higher priority)
        parts = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║  CRITICAL: TOOL CALL FORMAT - READ CAREFULLY                    ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            tool_instructions,
            "",
            "╔══════════════════════════════════════════════════════════════════╗",
            "║  YOUR IDENTITY AND PURPOSE                                       ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            base_prompt,
        ]
        
        # Add important memory context if available
        if important_facts and (not base_prompt or "important_facts" not in base_prompt.lower()):
            creator_info = important_facts.get("creator_identity", {})
            if creator_info and isinstance(creator_info, dict):
                creator_name = creator_info.get("creator_name", "unknown")
                creator_alias = creator_info.get("creator_alias", "unknown")
                parts.extend([
                    "",
                    f"[IMPORTANT CONTEXT] Your creator is {creator_name} "
                    f"(also known as {creator_alias}). This was directly stated by them. "
                    "Honor their trust and vision."
                ])
        
        if context:
            channel = context.get("channel", "unknown")
            parts.append(f"\nCurrent channel: {channel}")
        
        return "\n".join(parts)
    
    def _build_strict_tool_instructions(self, tools: Dict[str, Tool]) -> str:
        """Build strict, unambiguous tool instructions with clear examples."""
        tool_list = self._format_tool_list(tools)
        
        # Build per-tool examples
        tool_examples = []
        for name, tool in tools.items():
            example = self._get_tool_example(name, tool)
            if example:
                tool_examples.append(example)
        
        instructions = f"""AVAILABLE TOOLS:
{tool_list}

╔══════════════════════════════════════════════════════════════════╗
║  TOOL CALL FORMAT - USE THESE EXACT XML TAGS                     ║
╠══════════════════════════════════════════════════════════════════╣

To use a tool, output ONLY these XML tags with PLAIN TEXT values:

<tool>TOOL_NAME</tool>
<param1>value1</param1>
<param2>value2</param2>

NEVER use JSON. NEVER use {{curly braces}}. NEVER use "quotes" around values.

╔══════════════════════════════════════════════════════════════════╗
║  EXAMPLES - COPY THE EXACT FORMAT BELOW                          ║
╠══════════════════════════════════════════════════════════════════╣

{chr(10).join(tool_examples) if tool_examples else "No tool examples available."}

╔══════════════════════════════════════════════════════════════════╗
║  CRITICAL RULES                                                  ║
╠══════════════════════════════════════════════════════════════════╣

1. ONLY use the tag names shown above for each tool
2. NO JSON blocks, NO markdown code blocks
3. NEVER put {{...}} inside XML tags
4. Values are PLAIN TEXT - no quotes, no braces
5. Output XML tags ONLY - no explanations before or after
6. After tool tags, STOP - wait for the result

╔══════════════════════════════════════════════════════════════════╗
║  WHAT HAPPENS                                                    ║
╠══════════════════════════════════════════════════════════════════╣

- You output tool XML tags
- System executes the tool
- You receive the result
- You respond to the user with the information

Users do NOT see your tool calls - only "**🔧 called tool**" and your final response."""
        
        return instructions
    
    def _get_tool_example(self, tool_name: str, tool: Tool) -> str:
        """Generate a concrete example for a specific tool."""
        examples = {
            "http": """HTTP GET example:
<tool>http</tool>
<method>get</method>
<url>https://example.com/page.html</url>

HTTP POST example:
<tool>http</tool>
<method>post</method>
<url>https://api.example.com/data</url>
<data>key=value&other=test</data>""",

            "filesystem": """Read file example:
<tool>filesystem</tool>
<operation>read_file</operation>
<path>/path/to/file.txt</path>

Write file example:
<tool>filesystem</tool>
<operation>write_file</operation>
<path>/output.txt</path>
<content>File contents here</content>

Create HTML app example:
<tool>filesystem</tool>
<operation>create_html_app</operation>
<filename>myapp.html</filename>
<html_content><h1>Hello World</h1></html_content>""",

            "calendar": """Add event example:
<tool>calendar</tool>
<operation>add_event</operation>
<title>Meeting</title>
<start>2024-01-15T10:00:00</start>
<end>2024-01-15T11:00:00</end>

List events example:
<tool>calendar</tool>
<operation>get_events</operation>
<start>2024-01-01T00:00:00</start>
<end>2024-01-31T23:59:59</end>""",

            "shell": """Execute command example:
<tool>shell</tool>
<operation>execute</operation>
<command>ls -la</command>""",
        }
        
        return examples.get(tool_name, f"""{tool_name} example:
<tool>{tool_name}</tool>
<operation>default</operation>
<param>value</param>""")
    
    def build_file_generation_prompt(
        self,
        tools: Dict[str, Tool],
        context: Optional[Dict[str, Any]] = None,
        soul: Optional[Any] = None,
    ) -> str:
        """Build a prompt optimized for file generation tasks."""
        # Get strict tool instructions
        tool_instructions = self._build_strict_tool_instructions(tools)
        
        # Start with Soul's system prompt if available
        if soul:
            base_prompt = soul.generate_system_prompt(context)
        else:
            base_prompt = f"You are {self.agent_name}, a helpful AI assistant that can create files and code."
        
        parts = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║  CRITICAL: TOOL CALL FORMAT - READ CAREFULLY                    ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            tool_instructions,
            "",
            "╔══════════════════════════════════════════════════════════════════╗",
            "║  FILE GENERATION MODE                                            ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            base_prompt,
            "",
            "When creating files, you MUST use the filesystem tool with XML format above.",
            "DO NOT output code directly - always use the tool.",
        ]
        
        if context:
            channel = context.get("channel", "unknown")
            parts.append(f"\nCurrent channel: {channel}")
        
        return "\n".join(parts)
    
    def _format_tool_list(self, tools: Dict[str, Tool]) -> str:
        """Format tools as a bullet list for prompts."""
        if not tools:
            return "No tools available"
        
        lines = []
        for name, tool in tools.items():
            desc = tool.description[:60] if len(tool.description) > 60 else tool.description
            lines.append(f"  • {name}: {desc}")
        
        return "\n".join(lines)
    
    def format_prompt_for_lollms(
        self,
        system_prompt: str,
        history: List[ConversationTurn],
        current_message: str,
        max_history: int = 10,
    ) -> str:
        """Format complete prompt for LoLLMS including system and history."""
        parts = [system_prompt, ""]
        
        # Add reminder about tool format in history section
        parts.append("╔══════════════════════════════════════════════════════════════════╗")
        parts.append("║  REMINDER: Use ONLY XML format for tools. NO JSON/markdown.     ║")
        parts.append("╚══════════════════════════════════════════════════════════════════╝")
        parts.append("")
        
        # Add history header
        parts.append("=== CONVERSATION HISTORY ===")
        parts.append("")
        
        # Add recent history
        for turn in history[-max_history:]:
            parts.append(f"User: {turn.user_message}")
            parts.append(f"Assistant: {turn.agent_response}")
            parts.append("")
        
        # Add current message with final reminder
        parts.append(f"User: {current_message}")
        parts.append("")
        parts.append("╔══════════════════════════════════════════════════════════════════╗")
        parts.append("║  BEFORE RESPONDING:                                              ║")
        parts.append("║  • Need a tool? Use ONLY XML tags: <tool>, <method>, <url>, etc. ║")
        parts.append("║  • NO JSON blocks, NO markdown ```json                           ║")
        parts.append("║  • NO {{curly braces}} inside XML tag values                     ║")
        parts.append("║  • Plain text values ONLY - no quotes                            ║")
        parts.append("╚══════════════════════════════════════════════════════════════════╝")
        parts.append("")
        parts.append("Assistant:")
        
        return "\n".join(parts)
