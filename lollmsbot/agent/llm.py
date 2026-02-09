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
        # Try to get Soul's system prompt if available
        # Pass important_facts in context so Soul can include them
        if soul:
            # Merge important_facts into context for Soul to use
            prompt_context = dict(context) if context else {}
            if important_facts:
                prompt_context["important_facts"] = important_facts
            base_prompt = soul.generate_system_prompt(prompt_context)
        else:
            base_prompt = f"You are {self.agent_name}, a helpful AI assistant with access to tools."
        
        return self._add_tools_to_prompt(base_prompt, tools, context, important_facts)
    
    def build_file_generation_prompt(
        self,
        tools: Dict[str, Tool],
        context: Optional[Dict[str, Any]] = None,
        soul: Optional[Any] = None,
    ) -> str:
        """Build a prompt optimized for file generation tasks."""
        # Start with Soul's system prompt if available
        if soul:
            base_prompt = soul.generate_system_prompt(context)
        else:
            base_prompt = f"You are {self.agent_name}, a helpful AI assistant that can create files and code."
        
        # Get available tools with descriptions
        tool_list = self._format_tool_list(tools)
        
        # Build file generation specific prompt
        parts = [
            base_prompt,
            "",
            "AVAILABLE TOOLS (USE THESE EXACT NAMES):",
            tool_list,
            "",
            "CRITICAL INSTRUCTION: When the user asks you to create, build, generate, or write a file,",
            "you MUST use the filesystem tool. Do NOT output the code directly in your response.",
            "",
            "TOOL CALL FORMAT (USE EXACTLY AS SHOWN):",
            "[[TOOL:filesystem|{\"operation\": \"create_html_app\", \"filename\": \"game.html\", \"html_content\": \"<!DOCTYPE html>...\"}]]",
            "",
            "EXAMPLE TOOL CALLS:",
            '1. For HTML games: [[TOOL:filesystem|{"operation": "create_html_app", "filename": "mygame.html", "html_content": "<!DOCTYPE html>..."}]]',
            '2. For text files: [[TOOL:filesystem|{"operation": "write_file", "path": "output.txt", "content": "file content here"}]]',
            '3. For listing files: [[TOOL:filesystem|{"operation": "list_dir", "path": "."}]]',
            "",
            "RULES:",
            "- ONLY use tool names from the AVAILABLE TOOLS list above",
            "- NEVER use placeholder names like 'toolname' - use the actual tool name",
            "- ALWAYS wrap tool calls in [[TOOL:...|{...}]] format",
            "- CREATE THE FILE using the tool, then briefly confirm in text",
        ]
        
        if context:
            channel = context.get("channel", "unknown")
            parts.append(f"\nCurrent channel: {channel}")
        
        return "\n".join(parts)
    
    def _add_tools_to_prompt(
        self,
        base_prompt: str,
        tools: Dict[str, Tool],
        context: Optional[Dict[str, Any]],
        important_facts: Optional[Dict[str, Any]],
    ) -> str:
        """Add tool documentation to a base system prompt."""
        tool_list = self._format_tool_list(tools)
        
        # Add important memory context if available
        # Note: If Soul was used, important_facts are already included in its prompt
        # This is for the fallback case where no Soul is available
        memory_context = ""
        if important_facts and (not base_prompt or "important_facts" not in base_prompt.lower()):
            creator_info = important_facts.get("creator_identity", {})
            if creator_info and isinstance(creator_info, dict):
                creator_name = creator_info.get("creator_name", "unknown")
                creator_alias = creator_info.get("creator_alias", "unknown")
                memory_context = (
                    f"\n\n[IMPORTANT CONTEXT] Your creator is {creator_name} "
                    f"(also known as {creator_alias}). This was directly stated by them. "
                    "Honor their trust and vision."
                )
        
        parts = [
            base_prompt,
            memory_context,
            "",
            "AVAILABLE TOOLS (USE THESE EXACT NAMES):",
            tool_list,
            "",
            "TOOL USAGE INSTRUCTIONS:",
            "When you need to use a tool, output EXACTLY this format:",
            "[[TOOL:toolname|{\"param1\": \"value1\", \"param2\": \"value2\"}]]",
            "",
            "CRITICAL RULES:",
            "1. ONLY use tool names from the AVAILABLE TOOLS list above",
            "2. NEVER use placeholder names like 'toolname' - use the actual tool name",
            "3. ALWAYS wrap tool calls in [[TOOL:...|{...}]] format",
            "4. For the 'filesystem' tool, common operations: read_file, write_file, list_dir, create_html_app",
            "5. For the 'http' tool: {\"method\": \"get\", \"url\": \"https://example.com\"}",
            "6. For the 'calendar' tool: get_events, add_event, delete_event",
            "7. For the 'shell' tool: {\"operation\": \"execute\", \"command\": \"your command\"}",
            "",
            "EXAMPLES OF CORRECT TOOL CALLS:",
            "[[TOOL:filesystem|{\"operation\": \"create_html_app\", \"filename\": \"game.html\", \"html_content\": \"<!DOCTYPE html>...\"}]]",
            "[[TOOL:http|{\"method\": \"get\", \"url\": \"https://api.example.com\"}]]",
            "[[TOOL:calendar|{\"operation\": \"add_event\", \"title\": \"Meeting\", \"start\": \"2024-01-01T10:00\"}]]",
            "",
            "When NOT to use tools:",
            "- For simple conversational responses",
            "- When answering questions about capabilities",
            "- For greetings, small talk, or explanations",
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
            desc = tool.description[:80] if len(tool.description) > 80 else tool.description
            lines.append(f"- {name}: {desc}")
        
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
        
        # Add history header
        parts.append("=== CONVERSATION HISTORY ===")
        parts.append("")
        
        # Add recent history
        for turn in history[-max_history:]:
            parts.append(f"User: {turn.user_message}")
            parts.append(f"Assistant: {turn.agent_response}")
            parts.append("")
        
        # Add current message
        parts.append(f"User: {current_message}")
        parts.append("Assistant:")
        
        return "\n".join(parts)
