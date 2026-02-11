"""
Tool execution, parsing, and file generation utilities.

Handles parsing LLM responses to extract and execute tool calls,
and generating HTML games/apps for direct execution.

CRITICAL: Uses ONLY XML format. Markdown JSON format is REJECTED.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from lollmsbot.agent.config import Tool, ToolResult


class ToolParser:
    """Parses LLM responses to extract and execute tool calls.
    
    ONLY accepts XML format with <tool>, <method>, <url>, <query> tags.
    Markdown JSON format is REJECTED with a warning to enforce correct format.
    """
    
    # Pattern for XML tool calls - ONLY valid format
    XML_TOOL_PATTERN = r'<tool>([^<]+)</tool>'
    XML_METHOD_PATTERN = r'<method>([^<]+)</method>'
    XML_URL_PATTERN = r'<url>([^<]+)</url>'
    XML_QUERY_PATTERN = r'<query>(.*?)</query>'
    XML_OPERATION_PATTERN = r'<operation>([^<]+)</operation>'
    XML_PATH_PATTERN = r'<path>([^<]+)</path>'
    XML_FILENAME_PATTERN = r'<filename>([^<]+)</filename>'
    XML_HTML_CONTENT_PATTERN = r'<html_content>(.*?)</html_content>'
    XML_TITLE_PATTERN = r'<title>([^<]+)</title>'
    XML_START_PATTERN = r'<start>([^<]+)</start>'
    XML_END_PATTERN = r'<end>([^<]+)</end>'
    XML_DESCRIPTION_PATTERN = r'<description>([^<]+)</description>'
    XML_DATA_PATTERN = r'<data>(.*?)</data>'
    XML_HEADERS_PATTERN = r'<headers>(.*?)</headers>'
    XML_PARAMS_PATTERN = r'<params>(.*?)</params>'
    
    # Pattern to DETECT JSON markdown format (for rejection/warning)
    JSON_TOOL_PATTERN = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    
    # Pattern to detect JSON inside XML (malformed)
    JSON_IN_XML_PATTERN = r'<tool>\s*(\{[^}]+\})\s*</tool>'
    
    # Old native format (also rejected)
    NATIVE_PATTERN = r'\[\[TOOL:(\w+)\|(\{.*?\})\]\]'
    
    def __init__(self, tools: Dict[str, Tool]) -> None:
        self.tools = tools
        self._json_warnings: List[str] = []  # Track JSON attempts for feedback
        self._last_tool_result: Optional["ToolResult"] = None  # Store last result for debug
        self._last_tool_name: Optional[str] = None  # Store last tool name for debug
    
    async def parse_and_execute(
        self,
        llm_response: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Parse LLM response and execute any embedded tool calls.
        
        ONLY accepts XML format with proper tag structure.
        Markdown JSON format is rejected with warning to enforce schema compliance.
        
        Returns:
            Tuple of (cleaned_response, tools_used, files_generated)
        """
        tools_used: List[str] = []
        files_generated: List[Dict[str, Any]] = []
        
        # First: Check for JSON format attempts and inject warnings
        json_warnings = self._detect_json_attempts(llm_response)
        
        # Check for JSON inside XML (malformed but common LLM error)
        json_in_xml = self._detect_json_in_xml(llm_response)
        
        # Find XML tool calls (only valid format)
        xml_matches = self._find_xml_tool_calls(llm_response)
        
        # Also check for old [[TOOL:...]] format
        native_matches = self._find_native_tool_calls(llm_response)
        
        # Combine all matches, sorted by position
        all_matches = sorted(xml_matches + native_matches, key=lambda x: x[0])
        
        if not all_matches and not json_warnings and not json_in_xml:
            # No tool calls found, return response as-is
            return llm_response.strip(), tools_used, files_generated
        
        # Execute each tool call and build cleaned response
        response_parts = []
        last_end = 0
        
        for start, end, format_type, tool_data in all_matches:
            # Add text before this tool call
            response_parts.append(llm_response[last_end:start])
            
            # Handle native format (convert to tool call but warn)
            if format_type == 'native':
                tool_name = tool_data.get("tool", "").lower()
                # Add warning about deprecated format
                response_parts.append(
                    "\n\n**⚠️ WARNING: Using deprecated [[TOOL:...]] format. "
                    "Use XML format instead.**\n\n"
                )
            
            # Extract tool name - STRICT VALIDATION
            tool_name = tool_data.get("tool", "").lower().strip()
            
            # CRITICAL: Reject if tool name contains JSON artifacts or is malformed
            if not tool_name or '{' in tool_name or '"' in tool_name or ':' in tool_name:
                # This is a malformed tool call - reject it entirely
                response_parts.append(
                    "\n\n**❌ MALFORMED TOOL CALL REJECTED**\n\n"
                    f"The tool call contained invalid characters in the tool name: '{tool_name[:50]}...'\n"
                    "Tool names must be simple strings like 'http', 'filesystem', 'calendar'.\n"
                    "Do NOT put JSON inside XML tags.\n\n"
                    "**Correct format:**\n"
                    "```xml\n<tool>http</tool>\n<method>get</method>\n<url>https://example.com</url>\n```\n\n"
                )
                last_end = end
                continue
            
            if not tool_name:
                last_end = end
                continue
            
            # Check if tool exists
            if tool_name not in self.tools:
                response_parts.append(
                    f"\n\n**⚠️ Unknown tool '{tool_name}'. "
                    f"Available: {', '.join(self.tools.keys())}**\n\n"
                )
                last_end = end
                continue
            
            # Execute the tool
            tool = self.tools[tool_name]
            result = await tool.execute(**tool_data)
            
            # Store for debug
            self._last_tool_result = result
            self._last_tool_name = tool_name
            
            if result.success:
                tools_used.append(tool_name)
                if result.files_to_send:
                    files_generated.extend(result.files_to_send)
                
                # Simple indicator - no details
                response_parts.append("\n\n**🔧 called tool**\n\n")
                
                # For HTTP with RLM memory handle, silent (no user-facing info)
                if tool_name == "http" and isinstance(result.output, dict):
                    if "memory_handle" in result.output:
                        pass  # RLM handles this internally
            
            else:
                # Tool failed - minimal error
                response_parts.append("\n\n**⚠️ tool error**\n\n")
            
            last_end = end
        
        # Add remaining text
        response_parts.append(llm_response[last_end:])
        
        # Add JSON warnings at the start if any were detected
        warning_text = ""
        if json_warnings:
            warning_text += (
                "\n\n**⚠️ JSON FORMAT REJECTED**\n\n"
                "You attempted to use JSON markdown format.\n"
                "This format is NOT SUPPORTED.\n\n"
            )
        
        if json_in_xml:
            warning_text += (
                "\n\n**⚠️ MALFORMED XML DETECTED - JSON INSIDE XML TAGS**\n\n"
                "You put JSON content inside XML tags like <tool>{...}</tool>.\n"
                "This is WRONG. XML tags must contain PLAIN TEXT values only.\n\n"
            )
        
        if warning_text:
            warning_text += (
                "**Use ONLY this format:**\n"
                "```xml\n"
                "<tool>http</tool>\n"
                "<method>get</method>\n"
                "<url>https://example.com</url>\n"
                "```\n\n"
                "**NOT this:**\n"
                "```xml\n"
                '<tool>{"tool_type": "http", ...}</tool>  ❌ WRONG - JSON inside XML\n'
                "```\n\n"
                "**Key rules:**\n"
                "1. <tool>http</tool> - just the tool name, NO JSON, NO braces\n"
                "2. <method>get</method> - just the method name\n"
                "3. <url>https://...</url> - just the URL string\n"
                "4. NO curly braces {} inside any XML tag\n"
                "5. NO double quotes around values inside XML tags\n\n"
                "Please retry with correct XML format.\n\n"
            )
            response_parts.insert(0, warning_text)
        
        # Clean up the response
        final_response = self._clean_response("".join(response_parts))
        
        return final_response, tools_used, files_generated
    
    def _detect_json_attempts(self, text: str) -> List[Dict[str, Any]]:
        """Detect JSON format attempts for rejection."""
        warnings = []
        
        # Check for ```json blocks
        for match in re.finditer(self.JSON_TOOL_PATTERN, text, re.DOTALL):
            warnings.append({
                "type": "json_block",
                "position": match.start(),
                "content": match.group(0)[:100]
            })
        
        return warnings
    
    def _detect_json_in_xml(self, text: str) -> List[Dict[str, Any]]:
        """Detect JSON content inside XML tags (malformed)."""
        warnings = []
        
        # Check for { inside <tool> tags
        for match in re.finditer(self.JSON_IN_XML_PATTERN, text, re.DOTALL):
            warnings.append({
                "type": "json_in_xml",
                "position": match.start(),
                "content": match.group(0)[:100]
            })
        
        # Also check for any { inside other common XML tags
        json_in_tags = [
            (r'<method>\s*\{', 'method'),
            (r'<url>\s*\{', 'url'),
            (r'<operation>\s*\{', 'operation'),
        ]
        for pattern, tag_name in json_in_tags:
            for match in re.finditer(pattern, text, re.DOTALL):
                warnings.append({
                    "type": f"json_in_{tag_name}",
                    "position": match.start(),
                    "content": f"JSON inside <{tag_name}> tag"
                })
        
        return warnings
    
    def _find_xml_tool_calls(self, text: str) -> List[Tuple[int, int, str, Dict[str, Any]]]:
        """Find XML tool calls in the response."""
        matches = []
        
        # Find all <tool> tags
        for tool_match in re.finditer(self.XML_TOOL_PATTERN, text, re.IGNORECASE):
            tool_content = tool_match.group(1).strip()
            start_pos = tool_match.start()
            end_pos = tool_match.end()
            
            # CRITICAL: Reject if tool content contains JSON markers
            if '{' in tool_content or '}' in tool_content or '"tool_type"' in tool_content or '"tool_input"' in tool_content:
                # This is a malformed JSON-in-XML call - mark it but don't try to parse
                # It will be rejected later when we validate the tool name
                pass  # Let it through to be rejected by tool name validation
            
            # Look for related tags after this tool tag
            remaining_text = text[end_pos:]
            
            # Clean up the tool name - remove any JSON artifacts
            tool_name = tool_content.lower()
            # Remove any JSON-like artifacts
            tool_name = re.sub(r'[{}"]', '', tool_name)
            tool_name = re.sub(r'tool_type\s*:', '', tool_name)
            tool_name = re.sub(r'tool_input\s*:', '', tool_name)
            tool_name = tool_name.strip()
            
            # Additional cleanup: remove any remaining non-word characters
            tool_name = re.sub(r'[^\w]', '', tool_name)
            
            tool_data = {"tool": tool_name}
            
            # Find <method> if present
            method_match = re.search(self.XML_METHOD_PATTERN, remaining_text, re.IGNORECASE)
            if method_match:
                method_value = method_match.group(1).strip()
                # Clean up method value - remove JSON artifacts
                method_value = re.sub(r'[{}"]', '', method_value)
                tool_data["method"] = method_value.lower()
                end_pos = max(end_pos, tool_match.end() + method_match.end())
            
            # Find <url> if present
            url_match = re.search(self.XML_URL_PATTERN, remaining_text, re.IGNORECASE)
            if url_match:
                url_value = url_match.group(1).strip()
                # Clean up URL - remove JSON artifacts and quotes
                url_value = re.sub(r'^[\'"]', '', url_value)
                url_value = re.sub(r'[\'"]$', '', url_value)
                tool_data["url"] = url_value
                end_pos = max(end_pos, tool_match.end() + url_match.end())
            
            # Find <query> if present
            query_match = re.search(self.XML_QUERY_PATTERN, remaining_text, re.DOTALL | re.IGNORECASE)
            if query_match:
                tool_data["query"] = query_match.group(1)
                end_pos = max(end_pos, tool_match.end() + query_match.end())
            
            # Find <operation> if present
            operation_match = re.search(self.XML_OPERATION_PATTERN, remaining_text, re.IGNORECASE)
            if operation_match:
                tool_data["operation"] = operation_match.group(1).strip().lower()
                end_pos = max(end_pos, tool_match.end() + operation_match.end())
            
            # Find <path> if present
            path_match = re.search(self.XML_PATH_PATTERN, remaining_text, re.IGNORECASE)
            if path_match:
                tool_data["path"] = path_match.group(1).strip()
                end_pos = max(end_pos, tool_match.end() + path_match.end())
            
            # Find <filename> if present
            filename_match = re.search(self.XML_FILENAME_PATTERN, remaining_text, re.IGNORECASE)
            if filename_match:
                tool_data["filename"] = filename_match.group(1).strip()
                end_pos = max(end_pos, tool_match.end() + filename_match.end())
            
            # Find <html_content> if present
            html_content_match = re.search(self.XML_HTML_CONTENT_PATTERN, remaining_text, re.DOTALL | re.IGNORECASE)
            if html_content_match:
                tool_data["html_content"] = html_content_match.group(1)
                end_pos = max(end_pos, tool_match.end() + html_content_match.end())
            
            # Find <title> if present
            title_match = re.search(self.XML_TITLE_PATTERN, remaining_text, re.IGNORECASE)
            if title_match:
                tool_data["title"] = title_match.group(1).strip()
                end_pos = max(end_pos, tool_match.end() + title_match.end())
            
            # Find <start> if present
            start_match = re.search(self.XML_START_PATTERN, remaining_text, re.IGNORECASE)
            if start_match:
                tool_data["start"] = start_match.group(1).strip()
                end_pos = max(end_pos, tool_match.end() + start_match.end())
            
            # Find <end> if present
            end_match = re.search(self.XML_END_PATTERN, remaining_text, re.IGNORECASE)
            if end_match:
                tool_data["end"] = end_match.group(1).strip()
                end_pos = max(end_pos, tool_match.end() + end_match.end())
            
            # Find <description> if present
            description_match = re.search(self.XML_DESCRIPTION_PATTERN, remaining_text, re.IGNORECASE)
            if description_match:
                tool_data["description"] = description_match.group(1).strip()
                end_pos = max(end_pos, tool_match.end() + description_match.end())
            
            # Find <data> if present
            data_match = re.search(self.XML_DATA_PATTERN, remaining_text, re.DOTALL | re.IGNORECASE)
            if data_match:
                tool_data["data"] = data_match.group(1)
                end_pos = max(end_pos, tool_match.end() + data_match.end())
            
            # Calculate overall end position - include all related tags
            # Find the furthest tag end
            furthest_end = tool_match.end()
            for tag_pattern in [
                self.XML_METHOD_PATTERN, self.XML_URL_PATTERN, self.XML_QUERY_PATTERN,
                self.XML_OPERATION_PATTERN, self.XML_PATH_PATTERN, self.XML_FILENAME_PATTERN,
                self.XML_HTML_CONTENT_PATTERN, self.XML_TITLE_PATTERN, self.XML_START_PATTERN,
                self.XML_END_PATTERN, self.XML_DESCRIPTION_PATTERN, self.XML_DATA_PATTERN,
                self.XML_HEADERS_PATTERN, self.XML_PARAMS_PATTERN,
            ]:
                tag_match = re.search(tag_pattern, remaining_text, re.DOTALL | re.IGNORECASE)
                if tag_match:
                    furthest_end = max(furthest_end, tool_match.end() + tag_match.end())
            
            matches.append((start_pos, furthest_end, 'xml', tool_data))
        
        return matches
    
    def _find_native_tool_calls(self, text: str) -> List[Tuple[int, int, str, Dict[str, Any]]]:
        """Find old [[TOOL:...|{}]] format (deprecated but handled)."""
        matches = []
        
        for match in re.finditer(self.NATIVE_PATTERN, text, re.DOTALL):
            tool_name = match.group(1).lower()
            json_content = match.group(2)
            
            try:
                tool_data = json.loads(json_content)
                tool_data["tool"] = tool_name  # Ensure tool field is set
                matches.append((match.start(), match.end(), 'native', tool_data))
            except json.JSONDecodeError:
                # Try with just the tool name
                matches.append((match.start(), match.end(), 'native', {
                    "tool": tool_name,
                    "content": json_content
                }))
        
        return matches
    
    async def _execute_tool(
        self,
        tool: Any,
        tool_name: str,
        params: Dict[str, Any],
    ) -> "ToolResult":
        """Execute a single tool with error handling."""
        try:
            result = await tool.execute(**params)
            return result
        except Exception as e:
            from lollmsbot.agent.config import ToolResult
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )
    
    def _clean_response(self, text: str) -> str:
        """Clean all tool call syntax from user-facing response."""
        # Remove XML tool tags and their content
        text = re.sub(
            r'<tool>[^<]*</tool>\s*(?:<method>[^<]*</method>)?\s*(?:<url>[^<]*</url>)?(?:<query>.*?</query>)?\s*(?:<operation>[^<]*</operation>)?\s*(?:<path>[^<]*</path>)?\s*(?:<filename>[^<]*</filename>)?\s*(?:<html_content>.*?</html_content>)?\s*(?:<title>[^<]*</title>)?\s*(?:<start>[^<]*</start>)?\s*(?:<end>[^<]*</end>)?\s*(?:<description>[^<]*</description>)?\s*(?:<data>.*?</data>)?',
            '',
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # Remove JSON code blocks that contain tool calls
        def remove_tool_blocks(match: re.Match) -> str:
            try:
                data = json.loads(match.group(1))
                if "tool" in data or "tool_type" in data:
                    return ""
            except:
                pass
            return match.group(0)
        
        text = re.sub(self.JSON_TOOL_PATTERN, remove_tool_blocks, text, flags=re.DOTALL)
        
        # Remove native format
        text = re.sub(self.NATIVE_PATTERN, '', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up empty lines at start/end
        lines = [line for line in text.split('\n') if line.strip()]
        return '\n'.join(lines).strip()


class FileGenerator:
    """Generates HTML games and apps for direct tool execution."""
    
    FILE_KEYWORDS = [
        "create", "make", "build", "generate", "write", "save",
        "file", "html", "game", "app", "script", "code",
        ".html", ".js", ".css", ".py", ".txt", ".json"
    ]
    
    HTML_KEYWORDS = ["html", "game", "app", "page", "website"]
    
    FILE_REQUEST_KEYWORDS = ["create file", "make file", "write file", "save file", "generate file"]
    
    def __init__(self, tools: Dict[str, Tool]) -> None:
        self.tools = tools
        self._last_tool_result: Optional["ToolResult"] = None
        self._last_tool_name: Optional[str] = None
    
    def is_file_request(self, message: str) -> bool:
        """Check if message appears to be a file generation request."""
        message_lower = message.lower()
        return any(kw in message_lower for kw in self.FILE_KEYWORDS)
    
    def is_html_request(self, message: str) -> bool:
        """Check if message is specifically for HTML/game creation."""
        message_lower = message.lower()
        return any(kw in message_lower for kw in self.HTML_KEYWORDS)
    
    async def try_direct_execution(
        self,
        message: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Try to directly execute tools based on message patterns without LLM."""
        message_lower = message.lower()
        
        if self.is_html_request(message) and "filesystem" in self.tools:
            game_type = self._detect_game_type(message_lower)
            filename = f"{game_type.replace(' ', '_')}.html"
            
            html_content = self._generate_html_game(game_type)
            
            tool = self.tools["filesystem"]
            result = await tool.execute(
                operation="create_html_app",
                filename=filename,
                html_content=html_content,
            )
            
            self._last_tool_result = result
            self._last_tool_name = "filesystem"
            
            if result.success:
                files = result.files_to_send or []
                file_names = [f.get("filename", "unnamed") for f in files]
                return f"I've created a {game_type} for you! The file is ready: {', '.join(file_names)}"
            else:
                return f"I tried to create the {game_type} but encountered an error: {result.error}"
        
        if any(kw in message_lower for kw in self.FILE_REQUEST_KEYWORDS):
            return None
        
        return None
    
    def _detect_game_type(self, message_lower: str) -> str:
        """Detect what kind of game/app from message."""
        if "snake" in message_lower:
            return "snake game"
        elif "pong" in message_lower:
            return "pong game"
        elif "tetris" in message_lower:
            return "tetris game"
        elif "calculator" in message_lower:
            return "calculator"
        elif "todo" in message_lower:
            return "todo app"
        elif "star" in message_lower:
            return "catch_the_stars"
        return "game"
    
    def _generate_html_game(self, game_type: str) -> str:
        """Generate HTML5 game content based on type."""
        generators = {
            "snake game": self._generate_snake_game,
            "pong game": self._generate_pong_game,
            "catch_the_stars": self._generate_catch_stars_game,
        }
        
        generator = generators.get(game_type, self._generate_default_app)
        return generator()
    
    def _generate_snake_game(self) -> str:
        """Generate Snake game HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Game</title>
    <style>
        body { display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #1a1a2e; font-family: Arial, sans-serif; }
        canvas { border: 2px solid #0f3460; background: #16213e; }
        .info { color: #fff; text-align: center; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div>
        <div class="info"><h1>🐍 Snake Game</h1><p>Use arrow keys to play</p></div>
        <canvas id="game" width="400" height="400"></canvas>
    </div>
    <script>
        const canvas = document.getElementById('game');
        const ctx = canvas.getContext('2d');
        const grid = 20;
        let snake = [{x: 10, y: 10}];
        let food = {x: 15, y: 15};
        let dx = 1, dy = 0;
        let score = 0;
        function draw() {
            ctx.fillStyle = '#16213e';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#0f0';
            snake.forEach(s => ctx.fillRect(s.x*grid, s.y*grid, grid-2, grid-2));
            ctx.fillStyle = '#f00';
            ctx.fillRect(food.x*grid, food.y*grid, grid-2, grid-2);
            ctx.fillStyle = '#fff';
            ctx.font = '20px Arial';
            ctx.fillText('Score: ' + score, 10, 30);
        }
        function update() {
            const head = {x: snake[0].x + dx, y: snake[0].y + dy};
            if (head.x < 0) head.x = 19; if (head.x > 19) head.x = 0;
            if (head.y < 0) head.y = 19; if (head.y > 19) head.y = 0;
            if (snake.some(s => s.x === head.x && s.y === head.y)) {
                snake = [{x: 10, y: 10}]; score = 0; return;
            }
            snake.unshift(head);
            if (head.x === food.x && head.y === food.y) {
                score += 10;
                food = {x: Math.floor(Math.random()*20), y: Math.floor(Math.random()*20)};
            } else { snake.pop(); }
        }
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp' && dy === 0) { dx = 0; dy = -1; }
            if (e.key === 'ArrowDown' && dy === 0) { dx = 0; dy = 1; }
            if (e.key === 'ArrowLeft' && dx === 0) { dx = -1; dy = 0; }
            if (e.key === 'ArrowRight' && dx === 0) { dx = 1; dy = 0; }
        });
        setInterval(() => { update(); draw(); }, 100);
        draw();
    </script>
</body>
</html>'''
    
    def _generate_pong_game(self) -> str:
        """Generate Pong game HTML."""
        return '''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Pong Game</title>
<style>body{display:flex;justify-content:center;align-items:center;min-height:100vh;margin:0;background:#1a1a2e;}canvas{border:2px solid #0f3460;background:#16213e;}</style>
</head><body><canvas id="game" width="600" height="400"></canvas>
<script>
const canvas=document.getElementById('game'),ctx=canvas.getContext('2d');
let ball={x:300,y:200,dx:4,dy:4,radius:10};
let paddle1={x:10,y:150,width:10,height:100};
let paddle2={x:580,y:150,width:10,height:100};
let score1=0,score2=0;
function draw(){
    ctx.fillStyle='#16213e';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.beginPath();ctx.arc(ball.x,ball.y,ball.radius,0,Math.PI*2);
    ctx.fillStyle='#fff';ctx.fill();
    ctx.fillStyle='#0f0';ctx.fillRect(paddle1.x,paddle1.y,paddle1.width,paddle1.height);
    ctx.fillStyle='#f00';ctx.fillRect(paddle2.x,paddle2.y,paddle2.width,paddle2.height);
    ctx.fillStyle='#fff';ctx.font='30px Arial';ctx.fillText(score1+' - '+score2,270,40);
}
function update(){
    ball.x+=ball.dx;ball.y+=ball.dy;
    if(ball.y<ball.radius||ball.y>canvas.height-ball.radius)ball.dy*=-1;
    if(paddle2.y+paddle2.height/2<ball.y)paddle2.y+=3;
    if(paddle2.y+paddle2.height/2>ball.y)paddle2.y-=3;
    if(ball.x-ball.radius<paddle1.x+paddle1.width&&ball.y>paddle1.y&&ball.y<paddle1.y+paddle1.height)ball.dx=Math.abs(ball.dx);
    if(ball.x+ball.radius>paddle2.x&&ball.y>paddle2.y&&ball.y<paddle2.y+paddle2.height)ball.dx=-Math.abs(ball.dx);
    if(ball.x<0){score2++;ball={x:300,y:200,dx:4,dy:4,radius:10};}
    if(ball.x>canvas.width){score1++;ball={x:300,y:200,dx:-4,dy:4,radius:10};}
}
document.addEventListener('mousemove',(e)=>{const rect=canvas.getBoundingClientRect();paddle1.y=e.clientY-rect.top-paddle1.height/2;});
setInterval(()=>{update();draw();},16);draw();
</script></body></html>'''
    
    def _generate_catch_stars_game(self) -> str:
        """Generate Catch the Stars game HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Catch the Falling Stars</title>
<style>*{margin:0;padding:0;box-sizing:border-box;}body{display:flex;justify-content:center;align-items:center;min-height:100vh;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);font-family:'Segoe UI',sans-serif;overflow:hidden;}#gameContainer{position:relative;width:800px;height:600px;}#gameCanvas{border-radius:15px;box-shadow:0 0 30px rgba(233,69,96,0.3);background:linear-gradient(180deg,#0a0a1a 0%,#1a0a2e 100%);}#ui{position:absolute;top:0;left:0;width:100%;padding:20px;display:flex;justify-content:space-between;pointer-events:none;}.ui-element{background:rgba(0,0,0,0.6);padding:10px 20px;border-radius:25px;color:#fff;font-size:18px;font-weight:bold;}#score{color:#ffd700;}#lives{color:#e94560;}#startScreen{position:absolute;top:0;left:0;width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;align-items:center;background:rgba(0,0,0,0.85);border-radius:15px;}h1{color:#ffd700;font-size:48px;margin-bottom:10px;}p{color:#aaa;font-size:18px;margin-bottom:30px;}.btn{padding:15px 40px;font-size:20px;font-weight:bold;border:none;border-radius:50px;cursor:pointer;background:linear-gradient(45deg,#e94560,#ff6b6b);color:white;}</style>
</head>
<body>
<div id="gameContainer">
<canvas id="gameCanvas" width="800" height="600"></canvas>
<div id="ui"><div class="ui-element" id="score">⭐ Score: 0</div><div class="ui-element" id="lives">❤️ Lives: 3</div></div>
<div id="startScreen"><h1>⭐ Catch the Stars ⭐</h1><p>Move your mouse to control the basket.<br>Catch golden stars for points (+10)<br>Avoid red bombs or lose a life!</p><button class="btn" onclick="startGame()">Start Game</button></div>
</div>
<script>
const canvas=document.getElementById('gameCanvas'),ctx=canvas.getContext('2d');
const basket={x:350,y:520,width:100,height:60};
let mouseX=400,score=0,lives=3,gameRunning=false,objects=[];
canvas.addEventListener('mousemove',(e)=>{const rect=canvas.getBoundingClientRect();mouseX=e.clientX-rect.left;});
function startGame(){gameRunning=true;document.getElementById('startScreen').style.display='none';loop();}
function drawBasket(){ctx.fillStyle='#e94560';ctx.fillRect(basket.x,basket.y,basket.width,basket.height);}
function update(){
    if(!gameRunning)return;
    basket.x=mouseX-basket.width/2;basket.x=Math.max(0,Math.min(canvas.width-basket.width,basket.x));
    if(Math.random()<0.02){objects.push({x:Math.random()*(canvas.width-30)+15,y:-30,size:20,speed:2+Math.random(),type:Math.random()<0.2?'bomb':'star'});}
    objects.forEach(obj=>{
        obj.y+=obj.speed;
        if(obj.y+obj.size>basket.y&&obj.x>basket.x&&obj.x<basket.x+basket.width){
            if(obj.type==='star')score+=10;else lives--;obj.collected=true;
        }
    });
    objects=objects.filter(obj=>!obj.collected&&obj.y<canvas.height+50);
    if(lives<=0){gameRunning=false;document.getElementById('startScreen').style.display='flex';}
    document.getElementById('score').textContent='⭐ Score: '+score;
    document.getElementById('lives').textContent='❤️ Lives: '+lives;
}
function draw(){
    ctx.fillStyle='#0a0a1a';ctx.fillRect(0,0,canvas.width,canvas.height);
    objects.forEach(obj=>{ctx.beginPath();ctx.arc(obj.x,obj.y,obj.size,0,Math.PI*2);ctx.fillStyle=obj.type==='star'?'#ffd700':'#ff3333';ctx.fill();});
    drawBasket();
    if(gameRunning)requestAnimationFrame(loop);
}
function loop(){update();draw();}
draw();
</script>
</body>
</html>'''
    
    def _generate_default_app(self) -> str:
        """Generate default simple HTML app."""
        return '''<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>My App</title>
<style>body{font-family:system-ui,sans-serif;text-align:center;padding:50px;background:linear-gradient(135deg,#1a1a2e,#0f3460);color:white;min-height:100vh;}button{padding:15px 30px;font-size:18px;background:#e94560;color:white;border:none;border-radius:8px;cursor:pointer;}</style>
</head>
<body><h1>Welcome!</h1><p>This is your custom HTML app.</p><button onclick="alert('Hello!')">Click Me!</button></body>
</html>'''
    
    @property
    def last_tool_result(self) -> Optional["ToolResult"]:
        return self._last_tool_result
    
    @property
    def last_tool_name(self) -> Optional[str]:
        return self._last_tool_name
