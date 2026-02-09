"""
Tool execution, parsing, and file generation utilities.

Handles parsing LLM responses to extract and execute tool calls,
and generating HTML games/apps for direct execution.

CRITICAL: HTTP tool now returns memory handles via RLM. The LLM must
reference these handles to access content. This is not a bug - it's
the RLM architecture working as designed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from lollmsbot.agent.config import Tool, ToolResult


class ToolParser:
    """Parses LLM responses to extract and execute tool calls.
    
    IMPORTANT: For HTTP tool, the result is a MEMORY HANDLE, not raw content.
    The LLM must use the handle [[MEMORY:chunk_id]] to access content via RLM.
    This is the intended RLM architecture.
    """
    
    # Pattern for native [[TOOL:...]] format
    NATIVE_PATTERN = r'\[\[TOOL:(\w+)\|(\{.*?\})\]\]'
    
    # Pattern for XML-style function calls
    XML_PATTERN = r'<function_calls>\s*<invoke\s+name="([^"]+)">\s*(?:<parameter[^>]*>)?\s*<!\[CDATA\[(.*?)\]\]>\s*(?:</parameter>)?\s*</invoke>\s*</function_calls>'
    
    # Alternative XML without CDATA
    XML_PATTERN_ALT = r'<function_calls>\s*<invoke\s+name="([^"]+)">(.*?)</invoke>\s*</function_calls>'
    
    # Code block containing XML
    CODE_BLOCK_PATTERN = r'```(?:xml|json)?\s*(<function_calls>.*?</function_calls>)```'
    
    def __init__(self, tools: Dict[str, Tool]) -> None:
        self.tools = tools
    
    async def parse_and_execute(
        self,
        llm_response: str,
        user_id: str,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Parse LLM response and execute any embedded tool calls.
        
        For HTTP tool: The result includes a memory handle. The LLM should
        reference this handle to access content. We add a note explaining this.
        
        Returns:
            Tuple of (cleaned_response, tools_used, files_generated)
        """
        tools_used: List[str] = []
        files_generated: List[Dict[str, Any]] = []
        
        # Find all tool calls
        all_matches = self._find_tool_calls(llm_response)
        
        if not all_matches:
            # No tool calls found, return response as-is
            return llm_response.strip(), tools_used, files_generated
        
        # Execute each tool call and build cleaned response
        response_parts = []
        last_end = 0
        
        for start, end, format_type, tool_name, content in all_matches:
            # Add text before this tool call
            response_parts.append(llm_response[last_end:start])
            
            # Parse parameters based on format
            tool_params = self._parse_parameters(format_type, content)
            
            # Validate tool exists
            if tool_name not in self.tools:
                if tool_name in ("toolname", "tool"):
                    last_end = end
                    continue
                last_end = end
                continue
            
            # Execute the tool
            tool = self.tools[tool_name]
            result = await self._execute_tool(tool, tool_name, tool_params)
            
            if result.success:
                tools_used.append(tool_name)
                if result.files_to_send:
                    files_generated.extend(result.files_to_send)
                
                # HTTP TOOL: Result is a memory handle via RLM
                if tool_name == "http" and result.output:
                    # The HTTP tool returns a dict with memory_handle, chunk_id, etc.
                    if isinstance(result.output, dict) and "memory_handle" in result.output:
                        handle = result.output.get("memory_handle", "[[MEMORY:unknown]]")
                        summary = result.output.get("summary", "Web content")
                        url = result.output.get("url", "unknown URL")
                        
                        # Inform the LLM about the memory handle
                        # The content is now in RLM and accessible via this handle
                        response_parts.append(
                            f"\n\n[HTTP FETCH COMPLETE - Content stored in RLM memory]\n"
                            f"URL: {url}\n"
                            f"Memory handle: {handle}\n"
                            f"Summary: {summary[:100]}...\n"
                            f"To access this content, reference the memory handle in your reasoning.\n"
                        )
                    else:
                        # Fallback for old format
                        response_parts.append(f"\n\n[HTTP request completed]\n")
                
                # For other tools with structured output, add summary
                elif tool_name != "http":
                    output_preview = self._format_tool_output_preview(result.output, tool_name)
                    if output_preview:
                        response_parts.append(f"\n\n[{tool_name.upper()} RESULT]:\n{output_preview}\n\n")
            
            else:
                # Tool failed - include error for context
                error_msg = result.error or "Unknown error"
                response_parts.append(f"\n\n[{tool_name.upper()} ERROR: {error_msg}]\n\n")
            
            last_end = end
        
        # Add remaining text
        response_parts.append(llm_response[last_end:])
        
        # Clean up the response
        final_response = self._clean_response("".join(response_parts))
        
        return final_response, tools_used, files_generated
    
    def _format_tool_output_preview(self, output: Any, tool_name: str) -> str:
        """Create a readable preview of tool output for inclusion in response."""
        if output is None:
            return ""
        
        # Handle dict outputs
        if isinstance(output, dict):
            # If there's a 'content' key, use that
            if "content" in output:
                content = output["content"]
                if isinstance(content, str):
                    if len(content) > 2000:
                        return content[:2000] + f"\n... [truncated, total: {len(content)} chars]"
                    return content
                else:
                    return json.dumps(content, indent=2, ensure_ascii=False, default=str)
            
            # Format the whole dict nicely
            return json.dumps(output, indent=2, ensure_ascii=False, default=str)
        
        # Handle string outputs
        if isinstance(output, str):
            if len(output) > 2000:
                return output[:2000] + f"\n... [truncated, total: {len(output)} chars]"
            return output
        
        # Default
        return str(output)
    
    def _find_tool_calls(self, text: str) -> List[Tuple[int, int, str, str, str]]:
        """Find all tool calls in text with positions."""
        matches = []
        
        # Native format
        for match in re.finditer(self.NATIVE_PATTERN, text, re.DOTALL):
            matches.append((
                match.start(), match.end(), 'native',
                match.group(1).lower(), match.group(2)
            ))
        
        # XML format
        for match in re.finditer(self.XML_PATTERN, text, re.DOTALL | re.IGNORECASE):
            matches.append((
                match.start(), match.end(), 'xml',
                match.group(1).lower(), match.group(2)
            ))
        
        # Alternative XML
        for match in re.finditer(self.XML_PATTERN_ALT, text, re.DOTALL | re.IGNORECASE):
            if not any(m[0] == match.start() for m in matches):
                matches.append((
                    match.start(), match.end(), 'xml',
                    match.group(1).lower(), match.group(2)
                ))
        
        # Code blocks
        for block_match in re.finditer(self.CODE_BLOCK_PATTERN, text, re.DOTALL | re.IGNORECASE):
            inner = block_match.group(1)
            inner_match = re.search(
                r'<invoke\s+name="([^"]+)".*?>(.*?)</invoke>',
                inner, re.DOTALL | re.IGNORECASE
            )
            if inner_match:
                matches.append((
                    block_match.start(), block_match.end(), 'xml',
                    inner_match.group(1).lower(), inner_match.group(2)
                ))
        
        # Sort by position to maintain order
        matches.sort(key=lambda x: x[0])
        return matches
    
    def _parse_parameters(self, format_type: str, content: str) -> Dict[str, Any]:
        """Parse tool parameters from different formats."""
        tool_params: Dict[str, Any] = {}
        
        if format_type == 'xml':
            cdata_match = re.search(r'<!\[CDATA\[(.*?)\]\]>', content, re.DOTALL)
            if cdata_match:
                param_content = cdata_match.group(1)
                
                if "operation" not in content:
                    if "<!DOCTYPE html>" in param_content or "<html" in param_content:
                        tool_params["operation"] = "create_html_app"
                        tool_params["filename"] = "app.html"
                        tool_params["html_content"] = param_content
                    else:
                        tool_params["operation"] = "write_file"
                        tool_params["path"] = "output.txt"
                        tool_params["content"] = param_content
                else:
                    param_pattern = r'<parameter name="([^"]+)">\s*(?:<!\[CDATA\[(.*?)\]\]>|\s*([^<]*))\s*</parameter>'
                    for pmatch in re.finditer(param_pattern, content, re.DOTALL):
                        p_name = pmatch.group(1)
                        p_value = pmatch.group(2) if pmatch.group(2) else pmatch.group(3)
                        tool_params[p_name] = p_value.strip() if p_value else ""
            else:
                param_pattern = r'<parameter name="([^"]+)">(.*?)</parameter>'
                for pmatch in re.finditer(param_pattern, content, re.DOTALL):
                    p_name = pmatch.group(1)
                    p_value = pmatch.group(2).strip()
                    tool_params[p_name] = p_value
        
        elif format_type == 'native':
            try:
                tool_params = json.loads(content)
            except json.JSONDecodeError:
                pass
        
        return tool_params
    
    async def _execute_tool(
        self,
        tool: Any,
        tool_name: str,
        params: Dict[str, Any],
    ) -> ToolResult:
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
        """Clean tool call syntax from user-facing response."""
        # Remove all tool call formats
        text = re.sub(self.NATIVE_PATTERN, '', text)
        text = re.sub(r'<function_calls>.*?</function_calls>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up empty lines and whitespace
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
        self._last_tool_result: Optional[ToolResult] = None
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
    def last_tool_result(self) -> Optional[ToolResult]:
        return self._last_tool_result
    
    @property
    def last_tool_name(self) -> Optional[str]:
        return self._last_tool_name
