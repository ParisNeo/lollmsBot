"""
Swarm Bridge Tool - Enables Inter-Agent Collaboration via local networking.
"""
from __future__ import annotations
import httpx
from typing import Any, Dict, List, Optional
from lollmsbot.agent import Tool, ToolResult

class SwarmBridgeTool(Tool):
    """Tool allowing one agent to consult another networked agent instance."""
    name: str = "swarm_consult"
    description: str = (
        "Consults another agent in the sovereign swarm. "
        "Use this to delegate tasks to specialized agents (Alpha, Beta, Gamma)."
    )
    
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "target_agent": {
                "type": "string", 
                "description": "Hostname or IP of the target agent (e.g., 'beta-agent', '127.0.0.1:9601')"
            },
            "message": {
                "type": "string", 
                "description": "The request to send to the other agent."
            }
        },
        "required": ["target_agent", "message"],
    }
    
    risk_level: str = "medium"
    
    async def execute(self, target_agent: str, message: str, **kwargs) -> ToolResult:
        # Determine port - if not specified, default to 9600
        host = target_agent if ":" in target_agent else f"{target_agent}:9600"
        url = f"http://{host}/chat"
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {"message": message, "user_id": "swarm_system"}
                resp = await client.post(url, json=payload)
                
                if resp.status_code == 200:
                    data = resp.json()
                    return ToolResult(
                        success=True,
                        output=f"[{target_agent} Response]: {data.get('response')}",
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Agent {target_agent} returned status {resp.status_code}"
                    )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to connect to agent {target_agent}: {str(e)}"
            )