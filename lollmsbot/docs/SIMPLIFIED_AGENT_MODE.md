# SimplifiedAgant Mode for lollmsbot

This document describes the SimplifiedAgant-style minimal agent architecture integrated into lollmsbot.

## What is SimplifiedAgant Mode?

SimplifiedAgant mode transforms lollmsbot from a multi-tool agent into a **minimal, self-extending agent** inspired by [Pi](https://lucumr.pocoo.org/2026/1/31/pi/) and [SimplifiedAgant](https://github.com/VoltAgent/awesome-simplified_agant-skills).

### Core Philosophy

> "Software building software" — The agent maintains its own functionality by writing code.

Instead of downloading pre-built extensions, the agent **writes Python code** to extend itself. This creates a tight feedback loop where the agent can iterate on its own tools until they work.

## Architecture

### 4 Core Tools Only

| Tool | Purpose |
|------|---------|
| `read` | Read file contents with offset/limit |
| `write` | Write or append to files |
| `edit` | Replace text in files (find/replace) |
| `bash` | Execute shell commands |

### Self-Written Extensions (Skills)

Extensions are Python files stored in `~/.lollmsbot/simplified_agant/.extensions/`. Each extension has an `execute()` function:

```python
# Example: calculator extension
def execute(expression: str):
    import math
    try:
        result = eval(expression, {"__builtins__": {}}, math.__dict__)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
```

### Session Tree Structure

Conversations are organized as a tree:

```
main (trunk)
├── branch_1_experiment
│   └── branch_1_1_deeper
└── branch_2_fix
```

- **Branch**: Create experimental contexts without polluting main
- **Switch**: Move between branches
- **Merge**: Summarize and integrate branch changes back to parent

### Hot Reload

Extensions can be modified and reloaded **instantly** without restarting:

```python
# Agent writes extension
<tool>simplified_agant</tool>
<operation>create_extension</operation>
<extension_name>web_fetcher</extension_name>
<content>
import requests
def execute(url: str):
    r = requests.get(url, timeout=10)
    return {"status": r.status_code, "content": r.text[:1000]}
</content>

# Later, agent improves it and reloads
<tool>simplified_agant</tool>
<operation>hot_reload</operation>
<extension_name>web_fetcher</extension_name>
```

## Usage

### Enable SimplifiedAgant Mode

```bash
# Console chat with SimplifiedAgant mode
lollmsbot chat --simplified_agant

# Or programmatically
from lollmsbot.agent import integrate_openclaw

openclaw_agent, openclaw_tool = integrate_openclaw(your_agent)
await your_agent.register_tool(openclaw_tool)
```

### Creating Extensions

The agent creates extensions by writing Python code:

```xml
<tool>simplified_agant</tool>
<operation>create_extension</operation>
<extension_name>todo_manager</extension_name>
<extension_description>Manage todo items in a markdown file</extension_description>
<content>
import json
from pathlib import Path

TODO_FILE = Path.home() / ".lollmsbot" / "simplified_agant" / "todos.md"

def execute(operation: str = "list", item: str = ""):
    TODO_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if operation == "list":
        if not TODO_FILE.exists():
            return {"todos": []}
        content = TODO_FILE.read_text()
        todos = [line for line in content.split('\n') if line.strip().startswith('- [')]
        return {"todos": todos, "count": len(todos)}
    
    elif operation == "add":
        with open(TODO_FILE, 'a') as f:
            f.write(f"- [ ] {item}\\n")
        return {"added": item}
    
    elif operation == "done":
        if not TODO_FILE.exists():
            return {"error": "No todos file"}
        content = TODO_FILE.read_text()
        new_content = content.replace(f"- [ ] {item}", f"- [x] {item}")
        TODO_FILE.write_text(new_content)
        return {"marked_done": item}
    
    return {"error": f"Unknown operation: {operation}"}
</content>
```

### Session Branching

```bash
# In console chat
> /branch experiment with new extension

# Agent can now experiment freely
# If it breaks something, switch back to main
> /switch main

# If experiment succeeds, merge it
> /merge branch_id
```

## Comparison: Standard vs SimplifiedAgant Mode

| Feature | Standard lollmsbot | SimplifiedAgant Mode |
|---------|-------------------|---------------|
| Tools | 10+ pre-built | 4 core + self-written |
| Extensions | Download/install | Write Python code |
| Session | Linear history | Tree-structured |
| Learning | Static skills | Dynamic self-extension |
| Philosophy | Swiss Army knife | Minimal core, infinite growth |

## Security Considerations

SimplifiedAgant mode grants the agent **code execution capabilities**:

1. **Bash commands** are sandboxed with timeout and dangerous command filtering
2. **Python extensions** run in isolated namespaces
3. **File access** is restricted to the working directory
4. **Review before merge** - branches allow inspection before integration

## Integration with lollmsbot Features

SimplifiedAgant mode works alongside lollmsbot's existing features:

- **RLM Memory**: Extensions and session history are stored in the memory system
- **Project Memory**: Each branch can have its own project context
- **Guardian**: Security screening applies to extension code
- **Skills**: Traditional skills can call SimplifiedAgant tools

## Example Workflows

### 1. Research Assistant

```
User: I need to track research papers I read

Agent: <tool>simplified_agant</tool>
<operation>create_extension</operation>
<extension_name>paper_tracker</extension_name>
...

[Agent writes paper_tracker extension]

Agent: Now you can use: <tool>paper_tracker</tool><operation>add</operation>...
```

### 2. Bug Fix Branch

```
User: The web_fetcher extension is timing out

Agent: <tool>simplified_agant</tool><operation>branch</operation>
[Creates branch: fix_web_fetcher_timeout]

Agent: <tool>read</tool><path>.extensions/web_fetcher.py</path>
[Reads code]

Agent: <tool>edit</tool>...
[Fixes timeout handling]

Agent: <tool>simplified_agant</tool><operation>hot_reload</operation>
[Tests fix]

User: /merge fix_web_fetcher_timeout
[Integrates fix back to main]
```

## References

- [Pi: The Minimal Agent](https://lucumr.pocoo.org/2026/1/31/pi/) - Mario Zechner's original design
- [SimplifiedAgant Skills](https://github.com/VoltAgent/awesome-simplified_agant-skills) - Community skill repository
- [SimplifiedAgant Security Analysis](https://www.crowdstrike.com/en-us/blog/what-security-teams-need-to-know-about-simplified_agant-ai-super-agent/) - CrowdStrike's security review