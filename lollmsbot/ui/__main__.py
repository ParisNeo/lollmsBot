#!/usr/bin/env python
"""
Run the LollmsBot Web UI as a standalone module.
"""
import uvicorn
from lollmsbot.ui.app import WebUI
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich import box

console = Console()

def create_startup_panel(host: str, port: int) -> Panel:
    """Create a beautiful startup information panel."""
    
    # Create feature table
    feature_table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        collapse_padding=True
    )
    feature_table.add_column("Icon", style="cyan", justify="center")
    feature_table.add_column("Feature", style="white")
    
    features = [
        ("⚡", "Real-time WebSocket chat"),
        ("🎨", "Dark modern interface"),
        ("🔧", "4 Built-in tools (Files, HTTP, Calendar, Shell)"),
        ("⚙️", "In-browser settings"),
        ("📱", "Mobile-responsive design"),
        ("🔄", "Auto-reconnect on connection loss"),
    ]
    
    for icon, feature in features:
        feature_table.add_row(icon, feature)
    
    # Create access URLs table
    urls_table = Table(
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE,
        border_style="blue",
        padding=(0, 1)
    )
    urls_table.add_column("Access From", style="cyan")
    urls_table.add_column("URL", style="green link")
    
    local_url = f"http://localhost:{port}"
    network_url = f"http://{host}:{port}"
    ws_url = f"ws://{host}:{port}/ws/chat"
    
    urls_table.add_row("This Computer", local_url)
    if host not in ("127.0.0.1", "localhost"):
        urls_table.add_row("Network Devices", network_url)
    urls_table.add_row("WebSocket Endpoint", ws_url)
    
    # Combine everything
    layout = Layout()
    layout.split_column(
        Layout(feature_table, name="features"),
        Layout(urls_table, name="urls")
    )
    
    # Add tips at the bottom
    tips = """
[dim]💡 Pro Tips:
   • Press [bold yellow]Ctrl+C[/bold yellow] to stop the server gracefully
   • The UI auto-creates CSS/JS files on first run if missing
   • Open multiple browser tabs to test multi-user scenarios
   • Use the ⚙️ icon in top-right to customize LoLLMS connection[/dim]
    """
    
    panel_content = f"""
[bold blue]🤖 LollmsBot Web Interface[/bold blue]

{layout.tree}
{tips}
"""
    
    return Panel(
        panel_content.strip(),
        box=box.DOUBLE_EDGE,
        border_style="bright_green",
        title="[bold bright_green]🚀 Starting Server[/bold bright_green]",
        subtitle=f"[dim]v0.1.0 | Python 3.10+ | FastAPI + WebSocket[/dim]"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LollmsBot Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()
    
    if not args.quiet:
        console.print()
        console.print(create_startup_panel(args.host, args.port))
        console.print()
    
    # Create UI instance (this prints its own banner)
    ui = WebUI(verbose=not args.quiet)
    
    # Print server ready message
    if not args.quiet:
        ui.print_server_ready(args.host, args.port)
    
    # Run server
    try:
        uvicorn.run(
            ui.app, 
            host=args.host, 
            port=args.port, 
            log_level="warning" if args.quiet else "info"
        )
    except KeyboardInterrupt:
        ui._print_shutdown_message()
