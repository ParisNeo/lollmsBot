#!/usr/bin/env python
"""
lollmsBot CLI - Gateway + Wizard + UI
"""
from __future__ import annotations

import argparse
import sys
from typing import List

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    console = Console()
except ImportError:
    print("Install dev deps: pip install -e .[dev]")
    sys.exit(1)


def print_ui_banner() -> None:
    """Print beautiful UI launch banner."""
    console.print()
    
    # Create ASCII art style banner
    banner = Text()
    banner.append("╭─────────╮\n", style="blue")
    banner.append("│  🤖     │  ", style="blue")
    banner.append("LollmsBot", style="bold cyan")
    banner.append(" Web UI\n", style="bold blue")
    banner.append("╰─────────╯\n", style="blue")
    
    panel = Panel(
        banner,
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        title="[bold]Starting Interface[/bold]",
        subtitle="[dim]Real-time AI Chat[/dim]"
    )
    console.print(panel)


def print_gateway_banner(host: str, port: int, ui_enabled: bool, debug_mode: bool = False) -> None:
    """Print gateway startup banner with status."""
    
    # For display purposes, use localhost if host is 0.0.0.0 or empty
    # Browsers can't connect to 0.0.0.0, they need localhost/127.0.0.1
    display_host = "localhost" if host in ("0.0.0.0", "") else host
    
    # Status indicators
    status_table = Table(
        show_header=False,
        box=box.SIMPLE,
        border_style="blue",
        padding=(0, 2)
    )
    status_table.add_column("Service", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("URL", style="dim")
    
    status_table.add_row(
        "🔌 Gateway API",
        "✅ Active",
        f"http://{display_host}:{port}"
    )
    status_table.add_row(
        "📚 API Docs",
        "✅ Available",
        f"http://{display_host}:{port}/docs"
    )
    
    if ui_enabled:
        status_table.add_row(
            "🌐 Web UI",
            "✅ Mounted",
            f"http://{display_host}:{port}/ui"
        )
    else:
        status_table.add_row(
            "🌐 Web UI",
            "⭕ Disabled",
            "Use --ui to enable"
        )
    
    if debug_mode:
        status_table.add_row(
            "🐛 Debug Mode",
            "✅ Enabled",
            "Rich memory display active"
        )
    
    panel = Panel(
        status_table,
        box=box.ROUNDED,
        border_style="bright_green" if ui_enabled else "yellow",
        title="[bold bright_green]🚀 Gateway Starting[/bold bright_green]",
        subtitle=f"[dim]LoLLMS Agentic Bot | Host: {host}{' | DEBUG MODE' if debug_mode else ''}[/dim]"
    )
    console.print()
    console.print(panel)
    console.print()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lollmsbot",
        description="Agentic LoLLMS Assistant (Clawdbot-style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
┌─────────────────────────────────────────────────────────────┐
│  Examples:                                                  │
│    lollmsbot wizard          # Interactive setup            │
│    lollmsbot gateway         # Run API server               │
│    lollmsbot gateway --ui    # API + Web UI together        │
│    lollmsbot gateway --debug # Run with debug output        │
│    lollmsbot ui              # Web UI only (standalone)     │
│    lollmsbot ui --port 3000  # UI on custom port            │
└─────────────────────────────────────────────────────────────┘
        """
    )
    parser.add_argument("--version", action="version", version="lollmsBot 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Gateway command
    gateway_parser = subparsers.add_parser(
        "gateway", 
        help="Run API gateway server",
        description="Start the main API gateway with optional channels and UI"
    )
    gateway_parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    gateway_parser.add_argument("--port", type=int, default=8800, help="Port number (default: 8800)")
    gateway_parser.add_argument("--ui", action="store_true", help="Also start web UI at /ui")
    gateway_parser.add_argument("--debug", action="store_true", help="Enable debug mode with rich memory display")

    # UI command (standalone)
    ui_parser = subparsers.add_parser(
        "ui", 
        help="Run web UI only (standalone mode)",
        description="Start just the web interface without the full gateway"
    )
    ui_parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    ui_parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    ui_parser.add_argument("--quiet", "-q", action="store_true", help="Minimal console output")

    # Wizard command
    wizard_parser = subparsers.add_parser(
        "wizard", 
        help="Interactive setup wizard",
        description="Configure LoLLMS connection and bot settings interactively"
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "gateway":
            import uvicorn
            from lollmsbot.config import GatewaySettings
            from lollmsbot import gateway
            
            settings = GatewaySettings.from_env()
            host = args.host or settings.host
            port = args.port or settings.port
            
            # Set debug mode in gateway module
            gateway.DEBUG_MODE = getattr(args, 'debug', False)
            
            # Print startup banner
            print_gateway_banner(host, port, args.ui, gateway.DEBUG_MODE)
            
            # Enable UI if requested
            if args.ui:
                # Use localhost for UI server internally, gateway will mount it
                gateway.enable_ui(host="127.0.0.1", port=8080)
            
            # Run server
            uvicorn.run(
                "lollmsbot.gateway:app",
                host=host,
                port=port,
                reload=False,
                log_level="debug" if gateway.DEBUG_MODE else "info"
            )
            
        elif args.command == "ui":
            # Run standalone UI with full rich output
            from lollmsbot.ui.app import WebUI
            import uvicorn
            
            print_ui_banner()
            
            ui = WebUI(verbose=not args.quiet)
            ui.print_server_ready(args.host, args.port)
            
            try:
                uvicorn.run(
                    ui.app,
                    host=args.host,
                    port=args.port,
                    log_level="warning" if args.quiet else "info"
                )
            except KeyboardInterrupt:
                ui._print_shutdown_message()
            
        elif args.command == "wizard":
            from lollmsbot import wizard
            wizard.run_wizard()
            
        else:
            parser.print_help()
            console.print("\n[bold cyan]💡 Need help? Try: lollmsbot wizard[/]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/]")
        sys.exit(130)
    except ImportError as e:
        console.print(f"[red]❌ Missing dependency: {e}[/]")
        console.print("[cyan]💡 Run: pip install -e .[dev][/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]💥 Error: {e}[/]")
        console.print_exception(show_locals=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
