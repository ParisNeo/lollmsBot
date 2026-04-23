# LollmsBot Universal Windows Installer
Write-Host "🧊 LollmsBot Sovereign Engine Installer" -ForegroundColor Cyan
Write-Host "----------------------------------------"

# Check Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Python not found. Please install Python 3.10+ from python.org" -ForegroundColor Red
    exit
}

# Create Virtual Environment
Write-Host "📦 Creating Virtual Environment..."
python -m venv .venv
& .venv\Scripts\Activate.ps1

# Upgrade Pip
python -m pip install --upgrade pip

# Install with all dependencies
Write-Host "🚀 Installing LollmsBot and all Channel Adapters..."
pip install -e .[all]

# Run Wizard
Write-Host "🧬 Launching Configuration Wizard..."
lollmsbot wizard