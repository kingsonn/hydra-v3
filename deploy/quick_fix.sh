#!/bin/bash
# Quick fix for Python version and manual setup

set -e

echo "=========================================="
echo "Hydra Trading Bot - Quick Setup Fix"
echo "=========================================="

cd /home/ubuntu/hydra-v3

# Create virtual environment with system Python (3.12)
echo "[1/4] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "[2/4] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory
echo "[3/4] Creating logs directory..."
mkdir -p logs

# Create .env file if not exists
if [ ! -f ".env" ]; then
    echo "[4/4] Creating .env template..."
    cat > .env << 'EOF'
# Binance API (optional for live trading)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Dashboard port
DASHBOARD_PORT=8888
EOF
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "The systemd service is already installed."
echo "Now start the bot:"
echo "  sudo systemctl start hydra"
echo "  sudo systemctl status hydra"
echo ""
