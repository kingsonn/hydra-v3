#!/bin/bash
# Hydra Trading Bot - AWS EC2 Setup Script
# Run as: chmod +x setup.sh && ./setup.sh

set -e

echo "=========================================="
echo "Hydra Trading Bot - EC2 Setup"
echo "=========================================="

# Update system
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ and dependencies
echo "[2/8] Installing Python and dependencies..."
sudo apt install -y python3.11 python3.11-venv python3-pip git

# Create project directory
echo "[3/8] Setting up project directory..."
cd /home/ubuntu
if [ ! -d "hydra-v3" ]; then
    git clone https://github.com/YOUR_USERNAME/hydra-v3.git
fi
cd hydra-v3

# Create virtual environment
echo "[4/8] Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install requirements
echo "[5/8] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory
echo "[6/8] Creating logs directory..."
mkdir -p logs

# Setup systemd service
echo "[7/8] Setting up systemd service..."
sudo cp deploy/hydra.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hydra

# Create .env file if not exists
if [ ! -f ".env" ]; then
    echo "[!] Creating .env template - EDIT THIS FILE!"
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
echo "Next steps:"
echo "  1. Edit .env file: nano /home/ubuntu/hydra-v3/.env"
echo "  2. Start the bot: sudo systemctl start hydra"
echo "  3. Check status: sudo systemctl status hydra"
echo "  4. View logs: tail -f /home/ubuntu/hydra-v3/logs/hydra.log"
echo "  5. Access dashboard: http://YOUR_PUBLIC_IP:8888"
echo ""
echo "Useful commands:"
echo "  - Stop bot: sudo systemctl stop hydra"
echo "  - Restart bot: sudo systemctl restart hydra"
echo "  - View errors: tail -f /home/ubuntu/hydra-v3/logs/hydra_error.log"
echo ""
