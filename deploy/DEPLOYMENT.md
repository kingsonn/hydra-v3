# Hydra Trading Bot - AWS EC2 Deployment Guide

## Prerequisites
- AWS EC2 instance (Ubuntu 22.04 LTS recommended)
- Minimum: t3.medium (2 vCPU, 4GB RAM)
- Security group with port 8888 open for dashboard access
- SSH access to the instance

## Quick Start

### 1. SSH into your EC2 instance
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### 2. Clone the repository
```bash
cd /home/ubuntu
git clone https://github.com/YOUR_USERNAME/hydra-v3.git
cd hydra-v3
```

### 3. Run the setup script
```bash
chmod +x deploy/setup.sh
./deploy/setup.sh
```

### 4. Configure environment
```bash
nano .env
# Add your API keys if needed
```

### 5. Start the bot
```bash
sudo systemctl start hydra
```

### 6. Access the dashboard
Open in browser: `http://YOUR_EC2_PUBLIC_IP:8888`

---

## AWS Security Group Configuration

Add the following inbound rules:
| Type | Port | Source | Description |
|------|------|--------|-------------|
| SSH | 22 | Your IP | SSH access |
| Custom TCP | 8888 | 0.0.0.0/0 | Dashboard (or restrict to your IP) |

---

## Useful Commands

### Service Management
```bash
# Start the bot
sudo systemctl start hydra

# Stop the bot
sudo systemctl stop hydra

# Restart the bot
sudo systemctl restart hydra

# Check status
sudo systemctl status hydra

# Enable auto-start on boot
sudo systemctl enable hydra
```

### Logs
```bash
# View live logs
tail -f /home/ubuntu/hydra-v3/logs/hydra.log

# View error logs
tail -f /home/ubuntu/hydra-v3/logs/hydra_error.log

# View last 100 lines
tail -100 /home/ubuntu/hydra-v3/logs/hydra.log
```

### Updates
```bash
# Pull latest code
cd /home/ubuntu/hydra-v3
git pull origin main

# Restart service
sudo systemctl restart hydra
```

---

## Monitoring & Health Checks

### Check if bot is running
```bash
sudo systemctl is-active hydra
```

### Check memory usage
```bash
ps aux | grep python
free -h
```

### Check disk usage
```bash
df -h
du -sh /home/ubuntu/hydra-v3/logs/
```

### Log rotation (prevent disk full)
Create `/etc/logrotate.d/hydra`:
```
/home/ubuntu/hydra-v3/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ubuntu ubuntu
}
```

---

## Troubleshooting

### Bot not starting
```bash
# Check service status
sudo systemctl status hydra

# Check journal logs
sudo journalctl -u hydra -n 50

# Test manually
cd /home/ubuntu/hydra-v3
source venv/bin/activate
python -m src.dashboard.global_runner
```

### Dashboard not accessible
1. Check security group has port 8888 open
2. Check bot is running: `sudo systemctl status hydra`
3. Check firewall: `sudo ufw status`

### High memory usage
The bot is configured with 4GB memory limit. If exceeded, it will restart automatically.

### WebSocket disconnections
The bot handles reconnections automatically. Check logs for any persistent issues.

---

## Cost Optimization

For 3-week continuous running:
- **t3.medium**: ~$30/month (recommended)
- **t3.small**: ~$15/month (minimum, may have memory issues)

Consider using a Reserved Instance or Savings Plan for longer deployments.
