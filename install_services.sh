#!/bin/bash

echo "======================================================================"
echo "Installing QAGI Services"
echo "======================================================================"
echo

# Stop existing services if running
echo "Stopping existing services (if any)..."
sudo systemctl stop qagi.service 2>/dev/null || true
sudo systemctl stop qagi-dashboard.service 2>/dev/null || true
sudo systemctl disable qagi.service 2>/dev/null || true
sudo systemctl disable qagi-dashboard.service 2>/dev/null || true
echo

# Create log directory
echo "Creating log directory..."
sudo mkdir -p /var/log/qagi
sudo chown claudehome:claudehome /var/log/qagi
echo

# Copy service files
echo "Installing service files..."
sudo cp qagi.service /etc/systemd/system/
sudo cp qagi-dashboard.service /etc/systemd/system/
echo

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload
echo

# Enable services
echo "Enabling services..."
sudo systemctl enable qagi.service
sudo systemctl enable qagi-dashboard.service
echo

# Start services
echo "Starting services..."
sudo systemctl start qagi.service
sleep 2
sudo systemctl start qagi-dashboard.service
sleep 2
echo

# Check status
echo "======================================================================"
echo "Service Status:"
echo "======================================================================"
echo
echo "QAGI Core:"
sudo systemctl status qagi.service --no-pager -l | head -15
echo
echo "QAGI Dashboard:"
sudo systemctl status qagi-dashboard.service --no-pager -l | head -15
echo

echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo
echo "Monitor services:"
echo "  sudo systemctl status qagi"
echo "  sudo systemctl status qagi-dashboard"
echo
echo "View logs:"
echo "  sudo journalctl -u qagi -f"
echo "  tail -f /var/log/qagi/qagi.log"
echo
echo "Dashboard: http://localhost:9200"
echo
