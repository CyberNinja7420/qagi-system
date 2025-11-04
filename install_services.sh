#!/bin/bash

echo "======================================================================"
echo "Installing QAGI Complete System"
echo "======================================================================"
echo

# Check dependencies
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn, websockets" 2>/dev/null || {
    echo "Installing required Python packages..."
    pip3 install fastapi uvicorn websockets --quiet
}
echo

# Stop existing services if running
echo "Stopping existing services (if any)..."
sudo systemctl stop qagi.service 2>/dev/null || true
sudo systemctl stop qagi-dashboard.service 2>/dev/null || true
sudo systemctl stop qagi-assistant.service 2>/dev/null || true
sudo systemctl disable qagi.service 2>/dev/null || true
sudo systemctl disable qagi-dashboard.service 2>/dev/null || true
sudo systemctl disable qagi-assistant.service 2>/dev/null || true
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
sudo cp qagi-assistant.service /etc/systemd/system/
echo

# Reload systemd
echo "Reloading systemd..."
sudo systemctl daemon-reload
echo

# Enable services
echo "Enabling services..."
sudo systemctl enable qagi.service
sudo systemctl enable qagi-dashboard.service
sudo systemctl enable qagi-assistant.service
echo

# Start services
echo "Starting services..."
echo "  1. QAGI Core (autonomous system)..."
sudo systemctl start qagi.service
sleep 3

echo "  2. QAGI Dashboard (port 9200)..."
sudo systemctl start qagi-dashboard.service
sleep 2

echo "  3. QAGI Virtual Assistant (port 9300)..."
sudo systemctl start qagi-assistant.service
sleep 2
echo

# Check status
echo "======================================================================"
echo "Service Status:"
echo "======================================================================"
echo

echo "▶ QAGI Core (24/7 Autonomous System):"
sudo systemctl is-active qagi.service && echo "  ✅ Running" || echo "  ❌ Not running"
echo

echo "▶ QAGI Dashboard:"
sudo systemctl is-active qagi-dashboard.service && echo "  ✅ Running" || echo "  ❌ Not running"
echo

echo "▶ QAGI Virtual Assistant:"
sudo systemctl is-active qagi-assistant.service && echo "  ✅ Running" || echo "  ❌ Not running"
echo

echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo
echo "📊 Dashboards:"
echo "   • Main Dashboard:     http://localhost:9200"
echo "   • Virtual Assistant:  http://localhost:9300"
echo
echo "🔧 Management Commands:"
echo "   • Status:  sudo systemctl status qagi"
echo "   • Logs:    sudo journalctl -u qagi -f"
echo "   • Stop:    sudo systemctl stop qagi qagi-dashboard qagi-assistant"
echo "   • Start:   sudo systemctl start qagi qagi-dashboard qagi-assistant"
echo
echo "🤖 QAGI is now running 24/7!"
echo
