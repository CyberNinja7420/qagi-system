#!/bin/bash
# QAGI System Deployment Script
# Deploys QAGI for 24/7 operation

set -e

echo "🚀 Deploying QAGI System"
echo "========================"

# Create systemd service for 24/7 operation
cat > qagi.service << 'EOF'
[Unit]
Description=QAGI - Quantum GPU-Accelerated General Intelligence
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/qagi_core.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/qagi/qagi.log
StandardError=append:/var/log/qagi/qagi.error.log

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Created systemd service file"

# Create dashboard service
cat > qagi-dashboard.service << 'EOF'
[Unit]
Description=QAGI Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $(pwd)/qagi_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Created dashboard service file"

# Create installation script
cat > install_services.sh << 'EOFINSTALL'
#!/bin/bash
# Run this with sudo to install QAGI as system services

sudo mkdir -p /var/log/qagi
sudo cp qagi.service /etc/systemd/system/
sudo cp qagi-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qagi
sudo systemctl enable qagi-dashboard
sudo systemctl start qagi
sudo systemctl start qagi-dashboard

echo "✅ QAGI services installed and started"
echo "Check status with: sudo systemctl status qagi"
echo "Dashboard: http://localhost:9200"
EOFINSTALL

chmod +x install_services.sh

echo ""
echo "✅ Deployment files created!"
echo ""
echo "To deploy for 24/7 operation:"
echo "  sudo ./install_services.sh"
echo ""
echo "To run manually:"
echo "  python3 qagi_core.py"
echo "  python3 qagi_dashboard.py  # In another terminal"
echo ""
