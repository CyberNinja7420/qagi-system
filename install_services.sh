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
