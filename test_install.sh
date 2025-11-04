#!/bin/bash

echo "======================================================================"
echo "QAGI Installation Pre-Flight Check"
echo "======================================================================"
echo

# Check Python
echo "1. Checking Python..."
which python3 && python3 --version || echo "❌ Python3 not found"
echo

# Check dependencies
echo "2. Checking Python dependencies..."
python3 -c "import requests, asyncio; print('✅ Required modules available')" 2>/dev/null || echo "⚠️  Some modules may be missing (will install if needed)"
echo

# Check paths
echo "3. Checking paths..."
QAGI_DIR="/home/claudehome/claude-workspace/projects/qagi-system"
if [ -d "$QAGI_DIR" ]; then
    echo "✅ QAGI directory exists: $QAGI_DIR"
else
    echo "❌ QAGI directory not found: $QAGI_DIR"
    exit 1
fi
echo

# Check files
echo "4. Checking required files..."
for file in qagi_core.py qagi_dashboard.py qagi.service qagi-dashboard.service; do
    if [ -f "$QAGI_DIR/$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file missing"
    fi
done
echo

# Check service files
echo "5. Validating service files..."
if grep -q '\$USER\|$(pwd)' "$QAGI_DIR/qagi.service"; then
    echo "❌ qagi.service contains shell variables"
    exit 1
else
    echo "✅ qagi.service OK"
fi

if grep -q '\$USER\|$(pwd)' "$QAGI_DIR/qagi-dashboard.service"; then
    echo "❌ qagi-dashboard.service contains shell variables"
    exit 1
else
    echo "✅ qagi-dashboard.service OK"
fi
echo

# Test Ollama connectivity
echo "6. Testing Ollama connectivity..."
if curl -s http://96.31.83.171:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama accessible"
else
    echo "⚠️  Ollama not accessible (may work from systemd)"
fi
echo

echo "======================================================================"
echo "Pre-flight check complete!"
echo "======================================================================"
echo
echo "Run: cd ~/claude-workspace/projects/qagi-system && ./install_services.sh"
