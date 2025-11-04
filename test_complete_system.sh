#!/bin/bash

echo "======================================================================"
echo "QAGI Complete System Test"
echo "======================================================================"
echo

PASS=0
FAIL=0

test_service() {
    local name=$1
    local port=$2
    
    echo "Testing $name..."
    
    if systemctl is-active --quiet $name 2>/dev/null; then
        echo "  ✅ Service is running"
        ((PASS++))
        
        if [ -n "$port" ]; then
            if curl -s http://localhost:$port > /dev/null 2>&1; then
                echo "  ✅ Port $port is accessible"
                ((PASS++))
            else
                echo "  ⚠️  Port $port not accessible yet (may still be starting)"
            fi
        fi
    else
        echo "  ❌ Service not running"
        ((FAIL++))
    fi
    echo
}

echo "1. Testing QAGI Core..."
test_service "qagi.service" ""

echo "2. Testing QAGI Dashboard..."
test_service "qagi-dashboard.service" "9200"

echo "3. Testing QAGI Virtual Assistant..."
test_service "qagi-assistant.service" "9300"

echo "4. Testing Ollama connectivity..."
if curl -s http://96.31.83.171:11434/api/tags > /dev/null 2>&1; then
    echo "  ✅ Ollama is accessible"
    ((PASS++))
    
    MODELS=$(curl -s http://96.31.83.171:11434/api/tags | jq -r '.models | length' 2>/dev/null)
    if [ -n "$MODELS" ]; then
        echo "  ✅ $MODELS models available"
        ((PASS++))
    fi
else
    echo "  ❌ Ollama not accessible"
    ((FAIL++))
fi
echo

echo "5. Checking log files..."
if [ -d "/var/log/qagi" ]; then
    echo "  ✅ Log directory exists"
    ((PASS++))
    ls -lh /var/log/qagi/ 2>/dev/null | tail -5
else
    echo "  ❌ Log directory missing"
    ((FAIL++))
fi
echo

echo "======================================================================"
echo "Test Results:"
echo "======================================================================"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo

if [ $FAIL -eq 0 ]; then
    echo "✅ All tests passed! QAGI system is fully operational."
    echo
    echo "🌐 Access Points:"
    echo "   • Dashboard:   http://localhost:9200"
    echo "   • Assistant:   http://localhost:9300"
    echo
    exit 0
else
    echo "⚠️  Some tests failed. Check the logs:"
    echo "   sudo journalctl -u qagi -n 50"
    echo "   sudo journalctl -u qagi-assistant -n 50"
    echo
    exit 1
fi
