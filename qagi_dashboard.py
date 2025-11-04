#!/usr/bin/env python3
"""
QAGI Advanced Dashboard with Avatar
Real-time monitoring and control interface
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
from datetime import datetime
import uvicorn

app = FastAPI(title="QAGI Dashboard")

# Store for real-time data
dashboard_data = {
    "system_status": "initializing",
    "gpu_stats": {},
    "tasks": [],
    "learning_progress": {"skills": 0, "knowledge": 0},
    "avatar_state": "idle"
}

@app.get("/")
async def get_dashboard():
    """Serve dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>QAGI Control Center</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #fff;
                padding: 20px;
            }
            .container { max-width: 1600px; margin: 0 auto; }
            .header {
                text-align: center;
                padding: 30px 0;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                margin-bottom: 30px;
            }
            .header h1 {
                font-size: 3em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .card {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                border: 1px solid rgba(255,255,255,0.18);
            }
            .card h2 {
                margin-bottom: 20px;
                font-size: 1.5em;
                border-bottom: 2px solid rgba(255,255,255,0.3);
                padding-bottom: 10px;
            }
            .stat {
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }
            .stat:last-child { border-bottom: none; }
            .stat-value {
                font-weight: bold;
                font-size: 1.2em;
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-healthy { background: #4ade80; box-shadow: 0 0 10px #4ade80; }
            .status-warning { background: #fbbf24; box-shadow: 0 0 10px #fbbf24; }
            .status-offline { background: #ef4444; box-shadow: 0 0 10px #ef4444; }
            .avatar-container {
                width: 300px;
                height: 300px;
                margin: 0 auto;
                background: linear-gradient(45deg, #ff00ff, #00ffff);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 6em;
                animation: pulse 2s ease-in-out infinite;
            }
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 0.8; }
                50% { transform: scale(1.05); opacity: 1; }
            }
            .task-list {
                max-height: 300px;
                overflow-y: auto;
            }
            .task-item {
                background: rgba(255,255,255,0.05);
                padding: 10px;
                margin: 5px 0;
                border-radius: 8px;
                border-left: 3px solid #4ade80;
            }
            .gpu-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 10px;
            }
            .gpu-card {
                background: rgba(255,255,255,0.05);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            .gpu-util {
                font-size: 2em;
                font-weight: bold;
                color: #4ade80;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 QAGI Control Center</h1>
                <p>Quantum GPU-Accelerated General Intelligence</p>
                <p><small>24/7 Autonomous Operation</small></p>
            </div>

            <div class="grid">
                <!-- Avatar Card -->
                <div class="card">
                    <h2>QAGI Avatar</h2>
                    <div class="avatar-container" id="avatar">
                        🤖
                    </div>
                    <div style="text-align: center; margin-top: 15px;">
                        <p id="avatar-status">Status: <strong>Active</strong></p>
                        <p id="avatar-mood">Mood: <strong>Productive</strong></p>
                    </div>
                </div>

                <!-- System Status Card -->
                <div class="card">
                    <h2>System Status</h2>
                    <div class="stat">
                        <span>Ollama LLM</span>
                        <span><span class="status-indicator status-healthy"></span><span id="ollama-status">Checking...</span></span>
                    </div>
                    <div class="stat">
                        <span>Quantum API</span>
                        <span><span class="status-indicator status-healthy"></span><span id="quantum-status">Checking...</span></span>
                    </div>
                    <div class="stat">
                        <span>Uptime</span>
                        <span class="stat-value" id="uptime">0:00:00</span>
                    </div>
                    <div class="stat">
                        <span>Tasks Completed</span>
                        <span class="stat-value" id="tasks-completed">0</span>
                    </div>
                </div>

                <!-- GPU Fleet Status -->
                <div class="card">
                    <h2>GPU Fleet (7x RTX 2080 Ti)</h2>
                    <div class="gpu-grid" id="gpu-grid">
                        <!-- GPUs will be populated here -->
                    </div>
                </div>
            </div>

            <!-- Active Tasks -->
            <div class="card">
                <h2>Active Tasks</h2>
                <div class="task-list" id="task-list">
                    <div class="task-item">Initializing task queue...</div>
                </div>
            </div>

            <!-- Learning Progress -->
            <div class="card" style="margin-top: 20px;">
                <h2>Learning & Growth</h2>
                <div class="stat">
                    <span>Skills Acquired</span>
                    <span class="stat-value" id="skills">0</span>
                </div>
                <div class="stat">
                    <span>Knowledge Base Size</span>
                    <span class="stat-value" id="knowledge">0 MB</span>
                </div>
                <div class="stat">
                    <span>Learning Rate</span>
                    <span class="stat-value" id="learning-rate">Active</span>
                </div>
            </div>
        </div>

        <script>
            // WebSocket connection for real-time updates
            let ws = null;
            let startTime = Date.now();

            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:9200/ws');
                
                ws.onopen = () => {
                    console.log('Connected to QAGI');
                    updateStatus('Connected', 'healthy');
                };

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                ws.onclose = () => {
                    console.log('Disconnected from QAGI');
                    updateStatus('Disconnected', 'offline');
                    setTimeout(connectWebSocket, 5000);
                };
            }

            function updateDashboard(data) {
                // Update system status
                if (data.ollama_status) {
                    document.getElementById('ollama-status').textContent = data.ollama_status;
                }
                if (data.quantum_status) {
                    document.getElementById('quantum-status').textContent = data.quantum_status;
                }
                if (data.tasks_completed !== undefined) {
                    document.getElementById('tasks-completed').textContent = data.tasks_completed;
                }

                // Update avatar mood based on system state
                if (data.tasks_completed > 10) {
                    document.getElementById('avatar-mood').innerHTML = 'Mood: <strong>Highly Productive</strong>';
                }
            }

            function updateUptime() {
                const elapsed = Date.now() - startTime;
                const hours = Math.floor(elapsed / 3600000);
                const minutes = Math.floor((elapsed % 3600000) / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                document.getElementById('uptime').textContent = 
                    `${hours}:${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(2,'0')}`;
            }

            function simulateGPUs() {
                const gpuGrid = document.getElementById('gpu-grid');
                for (let i = 0; i < 7; i++) {
                    const util = Math.floor(Math.random() * 40) + 20;
                    const gpu = document.createElement('div');
                    gpu.className = 'gpu-card';
                    gpu.innerHTML = `
                        <div>GPU ${i}</div>
                        <div class="gpu-util">${util}%</div>
                        <div style="font-size: 0.8em; margin-top: 5px;">Active</div>
                    `;
                    gpuGrid.appendChild(gpu);
                }
            }

            // Initialize
            connectWebSocket();
            simulateGPUs();
            setInterval(updateUptime, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send updates every 2 seconds
            await websocket.send_json(dashboard_data)
            await asyncio.sleep(2)
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return dashboard_data

if __name__ == "__main__":
    print("🚀 Starting QAGI Dashboard on http://localhost:9200")
    uvicorn.run(app, host="0.0.0.0", port=9200)
