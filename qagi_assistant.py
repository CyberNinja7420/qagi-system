#!/usr/bin/env python3
"""
QAGI Virtual Assistant - Interactive AI Assistant Interface
Features chat, voice recognition, and visual avatar
"""

import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests
import json
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class QAGIAssistant:
    """Interactive QAGI Virtual Assistant"""
    
    def __init__(self):
        self.ollama_url = "http://96.31.83.171:11434"
        self.conversation_history: List[Dict] = []
        self.active_connections: List[WebSocket] = []
        
    async def process_message(self, message: str) -> str:
        """Process user message and generate response"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Generate response using Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5-coder:latest",
                    "prompt": f"You are QAGI, an autonomous AI assistant. User: {message}\nQAGI:",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_response = result.get("response", "I'm processing that...")
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                return assistant_response
            else:
                return "I'm having trouble connecting to my neural networks. Please try again."
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Error: {str(e)}"
    
    async def get_status(self) -> Dict:
        """Get QAGI system status"""
        try:
            # Check Ollama
            ollama_status = "offline"
            models = []
            try:
                resp = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                if resp.status_code == 200:
                    ollama_status = "online"
                    models = [m["name"] for m in resp.json().get("models", [])]
            except:
                pass
            
            return {
                "status": "online" if ollama_status == "online" else "limited",
                "ollama": ollama_status,
                "models_available": len(models),
                "models": models[:5],  # First 5 models
                "conversation_length": len(self.conversation_history),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"status": "error", "error": str(e)}

assistant = QAGIAssistant()

@app.get("/")
async def get_assistant():
    """Serve QAGI Assistant interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>QAGI - Virtual AI Assistant</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                width: 100%;
                max-width: 1200px;
                height: 80vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .avatar {
                font-size: 60px;
                animation: float 3s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            .title {
                flex: 1;
                text-align: center;
            }
            
            .title h1 {
                font-size: 32px;
                margin-bottom: 5px;
            }
            
            .title p {
                font-size: 14px;
                opacity: 0.9;
            }
            
            .status {
                display: flex;
                align-items: center;
                gap: 10px;
                background: rgba(255,255,255,0.2);
                padding: 10px 15px;
                border-radius: 20px;
            }
            
            .status-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #10b981;
                animation: pulse 2s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .messages {
                flex: 1;
                padding: 25px;
                overflow-y: auto;
                background: #f7fafc;
            }
            
            .message {
                margin-bottom: 20px;
                display: flex;
                gap: 15px;
                animation: slideIn 0.3s ease-out;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .message.user {
                justify-content: flex-end;
            }
            
            .message-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                flex-shrink: 0;
            }
            
            .message.user .message-avatar {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                order: 2;
            }
            
            .message.assistant .message-avatar {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }
            
            .message-content {
                max-width: 70%;
                padding: 15px 20px;
                border-radius: 20px;
                line-height: 1.5;
            }
            
            .message.user .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-bottom-right-radius: 5px;
            }
            
            .message.assistant .message-content {
                background: white;
                color: #2d3748;
                border-bottom-left-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .input-container {
                padding: 25px;
                background: white;
                border-top: 1px solid #e2e8f0;
                display: flex;
                gap: 15px;
            }
            
            #userInput {
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #e2e8f0;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
                transition: all 0.3s;
            }
            
            #userInput:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            #sendBtn {
                padding: 15px 35px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            #sendBtn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            #sendBtn:active {
                transform: translateY(0);
            }
            
            .typing-indicator {
                display: none;
                padding: 15px 20px;
                background: white;
                border-radius: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .typing-indicator.active {
                display: block;
            }
            
            .typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #cbd5e0;
                margin: 0 2px;
                animation: typing 1.4s infinite;
            }
            
            .typing-indicator span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-indicator span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                }
                30% {
                    transform: translateY(-10px);
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="avatar">🤖</div>
                <div class="title">
                    <h1>QAGI Assistant</h1>
                    <p>Quantum GPU-Accelerated General Intelligence</p>
                </div>
                <div class="status">
                    <div class="status-dot"></div>
                    <span id="statusText">Online</span>
                </div>
            </div>
            
            <div class="chat-container">
                <div class="messages" id="messages">
                    <div class="message assistant">
                        <div class="message-avatar">🤖</div>
                        <div class="message-content">
                            Hello! I'm QAGI, your 24/7 autonomous AI assistant. I'm running on your local GPU infrastructure with 11 Ollama models. How can I help you today?
                        </div>
                    </div>
                </div>
                
                <div class="input-container">
                    <input 
                        type="text" 
                        id="userInput" 
                        placeholder="Ask me anything..." 
                        autocomplete="off"
                    >
                    <button id="sendBtn">Send</button>
                </div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            const messages = document.getElementById('messages');
            const userInput = document.getElementById('userInput');
            const sendBtn = document.getElementById('sendBtn');
            const statusText = document.getElementById('statusText');
            
            ws.onopen = () => {
                console.log('Connected to QAGI');
                statusText.textContent = 'Online';
            };
            
            ws.onclose = () => {
                console.log('Disconnected from QAGI');
                statusText.textContent = 'Offline';
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'response') {
                    addMessage('assistant', data.content);
                }
            };
            
            function addMessage(role, content) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}`;
                
                const avatar = document.createElement('div');
                avatar.className = 'message-avatar';
                avatar.textContent = role === 'user' ? '👤' : '🤖';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = content;
                
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(contentDiv);
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }
            
            function sendMessage() {
                const message = userInput.value.trim();
                if (!message) return;
                
                addMessage('user', message);
                userInput.value = '';
                
                // Show typing indicator
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message assistant';
                typingDiv.innerHTML = `
                    <div class="message-avatar">🤖</div>
                    <div class="typing-indicator active">
                        <span></span><span></span><span></span>
                    </div>
                `;
                messages.appendChild(typingDiv);
                messages.scrollTop = messages.scrollHeight;
                
                ws.send(JSON.stringify({
                    type: 'message',
                    content: message
                }));
                
                // Remove typing indicator after response
                setTimeout(() => {
                    if (typingDiv.parentNode) {
                        typingDiv.remove();
                    }
                }, 500);
            }
            
            sendBtn.addEventListener('click', sendMessage);
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Focus input on load
            userInput.focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    assistant.active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "message":
                user_message = message_data["content"]
                logger.info(f"User: {user_message}")
                
                # Process message
                response = await assistant.process_message(user_message)
                logger.info(f"QAGI: {response[:100]}...")
                
                # Send response
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "content": response
                }))
                
    except WebSocketDisconnect:
        assistant.active_connections.remove(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        assistant.active_connections.remove(websocket)

@app.get("/api/status")
async def get_status():
    """Get QAGI system status"""
    return await assistant.get_status()

@app.get("/api/conversation")
async def get_conversation():
    """Get conversation history"""
    return {
        "history": assistant.conversation_history[-50:],  # Last 50 messages
        "total": len(assistant.conversation_history)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("QAGI Virtual Assistant Starting...")
    print("="*70)
    print("\nAccess the assistant at: http://localhost:9300")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=9300, log_level="info")
