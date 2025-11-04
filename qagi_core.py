#!/usr/bin/env python3
"""
QAGI Core System - 24/7 Autonomous AGI
Integrates all components for continuous operation
"""

import asyncio
import logging
from datetime import datetime
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAGICore:
    """
    Main QAGI system that coordinates all components for 24/7 operation
    """

    def __init__(self):
        self.running = False
        self.start_time = None
        self.tasks_completed = 0
        self.ollama_url = "http://96.31.83.171:11434"
        self.quantum_url = "http://96.31.83.171:8900"

    async def run_task(self, task_type: str, **kwargs):
        """Execute a task using available resources"""
        try:
            if task_type == "code_analysis":
                return await self.analyze_code(**kwargs)
            elif task_type == "quantum_optimization":
                return await self.quantum_optimize(**kwargs)
            elif task_type == "llm_generation":
                return await self.generate_with_llm(**kwargs)
            elif task_type == "system_monitor":
                return await self.monitor_system()
            else:
                logger.warning(f"Unknown task type: {task_type}")
                return None
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            return None

    async def analyze_code(self, **kwargs):
        """Analyze code using local LLM"""
        logger.info("Analyzing code...")
        # Use local Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5-coder:latest",
                    "prompt": "Analyze this system and suggest improvements",
                    "stream": False
                },
                timeout=30
            )
            return {"status": "completed", "analysis": response.json().get("response", "")[:200]}
        except Exception as e:
            logger.error(f"Code analysis error: {e}")
            return {"status": "failed", "error": str(e)}

    async def quantum_optimize(self, **kwargs):
        """Perform quantum optimization"""
        logger.info("Running quantum optimization...")
        try:
            response = requests.post(
                f"{self.quantum_url}/quantum/qaoa",
                json={'graph': [[0,1],[1,0]], 'p': 2},
                timeout=10
            )
            return {"status": "completed", "result": response.json()}
        except Exception as e:
            logger.error(f"Quantum optimization error: {e}")
            return {"status": "failed", "error": str(e)}

    async def generate_with_llm(self, prompt: str = "Generate a useful function", **kwargs):
        """Generate content with LLM"""
        logger.info(f"Generating with LLM: {prompt[:50]}...")
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5-coder:latest",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            return {"status": "completed", "generated": response.json().get("response", "")[:200]}
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {"status": "failed", "error": str(e)}

    async def monitor_system(self):
        """Monitor all system components"""
        logger.info("Monitoring system...")
        status = {
            "ollama": "unknown",
            "quantum": "unknown",
            "timestamp": datetime.now().isoformat()
        }

        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            status["ollama"] = "healthy" if resp.status_code == 200 else "unhealthy"
        except:
            status["ollama"] = "offline"

        try:
            resp = requests.get(f"{self.quantum_url}/health", timeout=2)
            status["quantum"] = "healthy" if resp.status_code == 200 else "unhealthy"
        except:
            status["quantum"] = "offline"

        return status

    async def autonomous_loop(self):
        """Main autonomous operation loop - runs 24/7"""
        logger.info("Starting autonomous loop...")
        self.running = True
        self.start_time = datetime.now()

        task_cycle = [
            ("system_monitor", {}),
            ("code_analysis", {}),
            ("llm_generation", {"prompt": "Generate a Python utility function"}),
            ("quantum_optimization", {}),
        ]

        cycle_count = 0

        while self.running:
            try:
                cycle_count += 1
                logger.info(f"\n{'='*50}\nCycle {cycle_count} - {datetime.now().isoformat()}\n{'='*50}")

                for task_type, kwargs in task_cycle:
                    result = await self.run_task(task_type, **kwargs)
                    if result:
                        self.tasks_completed += 1
                        logger.info(f"✅ Task {task_type}: {result.get('status', 'unknown')}")
                    
                    await asyncio.sleep(2)  # Brief pause between tasks

                # Pause between cycles
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """Stop the autonomous loop"""
        logger.info("Stopping QAGI...")
        self.running = False

    def get_stats(self):
        """Get current system statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        return {
            "running": self.running,
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "tasks_completed": self.tasks_completed,
            "tasks_per_hour": (self.tasks_completed / (uptime / 3600)) if uptime > 0 else 0
        }


async def main():
    """Run QAGI system"""
    print("\n" + "="*70)
    print("QAGI - Quantum GPU-Accelerated General Intelligence")
    print("Starting 24/7 Autonomous Operation...")
    print("="*70 + "\n")

    qagi = QAGICore()

    # Start autonomous loop
    try:
        loop_task = asyncio.create_task(qagi.autonomous_loop())
        
        # Run for demo duration (60 seconds)
        await asyncio.sleep(60)
        
        # Stop and show stats
        qagi.stop()
        await loop_task
        
        stats = qagi.get_stats()
        print("\n" + "="*70)
        print("QAGI STATISTICS")
        print("="*70)
        print(f"Uptime: {stats['uptime_hours']:.2f} hours")
        print(f"Tasks Completed: {stats['tasks_completed']}")
        print(f"Tasks/Hour: {stats['tasks_per_hour']:.1f}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nStopping QAGI...")
        qagi.stop()


if __name__ == "__main__":
    asyncio.run(main())
