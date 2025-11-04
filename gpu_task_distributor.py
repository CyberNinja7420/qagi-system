#!/usr/bin/env python3
"""
GPU Task Distributor for QAGI System
Intelligently distributes tasks across all 7 NVIDIA RTX 2080 Ti GPUs
"""

import asyncio
import subprocess
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUCapability(Enum):
    """GPU capability types"""
    QUANTUM = "quantum_computing"
    LARGE_LLM = "large_language_model"
    MULTI_MODEL = "multi_model_inference"
    DYNAMIC = "dynamic_allocation"
    GENERAL = "general_compute"


@dataclass
class GPUStatus:
    """GPU status information"""
    gpu_id: int
    name: str
    utilization: float  # Percentage
    memory_used: int  # MB
    memory_total: int  # MB
    memory_free: int  # MB
    temperature: int  # Celsius
    power_draw: float  # Watts
    capability: GPUCapability
    current_tasks: int
    max_tasks: int

    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization percentage"""
        return (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0

    @property
    def is_available(self) -> bool:
        """Check if GPU has capacity for more tasks"""
        return (self.current_tasks < self.max_tasks and
                self.memory_utilization < 90 and
                self.temperature < 85)

    @property
    def load_score(self) -> float:
        """Calculate load score (lower is better)"""
        util_score = self.utilization / 100
        mem_score = self.memory_utilization / 100
        temp_score = self.temperature / 100
        task_score = self.current_tasks / self.max_tasks if self.max_tasks > 0 else 1

        return (util_score * 0.3 + mem_score * 0.4 +
                temp_score * 0.1 + task_score * 0.2)


@dataclass
class Task:
    """Task to be executed on GPU"""
    task_id: str
    task_type: str
    capability_required: GPUCapability
    estimated_vram: int  # MB
    priority: int  # 1-10, higher is more important
    timeout: int  # seconds
    payload: Dict


class GPUTaskDistributor:
    """
    Distributes tasks across 7 GPUs intelligently
    """

    # GPU assignments based on capabilities
    GPU_ASSIGNMENTS = {
        0: {'capability': GPUCapability.QUANTUM, 'models': ['pennylane_quantum'], 'max_tasks': 3},
        1: {'capability': GPUCapability.LARGE_LLM, 'models': ['codellama-70b'], 'max_tasks': 1},
        2: {'capability': GPUCapability.LARGE_LLM, 'models': ['codellama-70b'], 'max_tasks': 1},
        3: {'capability': GPUCapability.LARGE_LLM, 'models': ['codellama-70b'], 'max_tasks': 1},
        4: {'capability': GPUCapability.MULTI_MODEL, 'models': ['qwen2.5-coder', 'mistral'], 'max_tasks': 5},
        5: {'capability': GPUCapability.MULTI_MODEL, 'models': ['qwen2.5-coder', 'mistral'], 'max_tasks': 5},
        6: {'capability': GPUCapability.DYNAMIC, 'models': ['ollama_primary'], 'max_tasks': 8}
    }

    def __init__(self, gpu_host: str = "tpa1"):
        self.gpu_host = gpu_host
        self.gpu_status: Dict[int, GPUStatus] = {}
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Tuple[int, Task]] = {}
        self.monitoring = False

    async def update_gpu_status(self):
        """Query all GPUs and update status"""
        try:
            cmd = ["ssh", self.gpu_host, "nvidia-smi",
                   "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                   "--format=csv,noheader,nounits"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        gpu_id = int(parts[0])
                        assignment = self.GPU_ASSIGNMENTS.get(gpu_id, {
                            'capability': GPUCapability.GENERAL, 'models': [], 'max_tasks': 5
                        })
                        current_tasks = sum(1 for gpu, _ in self.active_tasks.values() if gpu == gpu_id)
                        
                        self.gpu_status[gpu_id] = GPUStatus(
                            gpu_id=gpu_id, name=parts[1],
                            utilization=float(parts[2]),
                            memory_used=int(parts[3]), memory_total=int(parts[4]),
                            memory_free=int(parts[4]) - int(parts[3]),
                            temperature=int(parts[5]),
                            power_draw=float(parts[6]) if parts[6] != 'N/A' else 0.0,
                            capability=assignment['capability'],
                            current_tasks=current_tasks, max_tasks=assignment['max_tasks']
                        )
        except Exception as e:
            logger.error(f"Error updating GPU status: {e}")

    def get_gpu_stats(self) -> Dict:
        """Get current GPU statistics"""
        total_gpus = len(self.gpu_status)
        available_gpus = sum(1 for s in self.gpu_status.values() if s.is_available)
        total_utilization = sum(s.utilization for s in self.gpu_status.values()) / total_gpus if total_gpus > 0 else 0

        return {
            'total_gpus': total_gpus,
            'available_gpus': available_gpus,
            'avg_utilization': total_utilization,
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue)
        }

