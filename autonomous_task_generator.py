"""
Autonomous Task Generator for QAGI System.

This module provides a production-ready, continuously-running task generator that
creates and prioritizes tasks for the Quantum-Agentic General Intelligence (QAGI) system.
It operates 24/7, generating diverse task types based on system capacity and priorities.

Features:
- Continuous task generation across 6 task categories
- Intelligent priority calculation based on multiple factors
- Redis-based task queue management
- GPU Task Distributor integration for capacity monitoring
- Adaptive generation rate based on system load
- Comprehensive error handling and recovery
- Structured logging and metrics collection
- Graceful shutdown handling

Author: QAGI System
Version: 1.0.0
"""

import asyncio
import json
import logging
import random
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

try:
    import redis.asyncio as aioredis
except ImportError:
    import aioredis  # type: ignore

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/qagi/task_generator.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Task type enumeration with capacity allocation percentages."""

    LEARNING = "learning"        # 30% capacity
    OPTIMIZATION = "optimization"  # 20% capacity
    RESEARCH = "research"        # 15% capacity
    MAINTENANCE = "maintenance"  # 10% capacity
    CREATIVE = "creative"        # 15% capacity
    ANALYSIS = "analysis"        # 10% capacity


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"    # Score: 90-100
    HIGH = "high"           # Score: 70-89
    MEDIUM = "medium"       # Score: 40-69
    LOW = "low"            # Score: 0-39


@dataclass
class Task:
    """
    Represents a single task in the QAGI system.

    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (learning, optimization, etc.)
        title: Human-readable task title
        description: Detailed task description
        priority_score: Calculated priority score (0-100)
        priority_level: Priority level based on score
        estimated_duration: Estimated task duration in seconds
        gpu_required: Whether task requires GPU resources
        dependencies: List of task IDs this task depends on
        metadata: Additional task-specific metadata
        created_at: Task creation timestamp
        started_at: Task start timestamp
        completed_at: Task completion timestamp
        status: Current task status
        result: Task execution result
        error: Error message if task failed
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType = TaskType.LEARNING
    title: str = ""
    description: str = ""
    priority_score: float = 50.0
    priority_level: TaskPriority = TaskPriority.MEDIUM
    estimated_duration: int = 300  # seconds
    gpu_required: bool = False
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary representation."""
        # Convert string enums back to enum types
        if 'task_type' in data and isinstance(data['task_type'], str):
            data['task_type'] = TaskType(data['task_type'])
        if 'priority_level' in data and isinstance(data['priority_level'], str):
            data['priority_level'] = TaskPriority(data['priority_level'])
        return cls(**data)


@dataclass
class SystemCapacity:
    """
    Represents current system capacity and resource availability.

    Attributes:
        total_gpu_count: Total number of GPUs available
        available_gpu_count: Number of GPUs currently available
        active_tasks: Number of currently active tasks
        queue_depth: Number of tasks in pending queue
        cpu_usage: Current CPU usage percentage
        memory_usage: Current memory usage percentage
        last_updated: Last capacity update timestamp
    """

    total_gpu_count: int = 8
    available_gpu_count: int = 8
    active_tasks: int = 0
    queue_depth: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AutonomousTaskGenerator:
    """
    Autonomous Task Generator for the QAGI system.

    This class implements a continuously-running task generator that creates,
    prioritizes, and enqueues tasks based on system capacity and intelligent
    prioritization algorithms.

    The generator operates in a continuous loop, generating tasks across six
    categories with specific capacity allocations:
    - Learning: 30%
    - Optimization: 20%
    - Research: 15%
    - Maintenance: 10%
    - Creative: 15%
    - Analysis: 10%

    Tasks are prioritized using a multi-factor scoring algorithm that considers:
    - Task type and urgency
    - System capacity and load
    - Dependencies and prerequisites
    - Impact and value estimation
    - Resource requirements

    Example:
        >>> generator = AutonomousTaskGenerator(
        ...     redis_url="redis://localhost:6379",
        ...     generation_interval=5.0
        ... )
        >>> await generator.start()
    """

    # Task type capacity allocations (percentages)
    CAPACITY_ALLOCATION = {
        TaskType.LEARNING: 0.30,
        TaskType.OPTIMIZATION: 0.20,
        TaskType.RESEARCH: 0.15,
        TaskType.MAINTENANCE: 0.10,
        TaskType.CREATIVE: 0.15,
        TaskType.ANALYSIS: 0.10,
    }

    # Redis key prefixes
    REDIS_PREFIX = "QAGI"
    TASK_QUEUE_KEY = f"{REDIS_PREFIX}:tasks:pending"
    ACTIVE_TASKS_KEY = f"{REDIS_PREFIX}:tasks:active"
    COMPLETED_TASKS_KEY = f"{REDIS_PREFIX}:tasks:completed"
    METRICS_KEY = f"{REDIS_PREFIX}:metrics"
    CAPACITY_KEY = f"{REDIS_PREFIX}:capacity"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        redis_db: int = 0,
        generation_interval: float = 5.0,
        max_queue_size: int = 1000,
        max_concurrent_tasks: int = 100,
        gpu_distributor_url: Optional[str] = None,
        enable_metrics: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize the Autonomous Task Generator.

        Args:
            redis_url: Redis server URL
            redis_db: Redis database number
            generation_interval: Interval between task generation cycles (seconds)
            max_queue_size: Maximum number of tasks in pending queue
            max_concurrent_tasks: Maximum number of concurrent active tasks
            gpu_distributor_url: URL for GPU Task Distributor service
            enable_metrics: Whether to collect and store metrics
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.redis_url = redis_url
        self.redis_db = redis_db
        self.generation_interval = generation_interval
        self.max_queue_size = max_queue_size
        self.max_concurrent_tasks = max_concurrent_tasks
        self.gpu_distributor_url = gpu_distributor_url
        self.enable_metrics = enable_metrics

        # Set logging level
        logger.setLevel(getattr(logging, log_level.upper()))

        # Redis connection
        self.redis: Optional[aioredis.Redis] = None

        # System state
        self.running = False
        self.capacity = SystemCapacity()
        self.task_generation_counts: Dict[TaskType, int] = defaultdict(int)
        self.total_tasks_generated = 0
        self.total_tasks_completed = 0
        self.start_time: Optional[datetime] = None

        # Task generation strategies
        self.task_generators: Dict[TaskType, callable] = {
            TaskType.LEARNING: self._generate_learning_task,
            TaskType.OPTIMIZATION: self._generate_optimization_task,
            TaskType.RESEARCH: self._generate_research_task,
            TaskType.MAINTENANCE: self._generate_maintenance_task,
            TaskType.CREATIVE: self._generate_creative_task,
            TaskType.ANALYSIS: self._generate_analysis_task,
        }

        # Error tracking
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10

        # Shutdown event
        self.shutdown_event = asyncio.Event()

        logger.info(
            f"AutonomousTaskGenerator initialized: "
            f"redis_url={redis_url}, generation_interval={generation_interval}s"
        )

    async def start(self) -> None:
        """
        Start the autonomous task generator.

        This method initializes connections, sets up signal handlers,
        and begins the continuous task generation loop.

        Raises:
            RuntimeError: If generator is already running
        """
        if self.running:
            raise RuntimeError("Task generator is already running")

        logger.info("Starting Autonomous Task Generator...")

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Initialize Redis connection
        await self._connect_redis()

        # Initialize system state
        await self._initialize_system_state()

        # Start background tasks
        self.running = True
        self.start_time = datetime.utcnow()

        try:
            # Run main tasks concurrently
            await asyncio.gather(
                self._task_generation_loop(),
                self._capacity_monitoring_loop(),
                self._metrics_collection_loop(),
                self._cleanup_loop(),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Fatal error in task generator: {e}", exc_info=True)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the task generator.

        This method stops all running tasks, flushes metrics,
        and closes connections.
        """
        if not self.running:
            return

        logger.info("Shutting down Autonomous Task Generator...")
        self.running = False
        self.shutdown_event.set()

        # Flush final metrics
        if self.enable_metrics:
            await self._flush_metrics()

        # Close Redis connection
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

        logger.info("Autonomous Task Generator shutdown complete")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _connect_redis(self) -> None:
        """Establish Redis connection with retry logic."""
        max_retries = 5
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                self.redis = await aioredis.from_url(
                    self.redis_url,
                    db=self.redis_db,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis.ping()
                logger.info("Redis connection established")
                return
            except Exception as e:
                logger.warning(
                    f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    raise RuntimeError("Failed to connect to Redis") from e

    async def _initialize_system_state(self) -> None:
        """Initialize system state and metrics."""
        if not self.redis:
            return

        # Load capacity from Redis or use defaults
        capacity_data = await self.redis.hgetall(self.CAPACITY_KEY)
        if capacity_data:
            self.capacity = SystemCapacity(**{
                k: (int(v) if k.endswith('_count') or k == 'active_tasks' or k == 'queue_depth'
                    else float(v) if k.endswith('_usage') else v)
                for k, v in capacity_data.items()
            })

        # Initialize metrics
        if self.enable_metrics:
            await self.redis.hset(
                self.METRICS_KEY,
                mapping={
                    "generator_started_at": datetime.utcnow().isoformat(),
                    "total_tasks_generated": 0,
                    "total_tasks_completed": 0,
                }
            )

        logger.info(f"System state initialized: {self.capacity}")

    async def _task_generation_loop(self) -> None:
        """
        Main task generation loop.

        Continuously generates tasks based on system capacity and
        task type allocations.
        """
        logger.info("Task generation loop started")

        while self.running:
            try:
                # Update capacity
                await self._update_capacity()

                # Check if we should generate more tasks
                if await self._should_generate_tasks():
                    # Generate tasks for each type based on allocation
                    tasks_to_generate = await self._calculate_tasks_to_generate()

                    for task_type, count in tasks_to_generate.items():
                        for _ in range(count):
                            await self._generate_and_enqueue_task(task_type)

                    # Reset error counter on success
                    self.consecutive_errors = 0
                else:
                    logger.debug("Skipping task generation - system at capacity")

                # Wait for next generation cycle
                await asyncio.sleep(self.generation_interval)

            except Exception as e:
                self.consecutive_errors += 1
                logger.error(
                    f"Error in task generation loop (error {self.consecutive_errors}): {e}",
                    exc_info=True
                )

                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.critical("Max consecutive errors reached, shutting down")
                    await self.shutdown()
                    break

                # Exponential backoff
                await asyncio.sleep(min(2 ** self.consecutive_errors, 60))

        logger.info("Task generation loop stopped")

    async def _capacity_monitoring_loop(self) -> None:
        """
        Continuously monitor system capacity.

        Monitors GPU availability, active tasks, queue depth, and
        system resources.
        """
        logger.info("Capacity monitoring loop started")

        while self.running:
            try:
                await self._update_capacity()
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
            except Exception as e:
                logger.error(f"Error in capacity monitoring: {e}", exc_info=True)
                await asyncio.sleep(10.0)

        logger.info("Capacity monitoring loop stopped")

    async def _metrics_collection_loop(self) -> None:
        """
        Continuously collect and store metrics.

        Tracks task generation rates, completion rates, and
        system performance metrics.
        """
        if not self.enable_metrics:
            return

        logger.info("Metrics collection loop started")

        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(60.0)  # Collect metrics every minute
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}", exc_info=True)
                await asyncio.sleep(60.0)

        logger.info("Metrics collection loop stopped")

    async def _cleanup_loop(self) -> None:
        """
        Periodic cleanup of old tasks and metrics.

        Removes completed tasks older than retention period and
        performs general housekeeping.
        """
        logger.info("Cleanup loop started")

        while self.running:
            try:
                await self._cleanup_old_tasks()
                await asyncio.sleep(3600.0)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(3600.0)

        logger.info("Cleanup loop stopped")

    async def _should_generate_tasks(self) -> bool:
        """
        Determine if more tasks should be generated.

        Returns:
            True if tasks should be generated, False otherwise
        """
        # Check queue size
        if self.capacity.queue_depth >= self.max_queue_size:
            logger.debug(f"Queue at capacity: {self.capacity.queue_depth}/{self.max_queue_size}")
            return False

        # Check active tasks
        if self.capacity.active_tasks >= self.max_concurrent_tasks:
            logger.debug(
                f"Max concurrent tasks reached: "
                f"{self.capacity.active_tasks}/{self.max_concurrent_tasks}"
            )
            return False

        # Check system resources
        if self.capacity.cpu_usage > 90.0 or self.capacity.memory_usage > 90.0:
            logger.warning(
                f"High system resource usage: "
                f"CPU={self.capacity.cpu_usage}%, Memory={self.capacity.memory_usage}%"
            )
            return False

        return True

    async def _calculate_tasks_to_generate(self) -> Dict[TaskType, int]:
        """
        Calculate number of tasks to generate for each type.

        Returns:
            Dictionary mapping task types to generation counts
        """
        # Calculate available capacity
        available_capacity = max(
            0,
            min(
                self.max_queue_size - self.capacity.queue_depth,
                self.max_concurrent_tasks - self.capacity.active_tasks
            )
        )

        # Limit generation batch size
        batch_size = min(available_capacity, 10)

        # Calculate tasks per type based on allocation percentages
        tasks_per_type: Dict[TaskType, int] = {}
        remaining = batch_size

        for task_type, allocation in sorted(
            self.CAPACITY_ALLOCATION.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            count = max(1, int(batch_size * allocation))
            count = min(count, remaining)
            tasks_per_type[task_type] = count
            remaining -= count

            if remaining <= 0:
                break

        # Ensure at least one task type is selected
        if not tasks_per_type and batch_size > 0:
            tasks_per_type[random.choice(list(TaskType))] = 1

        logger.debug(f"Tasks to generate: {tasks_per_type}")
        return tasks_per_type

    async def _generate_and_enqueue_task(self, task_type: TaskType) -> None:
        """
        Generate a task of the specified type and enqueue it.

        Args:
            task_type: Type of task to generate
        """
        try:
            # Generate task
            generator = self.task_generators[task_type]
            task = await generator()

            # Calculate priority
            task.priority_score = self._calculate_priority(task)
            task.priority_level = self._get_priority_level(task.priority_score)

            # Enqueue task
            await self._enqueue_task(task)

            # Update counters
            self.task_generation_counts[task_type] += 1
            self.total_tasks_generated += 1

            logger.info(
                f"Generated {task_type.value} task: {task.task_id} "
                f"(priority={task.priority_score:.1f}, {task.priority_level.value})"
            )

        except Exception as e:
            logger.error(f"Failed to generate {task_type.value} task: {e}", exc_info=True)

    async def _enqueue_task(self, task: Task) -> None:
        """
        Add task to Redis priority queue.

        Args:
            task: Task to enqueue
        """
        if not self.redis:
            return

        # Store task data
        task_key = f"{self.REDIS_PREFIX}:task:{task.task_id}"
        await self.redis.hset(task_key, mapping=task.to_dict())

        # Add to priority queue (sorted set with negative score for highest-first)
        await self.redis.zadd(
            self.TASK_QUEUE_KEY,
            {task.task_id: -task.priority_score}
        )

        logger.debug(f"Task {task.task_id} enqueued with priority {task.priority_score}")

    def _calculate_priority(self, task: Task) -> float:
        """
        Calculate task priority score using multi-factor algorithm.

        Priority factors:
        - Task type urgency (0-20 points)
        - Estimated impact (0-30 points)
        - Resource availability (0-20 points)
        - Dependency readiness (0-15 points)
        - Queue age balance (0-15 points)

        Args:
            task: Task to prioritize

        Returns:
            Priority score (0-100)
        """
        score = 0.0

        # Task type urgency
        urgency_scores = {
            TaskType.MAINTENANCE: 20.0,
            TaskType.OPTIMIZATION: 15.0,
            TaskType.LEARNING: 12.0,
            TaskType.RESEARCH: 10.0,
            TaskType.ANALYSIS: 12.0,
            TaskType.CREATIVE: 8.0,
        }
        score += urgency_scores.get(task.task_type, 10.0)

        # Estimated impact (based on metadata)
        impact = task.metadata.get('impact', 'medium')
        impact_scores = {'critical': 30.0, 'high': 20.0, 'medium': 15.0, 'low': 5.0}
        score += impact_scores.get(impact, 15.0)

        # Resource availability
        if task.gpu_required:
            gpu_availability = self.capacity.available_gpu_count / max(self.capacity.total_gpu_count, 1)
            score += gpu_availability * 20.0
        else:
            score += 15.0  # Bonus for not requiring GPU

        # Dependency readiness (bonus if no dependencies)
        if not task.dependencies:
            score += 15.0
        else:
            score += 5.0  # Partial credit for having dependencies

        # Queue balance (favor underrepresented task types)
        total_generated = max(sum(self.task_generation_counts.values()), 1)
        type_ratio = self.task_generation_counts[task.task_type] / total_generated
        expected_ratio = self.CAPACITY_ALLOCATION[task.task_type]
        if type_ratio < expected_ratio:
            score += 15.0 * (expected_ratio - type_ratio)

        return max(0.0, min(100.0, score))

    def _get_priority_level(self, score: float) -> TaskPriority:
        """
        Convert priority score to priority level.

        Args:
            score: Priority score (0-100)

        Returns:
            Priority level
        """
        if score >= 90:
            return TaskPriority.CRITICAL
        elif score >= 70:
            return TaskPriority.HIGH
        elif score >= 40:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW

    # Task Generation Methods

    async def _generate_learning_task(self) -> Task:
        """
        Generate a learning task (30% capacity allocation).

        Learning tasks focus on acquiring new knowledge, skills,
        and expanding the system's capabilities.

        Returns:
            Generated learning task
        """
        learning_types = [
            "skill_acquisition",
            "knowledge_graph_expansion",
            "concept_mastery",
            "curriculum_completion",
            "competency_development"
        ]

        learning_type = random.choice(learning_types)

        # Generate task based on learning type
        if learning_type == "skill_acquisition":
            skills = ["python_advanced", "distributed_systems", "ml_optimization", "nlp_techniques"]
            skill = random.choice(skills)
            title = f"Acquire skill: {skill}"
            description = f"Learn and practice {skill} through structured exercises and projects"

        elif learning_type == "knowledge_graph_expansion":
            domains = ["quantum_computing", "neuroscience", "mathematics", "philosophy"]
            domain = random.choice(domains)
            title = f"Expand knowledge in {domain}"
            description = f"Research and integrate new concepts in {domain} into knowledge graph"

        elif learning_type == "concept_mastery":
            concepts = ["transformer_architecture", "reinforcement_learning", "graph_theory", "cryptography"]
            concept = random.choice(concepts)
            title = f"Master concept: {concept}"
            description = f"Deep dive into {concept} with theoretical and practical components"

        elif learning_type == "curriculum_completion":
            courses = ["advanced_algorithms", "system_design", "ai_safety", "quantum_ml"]
            course = random.choice(courses)
            title = f"Complete curriculum: {course}"
            description = f"Work through structured curriculum for {course}"

        else:  # competency_development
            competencies = ["problem_solving", "creative_thinking", "analytical_reasoning", "meta_learning"]
            competency = random.choice(competencies)
            title = f"Develop competency: {competency}"
            description = f"Improve {competency} through targeted exercises and reflection"

        return Task(
            task_type=TaskType.LEARNING,
            title=title,
            description=description,
            estimated_duration=random.randint(600, 3600),
            gpu_required=random.random() < 0.3,
            metadata={
                "learning_type": learning_type,
                "impact": random.choice(["high", "medium", "medium", "low"]),
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
            }
        )

    async def _generate_optimization_task(self) -> Task:
        """
        Generate an optimization task (20% capacity allocation).

        Optimization tasks focus on improving system performance,
        resource utilization, and operational efficiency.

        Returns:
            Generated optimization task
        """
        optimization_types = [
            "performance_tuning",
            "resource_allocation",
            "algorithm_improvement",
            "cost_reduction",
            "latency_optimization"
        ]

        opt_type = random.choice(optimization_types)

        if opt_type == "performance_tuning":
            components = ["task_scheduler", "memory_manager", "query_engine", "inference_pipeline"]
            component = random.choice(components)
            title = f"Optimize performance: {component}"
            description = f"Profile and improve {component} performance by 10-20%"

        elif opt_type == "resource_allocation":
            resources = ["GPU_utilization", "CPU_cores", "memory_bandwidth", "network_traffic"]
            resource = random.choice(resources)
            title = f"Optimize resource allocation: {resource}"
            description = f"Improve {resource} efficiency through better allocation strategies"

        elif opt_type == "algorithm_improvement":
            algorithms = ["search", "sort", "hash", "graph_traversal"]
            algorithm = random.choice(algorithms)
            title = f"Improve algorithm: {algorithm}"
            description = f"Optimize {algorithm} implementation for better time/space complexity"

        elif opt_type == "cost_reduction":
            areas = ["compute_costs", "storage_costs", "bandwidth_costs", "energy_consumption"]
            area = random.choice(areas)
            title = f"Reduce costs: {area}"
            description = f"Identify and implement cost savings in {area}"

        else:  # latency_optimization
            services = ["API_endpoints", "database_queries", "model_inference", "data_processing"]
            service = random.choice(services)
            title = f"Reduce latency: {service}"
            description = f"Minimize {service} latency through optimization techniques"

        return Task(
            task_type=TaskType.OPTIMIZATION,
            title=title,
            description=description,
            estimated_duration=random.randint(1800, 7200),
            gpu_required=random.random() < 0.5,
            metadata={
                "optimization_type": opt_type,
                "impact": random.choice(["critical", "high", "high", "medium"]),
                "expected_improvement": f"{random.randint(10, 50)}%",
            }
        )

    async def _generate_research_task(self) -> Task:
        """
        Generate a research task (15% capacity allocation).

        Research tasks focus on exploration, experimentation,
        and discovering novel approaches.

        Returns:
            Generated research task
        """
        research_types = [
            "literature_review",
            "experimental_design",
            "hypothesis_testing",
            "novel_approach_exploration",
            "state_of_art_analysis"
        ]

        research_type = random.choice(research_types)

        if research_type == "literature_review":
            topics = ["few_shot_learning", "neural_architecture_search", "federated_learning", "explainable_ai"]
            topic = random.choice(topics)
            title = f"Literature review: {topic}"
            description = f"Comprehensive review of recent research in {topic}"

        elif research_type == "experimental_design":
            experiments = ["model_architecture", "training_strategy", "data_augmentation", "ensemble_methods"]
            experiment = random.choice(experiments)
            title = f"Design experiment: {experiment}"
            description = f"Create rigorous experimental design for testing {experiment}"

        elif research_type == "hypothesis_testing":
            hypotheses = [
                "transfer_learning_effectiveness",
                "attention_mechanism_variants",
                "optimization_algorithm_comparison",
                "regularization_techniques"
            ]
            hypothesis = random.choice(hypotheses)
            title = f"Test hypothesis: {hypothesis}"
            description = f"Design and execute experiments to validate {hypothesis}"

        elif research_type == "novel_approach_exploration":
            approaches = ["meta_learning", "neural_ODE", "graph_neural_networks", "self_supervised_learning"]
            approach = random.choice(approaches)
            title = f"Explore novel approach: {approach}"
            description = f"Investigate applicability of {approach} to current problems"

        else:  # state_of_art_analysis
            domains = ["computer_vision", "natural_language_processing", "reinforcement_learning", "generative_models"]
            domain = random.choice(domains)
            title = f"State-of-art analysis: {domain}"
            description = f"Analyze current state-of-the-art methods in {domain}"

        return Task(
            task_type=TaskType.RESEARCH,
            title=title,
            description=description,
            estimated_duration=random.randint(3600, 14400),
            gpu_required=random.random() < 0.6,
            metadata={
                "research_type": research_type,
                "impact": random.choice(["high", "medium", "medium", "low"]),
                "exploratory": True,
            }
        )

    async def _generate_maintenance_task(self) -> Task:
        """
        Generate a maintenance task (10% capacity allocation).

        Maintenance tasks focus on system health, updates,
        and general housekeeping.

        Returns:
            Generated maintenance task
        """
        maintenance_types = [
            "health_check",
            "log_rotation",
            "dependency_update",
            "security_patch",
            "database_optimization",
            "cache_cleanup"
        ]

        maint_type = random.choice(maintenance_types)

        if maint_type == "health_check":
            systems = ["database", "cache", "message_queue", "file_system", "network"]
            system = random.choice(systems)
            title = f"Health check: {system}"
            description = f"Perform comprehensive health check on {system}"

        elif maint_type == "log_rotation":
            title = "Rotate and archive logs"
            description = "Rotate system logs and archive old entries"

        elif maint_type == "dependency_update":
            packages = ["security_libs", "ml_frameworks", "system_utilities", "database_drivers"]
            package = random.choice(packages)
            title = f"Update dependencies: {package}"
            description = f"Update {package} to latest stable versions"

        elif maint_type == "security_patch":
            components = ["authentication", "encryption", "api_gateway", "data_access"]
            component = random.choice(components)
            title = f"Apply security patch: {component}"
            description = f"Apply latest security patches to {component}"

        elif maint_type == "database_optimization":
            title = "Optimize database"
            description = "Run database optimization: vacuum, reindex, analyze statistics"

        else:  # cache_cleanup
            title = "Clean up caches"
            description = "Clear stale cache entries and optimize cache storage"

        return Task(
            task_type=TaskType.MAINTENANCE,
            title=title,
            description=description,
            estimated_duration=random.randint(300, 1800),
            gpu_required=False,
            metadata={
                "maintenance_type": maint_type,
                "impact": "medium" if maint_type in ["security_patch", "dependency_update"] else "low",
                "scheduled": random.random() < 0.7,
            }
        )

    async def _generate_creative_task(self) -> Task:
        """
        Generate a creative task (15% capacity allocation).

        Creative tasks focus on generation, synthesis,
        and innovative problem-solving.

        Returns:
            Generated creative task
        """
        creative_types = [
            "solution_generation",
            "cross_domain_synthesis",
            "pattern_discovery",
            "novel_combination",
            "artistic_creation"
        ]

        creative_type = random.choice(creative_types)

        if creative_type == "solution_generation":
            problems = ["scalability", "efficiency", "user_experience", "data_quality"]
            problem = random.choice(problems)
            title = f"Generate solutions for: {problem}"
            description = f"Brainstorm and evaluate creative solutions to improve {problem}"

        elif creative_type == "cross_domain_synthesis":
            domains = [
                ("biology", "computing"),
                ("music", "mathematics"),
                ("architecture", "algorithms"),
                ("linguistics", "machine_learning")
            ]
            domain1, domain2 = random.choice(domains)
            title = f"Synthesize insights: {domain1} + {domain2}"
            description = f"Find creative connections between {domain1} and {domain2}"

        elif creative_type == "pattern_discovery":
            datasets = ["user_behavior", "system_metrics", "error_logs", "performance_data"]
            dataset = random.choice(datasets)
            title = f"Discover patterns in: {dataset}"
            description = f"Apply creative analysis to find novel patterns in {dataset}"

        elif creative_type == "novel_combination":
            techniques = [
                ("ensemble_learning", "active_learning"),
                ("transfer_learning", "meta_learning"),
                ("generative_models", "reinforcement_learning"),
                ("attention_mechanisms", "graph_networks")
            ]
            tech1, tech2 = random.choice(techniques)
            title = f"Combine approaches: {tech1} + {tech2}"
            description = f"Explore creative combinations of {tech1} and {tech2}"

        else:  # artistic_creation
            artifacts = ["visualization", "interface_design", "documentation", "presentation"]
            artifact = random.choice(artifacts)
            title = f"Create artistic {artifact}"
            description = f"Design innovative and aesthetic {artifact} for system components"

        return Task(
            task_type=TaskType.CREATIVE,
            title=title,
            description=description,
            estimated_duration=random.randint(1800, 5400),
            gpu_required=random.random() < 0.4,
            metadata={
                "creative_type": creative_type,
                "impact": random.choice(["high", "medium", "medium", "low"]),
                "novelty": random.choice(["very_high", "high", "medium"]),
            }
        )

    async def _generate_analysis_task(self) -> Task:
        """
        Generate an analysis task (10% capacity allocation).

        Analysis tasks focus on data analysis, pattern recognition,
        and deriving insights.

        Returns:
            Generated analysis task
        """
        analysis_types = [
            "performance_analysis",
            "trend_analysis",
            "anomaly_detection",
            "correlation_analysis",
            "predictive_modeling"
        ]

        analysis_type = random.choice(analysis_types)

        if analysis_type == "performance_analysis":
            metrics = ["latency", "throughput", "error_rates", "resource_usage"]
            metric = random.choice(metrics)
            title = f"Analyze performance: {metric}"
            description = f"Detailed analysis of {metric} trends and bottlenecks"

        elif analysis_type == "trend_analysis":
            data_sources = ["user_activity", "system_load", "task_completion", "error_patterns"]
            source = random.choice(data_sources)
            title = f"Analyze trends: {source}"
            description = f"Identify and characterize trends in {source} data"

        elif analysis_type == "anomaly_detection":
            systems = ["network_traffic", "database_queries", "api_requests", "resource_consumption"]
            system = random.choice(systems)
            title = f"Detect anomalies: {system}"
            description = f"Identify unusual patterns and anomalies in {system}"

        elif analysis_type == "correlation_analysis":
            variables = [
                ("task_complexity", "completion_time"),
                ("resource_allocation", "performance"),
                ("error_frequency", "system_load"),
                ("user_patterns", "system_efficiency")
            ]
            var1, var2 = random.choice(variables)
            title = f"Analyze correlation: {var1} vs {var2}"
            description = f"Study relationship between {var1} and {var2}"

        else:  # predictive_modeling
            targets = ["system_load", "failure_probability", "resource_demand", "task_duration"]
            target = random.choice(targets)
            title = f"Build predictive model: {target}"
            description = f"Create predictive model for {target} based on historical data"

        return Task(
            task_type=TaskType.ANALYSIS,
            title=title,
            description=description,
            estimated_duration=random.randint(1200, 3600),
            gpu_required=random.random() < 0.5,
            metadata={
                "analysis_type": analysis_type,
                "impact": random.choice(["high", "medium", "medium", "low"]),
                "data_driven": True,
            }
        )

    async def _update_capacity(self) -> None:
        """Update system capacity information."""
        if not self.redis:
            return

        try:
            # Get queue depth
            queue_depth = await self.redis.zcard(self.TASK_QUEUE_KEY)

            # Get active tasks count
            active_tasks = await self.redis.hlen(self.ACTIVE_TASKS_KEY)

            # Update capacity
            self.capacity.queue_depth = queue_depth
            self.capacity.active_tasks = active_tasks
            self.capacity.last_updated = datetime.utcnow().isoformat()

            # Try to get GPU info from distributor (simplified)
            # In production, this would make an actual API call
            if self.gpu_distributor_url:
                # Placeholder for GPU distributor integration
                # self.capacity.available_gpu_count = await self._query_gpu_distributor()
                pass

            # Store capacity in Redis
            await self.redis.hset(
                self.CAPACITY_KEY,
                mapping={
                    "total_gpu_count": self.capacity.total_gpu_count,
                    "available_gpu_count": self.capacity.available_gpu_count,
                    "active_tasks": self.capacity.active_tasks,
                    "queue_depth": self.capacity.queue_depth,
                    "cpu_usage": self.capacity.cpu_usage,
                    "memory_usage": self.capacity.memory_usage,
                    "last_updated": self.capacity.last_updated,
                }
            )

        except Exception as e:
            logger.error(f"Failed to update capacity: {e}", exc_info=True)

    async def _collect_metrics(self) -> None:
        """Collect and store system metrics."""
        if not self.redis or not self.enable_metrics:
            return

        try:
            # Calculate uptime
            uptime_seconds = 0
            if self.start_time:
                uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()

            # Calculate generation rate
            generation_rate = 0.0
            if uptime_seconds > 0:
                generation_rate = self.total_tasks_generated / uptime_seconds * 60  # per minute

            # Store metrics
            metrics = {
                "uptime_seconds": int(uptime_seconds),
                "total_tasks_generated": self.total_tasks_generated,
                "generation_rate_per_minute": f"{generation_rate:.2f}",
                "queue_depth": self.capacity.queue_depth,
                "active_tasks": self.capacity.active_tasks,
                "last_updated": datetime.utcnow().isoformat(),
            }

            # Add per-type counts
            for task_type, count in self.task_generation_counts.items():
                metrics[f"generated_{task_type.value}"] = count

            await self.redis.hset(self.METRICS_KEY, mapping=metrics)

            logger.debug(
                f"Metrics updated: generated={self.total_tasks_generated}, "
                f"rate={generation_rate:.2f}/min, queue={self.capacity.queue_depth}"
            )

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}", exc_info=True)

    async def _flush_metrics(self) -> None:
        """Flush final metrics before shutdown."""
        await self._collect_metrics()
        logger.info("Final metrics flushed")

    async def _cleanup_old_tasks(self) -> None:
        """Remove completed tasks older than retention period."""
        if not self.redis:
            return

        try:
            # Retention period: 7 days
            retention_seconds = 7 * 24 * 3600
            cutoff_time = datetime.utcnow() - timedelta(seconds=retention_seconds)

            # Get all completed task IDs
            completed_task_ids = await self.redis.zrange(
                self.COMPLETED_TASKS_KEY,
                0,
                -1
            )

            removed_count = 0
            for task_id in completed_task_ids:
                task_key = f"{self.REDIS_PREFIX}:task:{task_id}"
                task_data = await self.redis.hgetall(task_key)

                if task_data and 'completed_at' in task_data:
                    completed_at = datetime.fromisoformat(task_data['completed_at'])
                    if completed_at < cutoff_time:
                        # Remove task
                        await self.redis.delete(task_key)
                        await self.redis.zrem(self.COMPLETED_TASKS_KEY, task_id)
                        removed_count += 1

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old completed tasks")

        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {e}", exc_info=True)

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current generator status.

        Returns:
            Dictionary containing generator status and metrics
        """
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "running": self.running,
            "uptime_seconds": int(uptime_seconds),
            "total_tasks_generated": self.total_tasks_generated,
            "generation_counts": dict(self.task_generation_counts),
            "capacity": {
                "queue_depth": self.capacity.queue_depth,
                "active_tasks": self.capacity.active_tasks,
                "available_gpus": self.capacity.available_gpu_count,
                "cpu_usage": self.capacity.cpu_usage,
                "memory_usage": self.capacity.memory_usage,
            },
            "consecutive_errors": self.consecutive_errors,
        }


async def main():
    """
    Main entry point for the Autonomous Task Generator.

    This function initializes and starts the task generator with
    default configuration. Configuration can be customized via
    environment variables.
    """
    import os

    # Load configuration from environment
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_db = int(os.getenv("REDIS_DB", "0"))
    generation_interval = float(os.getenv("GENERATION_INTERVAL", "5.0"))
    max_queue_size = int(os.getenv("MAX_QUEUE_SIZE", "1000"))
    max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "100"))
    gpu_distributor_url = os.getenv("GPU_DISTRIBUTOR_URL")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    # Create and start generator
    generator = AutonomousTaskGenerator(
        redis_url=redis_url,
        redis_db=redis_db,
        generation_interval=generation_interval,
        max_queue_size=max_queue_size,
        max_concurrent_tasks=max_concurrent_tasks,
        gpu_distributor_url=gpu_distributor_url,
        enable_metrics=True,
        log_level=log_level
    )

    logger.info("=" * 80)
    logger.info("QAGI Autonomous Task Generator")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Redis URL: {redis_url}")
    logger.info(f"  Generation Interval: {generation_interval}s")
    logger.info(f"  Max Queue Size: {max_queue_size}")
    logger.info(f"  Max Concurrent Tasks: {max_concurrent_tasks}")
    logger.info(f"  Log Level: {log_level}")
    logger.info("=" * 80)

    await generator.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
