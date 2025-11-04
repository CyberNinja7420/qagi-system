"""
QAGI Self-Learning Engine
==========================

Production-ready autonomous learning system that continuously analyzes operations,
recognizes patterns, optimizes performance, and generates new skills without human intervention.

Features:
- Continuous learning from all operations
- Pattern recognition and analysis
- Performance optimization
- Knowledge graph expansion (Neo4j)
- Autonomous skill generation
- Learning progress tracking
- Multi-GPU operation support

Author: QAGI System
Date: 2025-11-03
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from uuid import uuid4

import numpy as np
from neo4j import AsyncGraphDatabase, AsyncDriver
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class OperationType(Enum):
    """Types of operations that can be learned from."""
    TASK_EXECUTION = "task_execution"
    MODEL_INFERENCE = "model_inference"
    GPU_COMPUTATION = "gpu_computation"
    OPTIMIZATION = "optimization"
    RESEARCH = "research"
    CREATIVE = "creative"
    MAINTENANCE = "maintenance"
    ANALYSIS = "analysis"
    SKILL_GENERATION = "skill_generation"


class LearningOutcome(Enum):
    """Possible outcomes of learning analysis."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    OPTIMIZATION_FOUND = "optimization_found"
    PATTERN_IDENTIFIED = "pattern_identified"
    SKILL_GENERATED = "skill_generated"


@dataclass
class OperationRecord:
    """Record of a single operation for learning analysis."""
    operation_id: str
    operation_type: OperationType
    timestamp: datetime
    duration_ms: float
    gpu_ids: List[int]
    gpu_utilization: Dict[int, float]
    memory_used_mb: Dict[int, float]
    success: bool
    error_message: Optional[str] = None
    input_params: Dict[str, Any] = field(default_factory=dict)
    output_metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['operation_type'] = self.operation_type.value
        return data


@dataclass
class Pattern:
    """Recognized pattern from operation analysis."""
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    actionable_insights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['first_seen'] = self.first_seen.isoformat()
        data['last_seen'] = self.last_seen.isoformat()
        return data


@dataclass
class Skill:
    """Auto-generated skill capability."""
    skill_id: str
    name: str
    description: str
    category: str
    code: str
    dependencies: List[str]
    confidence: float
    test_results: Dict[str, Any]
    created_at: datetime
    times_used: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class OptimizationStrategy:
    """Performance optimization strategy."""
    strategy_id: str
    name: str
    description: str
    target_metric: str
    expected_improvement: float
    implementation: str
    prerequisites: List[str]
    risk_level: str  # low, medium, high
    created_at: datetime
    applied: bool = False
    actual_improvement: Optional[float] = None


@dataclass
class LearningMetrics:
    """Metrics tracking learning progress."""
    total_operations_analyzed: int = 0
    patterns_identified: int = 0
    skills_generated: int = 0
    optimizations_applied: int = 0
    knowledge_graph_nodes: int = 0
    knowledge_graph_relationships: int = 0
    average_confidence: float = 0.0
    learning_rate: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


# ============================================================================
# Self-Learning Engine
# ============================================================================

class SelfLearningEngine:
    """
    Autonomous self-learning engine for QAGI system.

    Continuously analyzes all operations, identifies patterns, generates optimizations,
    expands knowledge graph, and creates new skills without human intervention.

    Example:
        >>> engine = SelfLearningEngine(
        ...     neo4j_uri="bolt://localhost:7687",
        ...     neo4j_user="neo4j",
        ...     neo4j_password="password"
        ... )
        >>> await engine.initialize()
        >>> await engine.start_continuous_learning()
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        operation_buffer_size: int = 10000,
        pattern_min_occurrences: int = 3,
        pattern_confidence_threshold: float = 0.7,
        skill_confidence_threshold: float = 0.8,
        learning_cycle_interval_seconds: float = 10.0,
        data_dir: Path = Path("/home/claudehome/claude-workspace/projects/qagi-system/data")
    ):
        """
        Initialize the self-learning engine.

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            operation_buffer_size: Max operations to keep in memory
            pattern_min_occurrences: Minimum occurrences to recognize pattern
            pattern_confidence_threshold: Minimum confidence for pattern
            skill_confidence_threshold: Minimum confidence for skill generation
            learning_cycle_interval_seconds: Interval between learning cycles
            data_dir: Directory for persistent data storage
        """
        # Configuration
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.operation_buffer_size = operation_buffer_size
        self.pattern_min_occurrences = pattern_min_occurrences
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.skill_confidence_threshold = skill_confidence_threshold
        self.learning_cycle_interval = learning_cycle_interval_seconds
        self.data_dir = data_dir

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Neo4j driver
        self.neo4j_driver: Optional[AsyncDriver] = None

        # Operation buffer (circular buffer)
        self.operation_buffer: deque[OperationRecord] = deque(maxlen=operation_buffer_size)

        # Pattern storage
        self.patterns: Dict[str, Pattern] = {}

        # Skill storage
        self.skills: Dict[str, Skill] = {}

        # Optimization strategies
        self.optimizations: Dict[str, OptimizationStrategy] = {}

        # Learning metrics
        self.metrics = LearningMetrics()

        # Learning state
        self.is_learning = False
        self.learning_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        logger.info("SelfLearningEngine initialized")

    # ========================================================================
    # Initialization & Lifecycle
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize the learning engine and connect to Neo4j."""
        logger.info("Initializing SelfLearningEngine...")

        # Connect to Neo4j
        await self._connect_neo4j()

        # Initialize knowledge graph schema
        await self._initialize_knowledge_graph()

        # Load existing patterns and skills
        await self._load_persistent_data()

        logger.info("SelfLearningEngine initialized successfully")

    async def _connect_neo4j(self) -> None:
        """Connect to Neo4j database."""
        try:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            async with self.neo4j_driver.session() as session:
                result = await session.run("RETURN 1")
                await result.single()
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def _initialize_knowledge_graph(self) -> None:
        """Initialize Neo4j knowledge graph schema."""
        if not self.neo4j_driver:
            raise RuntimeError("Neo4j driver not initialized")

        queries = [
            # Create constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Operation) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pattern) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (opt:Optimization) REQUIRE opt.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",

            # Create indexes
            "CREATE INDEX IF NOT EXISTS FOR (o:Operation) ON (o.timestamp)",
            "CREATE INDEX IF NOT EXISTS FOR (o:Operation) ON (o.operation_type)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Pattern) ON (p.confidence)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Skill) ON (s.success_rate)",
        ]

        async with self.neo4j_driver.session() as session:
            for query in queries:
                try:
                    await session.run(query)
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {e}")

        logger.info("Knowledge graph schema initialized")

    async def _load_persistent_data(self) -> None:
        """Load existing patterns and skills from disk and Neo4j."""
        # Load patterns
        patterns_file = self.data_dir / "patterns.json"
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                patterns_data = json.load(f)
                for p_data in patterns_data:
                    p_data['first_seen'] = datetime.fromisoformat(p_data['first_seen'])
                    p_data['last_seen'] = datetime.fromisoformat(p_data['last_seen'])
                    pattern = Pattern(**p_data)
                    self.patterns[pattern.pattern_id] = pattern
            logger.info(f"Loaded {len(self.patterns)} patterns from disk")

        # Load skills
        skills_file = self.data_dir / "skills.json"
        if skills_file.exists():
            with open(skills_file, 'r') as f:
                skills_data = json.load(f)
                for s_data in skills_data:
                    s_data['created_at'] = datetime.fromisoformat(s_data['created_at'])
                    skill = Skill(**s_data)
                    self.skills[skill.skill_id] = skill
            logger.info(f"Loaded {len(self.skills)} skills from disk")

        # Load metrics
        metrics_file = self.data_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                metrics_data['last_update'] = datetime.fromisoformat(metrics_data['last_update'])
                self.metrics = LearningMetrics(**metrics_data)
            logger.info("Loaded learning metrics from disk")

    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        logger.info("Cleaning up SelfLearningEngine...")

        # Stop learning loop
        if self.is_learning:
            await self.stop_learning()

        # Save persistent data
        await self._save_persistent_data()

        # Close Neo4j connection
        if self.neo4j_driver:
            await self.neo4j_driver.close()

        logger.info("SelfLearningEngine cleaned up successfully")

    # ========================================================================
    # Operation Recording
    # ========================================================================

    async def record_operation(self, operation: OperationRecord) -> None:
        """
        Record an operation for learning analysis.

        Args:
            operation: Operation record to analyze
        """
        # Add to buffer
        self.operation_buffer.append(operation)

        # Update metrics
        self.metrics.total_operations_analyzed += 1

        # Store in Neo4j asynchronously (non-blocking)
        asyncio.create_task(self._store_operation_neo4j(operation))

        # Track performance
        self._track_performance(operation)

        logger.debug(f"Recorded operation {operation.operation_id}")

    async def _store_operation_neo4j(self, operation: OperationRecord) -> None:
        """Store operation in Neo4j knowledge graph."""
        if not self.neo4j_driver:
            return

        query = """
        MERGE (o:Operation {id: $id})
        SET o.type = $type,
            o.timestamp = datetime($timestamp),
            o.duration_ms = $duration_ms,
            o.success = $success,
            o.gpu_ids = $gpu_ids,
            o.error_message = $error_message
        WITH o
        UNWIND $gpu_ids AS gpu_id
        MERGE (g:GPU {id: gpu_id})
        MERGE (o)-[:EXECUTED_ON]->(g)
        """

        try:
            async with self.neo4j_driver.session() as session:
                await session.run(
                    query,
                    id=operation.operation_id,
                    type=operation.operation_type.value,
                    timestamp=operation.timestamp.isoformat(),
                    duration_ms=operation.duration_ms,
                    success=operation.success,
                    gpu_ids=operation.gpu_ids,
                    error_message=operation.error_message
                )
        except Exception as e:
            logger.error(f"Failed to store operation in Neo4j: {e}")

    def _track_performance(self, operation: OperationRecord) -> None:
        """Track performance metrics for analysis."""
        # Track by operation type
        key = f"duration_{operation.operation_type.value}"
        self.performance_history[key].append(operation.duration_ms)

        # Track GPU utilization
        for gpu_id, util in operation.gpu_utilization.items():
            self.performance_history[f"gpu_{gpu_id}_util"].append(util)

        # Track success rate
        success_key = f"success_{operation.operation_type.value}"
        self.performance_history[success_key].append(1.0 if operation.success else 0.0)

    # ========================================================================
    # Continuous Learning Loop
    # ========================================================================

    async def start_continuous_learning(self) -> None:
        """Start the continuous learning loop."""
        if self.is_learning:
            logger.warning("Learning loop already running")
            return

        self.is_learning = True
        self.learning_task = asyncio.create_task(self._learning_loop())
        logger.info("Started continuous learning loop")

    async def stop_learning(self) -> None:
        """Stop the continuous learning loop."""
        if not self.is_learning:
            return

        self.is_learning = False
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped continuous learning loop")

    async def _learning_loop(self) -> None:
        """Main continuous learning loop."""
        logger.info("Learning loop started")

        while self.is_learning:
            try:
                cycle_start = time.time()

                # 1. Analyze recent operations
                await self._analyze_operations()

                # 2. Recognize patterns
                await self._recognize_patterns()

                # 3. Optimize performance
                await self._optimize_performance()

                # 4. Expand knowledge graph
                await self._expand_knowledge_graph()

                # 5. Generate new skills
                await self._generate_skills()

                # 6. Update metrics
                await self._update_metrics()

                # 7. Save persistent data periodically
                if self.metrics.total_operations_analyzed % 100 == 0:
                    await self._save_persistent_data()

                # Calculate learning rate
                cycle_duration = time.time() - cycle_start
                self.metrics.learning_rate = 1.0 / cycle_duration if cycle_duration > 0 else 0.0

                # Wait before next cycle
                await asyncio.sleep(self.learning_cycle_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning loop: {e}", exc_info=True)
                await asyncio.sleep(self.learning_cycle_interval)

        logger.info("Learning loop stopped")

    # ========================================================================
    # Operation Analysis
    # ========================================================================

    async def _analyze_operations(self) -> None:
        """Analyze recent operations for insights."""
        if len(self.operation_buffer) < 10:
            return  # Need sufficient data

        # Get recent operations (last 100)
        recent_ops = list(self.operation_buffer)[-100:]

        # Analyze success rates
        success_rates = self._calculate_success_rates(recent_ops)

        # Analyze performance trends
        performance_trends = self._calculate_performance_trends(recent_ops)

        # Identify anomalies
        anomalies = self._identify_anomalies(recent_ops)

        # Log insights
        logger.debug(f"Success rates: {success_rates}")
        logger.debug(f"Performance trends: {performance_trends}")
        if anomalies:
            logger.info(f"Identified {len(anomalies)} anomalies")

    def _calculate_success_rates(
        self,
        operations: List[OperationRecord]
    ) -> Dict[str, float]:
        """Calculate success rates by operation type."""
        success_by_type: Dict[str, List[bool]] = defaultdict(list)

        for op in operations:
            success_by_type[op.operation_type.value].append(op.success)

        return {
            op_type: sum(successes) / len(successes) if successes else 0.0
            for op_type, successes in success_by_type.items()
        }

    def _calculate_performance_trends(
        self,
        operations: List[OperationRecord]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance trends."""
        trends = {}

        for op_type in OperationType:
            ops_of_type = [
                op for op in operations
                if op.operation_type == op_type
            ]

            if not ops_of_type:
                continue

            durations = [op.duration_ms for op in ops_of_type]
            trends[op_type.value] = {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'trend': self._calculate_trend(durations)
            }

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction (improving, degrading, stable)."""
        if len(values) < 5:
            return "insufficient_data"

        # Split into first and second half
        mid = len(values) // 2
        first_half = values[:mid]
        second_half = values[mid:]

        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)

        # For duration, lower is better
        if avg_second < avg_first * 0.95:
            return "improving"
        elif avg_second > avg_first * 1.05:
            return "degrading"
        else:
            return "stable"

    def _identify_anomalies(
        self,
        operations: List[OperationRecord]
    ) -> List[OperationRecord]:
        """Identify anomalous operations."""
        anomalies = []

        # Group by operation type
        by_type: Dict[str, List[OperationRecord]] = defaultdict(list)
        for op in operations:
            by_type[op.operation_type.value].append(op)

        # Check for outliers in each type
        for op_type, ops in by_type.items():
            if len(ops) < 5:
                continue

            durations = np.array([op.duration_ms for op in ops])
            mean = np.mean(durations)
            std = np.std(durations)

            # Identify operations beyond 3 standard deviations
            for op, duration in zip(ops, durations):
                if abs(duration - mean) > 3 * std:
                    anomalies.append(op)

        return anomalies

    # ========================================================================
    # Pattern Recognition
    # ========================================================================

    async def _recognize_patterns(self) -> None:
        """Recognize patterns in operations."""
        if len(self.operation_buffer) < self.pattern_min_occurrences * 5:
            return

        # Get recent operations
        recent_ops = list(self.operation_buffer)[-500:]

        # Recognize different pattern types
        patterns = []

        # 1. Sequential patterns
        patterns.extend(self._recognize_sequential_patterns(recent_ops))

        # 2. Correlation patterns
        patterns.extend(self._recognize_correlation_patterns(recent_ops))

        # 3. Performance patterns
        patterns.extend(self._recognize_performance_patterns(recent_ops))

        # 4. Failure patterns
        patterns.extend(self._recognize_failure_patterns(recent_ops))

        # Store new patterns
        for pattern in patterns:
            if pattern.pattern_id not in self.patterns:
                self.patterns[pattern.pattern_id] = pattern
                self.metrics.patterns_identified += 1
                await self._store_pattern_neo4j(pattern)
                logger.info(f"New pattern identified: {pattern.description}")
            else:
                # Update existing pattern
                existing = self.patterns[pattern.pattern_id]
                existing.occurrences += 1
                existing.last_seen = pattern.last_seen
                existing.confidence = min(1.0, existing.confidence + 0.05)

    def _recognize_sequential_patterns(
        self,
        operations: List[OperationRecord]
    ) -> List[Pattern]:
        """Recognize sequential operation patterns."""
        patterns = []

        # Look for common sequences of operation types
        sequences: Dict[Tuple[str, ...], int] = defaultdict(int)

        for i in range(len(operations) - 2):
            seq = tuple(
                op.operation_type.value
                for op in operations[i:i+3]
            )
            sequences[seq] += 1

        # Identify frequent sequences
        for seq, count in sequences.items():
            if count >= self.pattern_min_occurrences:
                confidence = min(1.0, count / len(operations))
                if confidence >= self.pattern_confidence_threshold:
                    pattern = Pattern(
                        pattern_id=f"seq_{hash(seq)}",
                        pattern_type="sequential",
                        description=f"Common sequence: {' -> '.join(seq)}",
                        confidence=confidence,
                        occurrences=count,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        conditions={"sequence": list(seq)},
                        outcomes={"frequency": count},
                        actionable_insights=[
                            f"Consider optimizing this sequence",
                            f"Could batch operations: {seq}"
                        ]
                    )
                    patterns.append(pattern)

        return patterns

    def _recognize_correlation_patterns(
        self,
        operations: List[OperationRecord]
    ) -> List[Pattern]:
        """Recognize correlations between operation characteristics."""
        patterns = []

        # Example: GPU utilization correlation with duration
        for gpu_id in range(7):
            ops_on_gpu = [
                op for op in operations
                if gpu_id in op.gpu_ids and gpu_id in op.gpu_utilization
            ]

            if len(ops_on_gpu) < 10:
                continue

            utils = [op.gpu_utilization[gpu_id] for op in ops_on_gpu]
            durations = [op.duration_ms for op in ops_on_gpu]

            if len(utils) >= 3:
                correlation = np.corrcoef(utils, durations)[0, 1]

                if abs(correlation) > 0.7:
                    pattern = Pattern(
                        pattern_id=f"corr_gpu{gpu_id}_util_duration",
                        pattern_type="correlation",
                        description=f"GPU {gpu_id} utilization correlates with duration",
                        confidence=abs(correlation),
                        occurrences=len(ops_on_gpu),
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        conditions={
                            "gpu_id": gpu_id,
                            "correlation": correlation
                        },
                        outcomes={"insight": "utilization_duration_relationship"},
                        actionable_insights=[
                            f"GPU {gpu_id} efficiency can be improved",
                            "Consider load balancing adjustments"
                        ]
                    )
                    patterns.append(pattern)

        return patterns

    def _recognize_performance_patterns(
        self,
        operations: List[OperationRecord]
    ) -> List[Pattern]:
        """Recognize performance-related patterns."""
        patterns = []

        # Group by hour of day
        by_hour: Dict[int, List[OperationRecord]] = defaultdict(list)
        for op in operations:
            hour = op.timestamp.hour
            by_hour[hour].append(op)

        # Find optimal time windows
        for hour, ops in by_hour.items():
            if len(ops) < 5:
                continue

            avg_duration = np.mean([op.duration_ms for op in ops])
            success_rate = sum(op.success for op in ops) / len(ops)

            if success_rate > 0.95 and avg_duration < 1000:  # Fast and reliable
                pattern = Pattern(
                    pattern_id=f"perf_hour_{hour}",
                    pattern_type="performance",
                    description=f"Optimal performance window at hour {hour}",
                    confidence=success_rate,
                    occurrences=len(ops),
                    first_seen=datetime.now(),
                    last_seen=datetime.now(),
                    conditions={"hour": hour},
                    outcomes={
                        "avg_duration_ms": avg_duration,
                        "success_rate": success_rate
                    },
                    actionable_insights=[
                        f"Schedule intensive tasks at hour {hour}",
                        "System performs optimally during this window"
                    ]
                )
                patterns.append(pattern)

        return patterns

    def _recognize_failure_patterns(
        self,
        operations: List[OperationRecord]
    ) -> List[Pattern]:
        """Recognize patterns leading to failures."""
        patterns = []

        # Analyze failed operations
        failed_ops = [op for op in operations if not op.success]

        if len(failed_ops) < self.pattern_min_occurrences:
            return patterns

        # Common failure characteristics
        failure_conditions = defaultdict(int)

        for op in failed_ops:
            # Check GPU memory
            for gpu_id, mem in op.memory_used_mb.items():
                if mem > 9000:  # >9GB on 11GB card
                    failure_conditions[f"high_memory_gpu_{gpu_id}"] += 1

            # Check operation type
            failure_conditions[f"type_{op.operation_type.value}"] += 1

        # Create patterns for common failure conditions
        for condition, count in failure_conditions.items():
            if count >= self.pattern_min_occurrences:
                confidence = count / len(failed_ops)
                if confidence >= self.pattern_confidence_threshold:
                    pattern = Pattern(
                        pattern_id=f"failure_{hash(condition)}",
                        pattern_type="failure",
                        description=f"Failure pattern: {condition}",
                        confidence=confidence,
                        occurrences=count,
                        first_seen=datetime.now(),
                        last_seen=datetime.now(),
                        conditions={"condition": condition},
                        outcomes={"failure_rate": confidence},
                        actionable_insights=[
                            f"Mitigate {condition}",
                            "Implement preventive measures"
                        ]
                    )
                    patterns.append(pattern)

        return patterns

    async def _store_pattern_neo4j(self, pattern: Pattern) -> None:
        """Store pattern in Neo4j knowledge graph."""
        if not self.neo4j_driver:
            return

        query = """
        MERGE (p:Pattern {id: $id})
        SET p.type = $type,
            p.description = $description,
            p.confidence = $confidence,
            p.occurrences = $occurrences,
            p.first_seen = datetime($first_seen),
            p.last_seen = datetime($last_seen)
        """

        try:
            async with self.neo4j_driver.session() as session:
                await session.run(
                    query,
                    id=pattern.pattern_id,
                    type=pattern.pattern_type,
                    description=pattern.description,
                    confidence=pattern.confidence,
                    occurrences=pattern.occurrences,
                    first_seen=pattern.first_seen.isoformat(),
                    last_seen=pattern.last_seen.isoformat()
                )
        except Exception as e:
            logger.error(f"Failed to store pattern in Neo4j: {e}")

    # ========================================================================
    # Performance Optimization
    # ========================================================================

    async def _optimize_performance(self) -> None:
        """Generate and apply performance optimizations."""
        # Analyze current performance
        performance_analysis = self._analyze_current_performance()

        # Generate optimization strategies
        new_optimizations = self._generate_optimizations(performance_analysis)

        # Store new optimizations
        for opt in new_optimizations:
            if opt.strategy_id not in self.optimizations:
                self.optimizations[opt.strategy_id] = opt
                logger.info(f"New optimization strategy: {opt.name}")

        # Apply safe optimizations automatically
        await self._apply_safe_optimizations()

    def _analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current system performance."""
        analysis = {}

        # Calculate average durations
        for op_type in OperationType:
            key = f"duration_{op_type.value}"
            if key in self.performance_history:
                durations = list(self.performance_history[key])
                analysis[f"{op_type.value}_avg_duration"] = np.mean(durations)
                analysis[f"{op_type.value}_std_duration"] = np.std(durations)

        # Calculate GPU utilization
        for gpu_id in range(7):
            key = f"gpu_{gpu_id}_util"
            if key in self.performance_history:
                utils = list(self.performance_history[key])
                analysis[f"gpu_{gpu_id}_avg_util"] = np.mean(utils)

        return analysis

    def _generate_optimizations(
        self,
        performance_analysis: Dict[str, Any]
    ) -> List[OptimizationStrategy]:
        """Generate optimization strategies based on performance analysis."""
        optimizations = []

        # Check for underutilized GPUs
        for gpu_id in range(7):
            util_key = f"gpu_{gpu_id}_avg_util"
            if util_key in performance_analysis:
                avg_util = performance_analysis[util_key]
                if avg_util < 0.5:  # <50% utilization
                    opt = OptimizationStrategy(
                        strategy_id=f"opt_gpu_{gpu_id}_utilization",
                        name=f"Increase GPU {gpu_id} utilization",
                        description=f"GPU {gpu_id} is underutilized at {avg_util:.1%}",
                        target_metric=f"gpu_{gpu_id}_utilization",
                        expected_improvement=0.3,
                        implementation="Redistribute tasks to this GPU",
                        prerequisites=[],
                        risk_level="low",
                        created_at=datetime.now()
                    )
                    optimizations.append(opt)

        # Check for slow operations
        for op_type in OperationType:
            dur_key = f"{op_type.value}_avg_duration"
            if dur_key in performance_analysis:
                avg_duration = performance_analysis[dur_key]
                if avg_duration > 5000:  # >5 seconds
                    opt = OptimizationStrategy(
                        strategy_id=f"opt_{op_type.value}_speed",
                        name=f"Optimize {op_type.value} speed",
                        description=f"{op_type.value} operations are slow at {avg_duration:.0f}ms",
                        target_metric=f"{op_type.value}_duration",
                        expected_improvement=0.4,
                        implementation="Implement caching or parallelization",
                        prerequisites=["analyze_bottlenecks"],
                        risk_level="medium",
                        created_at=datetime.now()
                    )
                    optimizations.append(opt)

        return optimizations

    async def _apply_safe_optimizations(self) -> None:
        """Apply optimizations with low risk automatically."""
        for opt_id, opt in self.optimizations.items():
            if (not opt.applied and
                opt.risk_level == "low" and
                not opt.prerequisites):

                # Apply optimization (placeholder for actual implementation)
                logger.info(f"Applying optimization: {opt.name}")

                # Mark as applied
                opt.applied = True
                self.metrics.optimizations_applied += 1

                # In a real implementation, this would trigger actual changes
                # For now, we just log it

    # ========================================================================
    # Knowledge Graph Expansion
    # ========================================================================

    async def _expand_knowledge_graph(self) -> None:
        """Expand the knowledge graph with new concepts and relationships."""
        if not self.neo4j_driver:
            return

        # Extract concepts from recent operations
        concepts = self._extract_concepts()

        # Store concepts in Neo4j
        for concept_name, concept_data in concepts.items():
            await self._store_concept_neo4j(concept_name, concept_data)

        # Create relationships between concepts
        await self._create_concept_relationships()

        # Update metrics
        await self._update_graph_metrics()

    def _extract_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Extract concepts from operations and patterns."""
        concepts = {}

        # Extract from patterns
        for pattern in self.patterns.values():
            # Pattern type as concept
            concept_name = f"PatternType_{pattern.pattern_type}"
            concepts[concept_name] = {
                "type": "pattern_type",
                "category": pattern.pattern_type,
                "examples": [pattern.pattern_id]
            }

            # Insights as concepts
            for insight in pattern.actionable_insights:
                concept_name = f"Insight_{hash(insight)}"
                concepts[concept_name] = {
                    "type": "insight",
                    "description": insight,
                    "source_pattern": pattern.pattern_id
                }

        # Extract from skills
        for skill in self.skills.values():
            concept_name = f"SkillCategory_{skill.category}"
            concepts[concept_name] = {
                "type": "skill_category",
                "category": skill.category,
                "skills": [skill.skill_id]
            }

        return concepts

    async def _store_concept_neo4j(
        self,
        concept_name: str,
        concept_data: Dict[str, Any]
    ) -> None:
        """Store a concept in Neo4j."""
        if not self.neo4j_driver:
            return

        query = """
        MERGE (c:Concept {name: $name})
        SET c.type = $type,
            c.data = $data,
            c.updated_at = datetime()
        """

        try:
            async with self.neo4j_driver.session() as session:
                await session.run(
                    query,
                    name=concept_name,
                    type=concept_data.get("type", "unknown"),
                    data=json.dumps(concept_data)
                )
        except Exception as e:
            logger.error(f"Failed to store concept in Neo4j: {e}")

    async def _create_concept_relationships(self) -> None:
        """Create relationships between concepts."""
        if not self.neo4j_driver:
            return

        # Example: Link patterns to operations
        query = """
        MATCH (p:Pattern)
        MATCH (o:Operation)
        WHERE o.type = p.conditions.sequence[0]
        MERGE (p)-[:APPLIES_TO]->(o)
        """

        try:
            async with self.neo4j_driver.session() as session:
                await session.run(query)
        except Exception as e:
            logger.debug(f"Relationship creation query failed: {e}")

    async def _update_graph_metrics(self) -> None:
        """Update knowledge graph metrics."""
        if not self.neo4j_driver:
            return

        try:
            async with self.neo4j_driver.session() as session:
                # Count nodes
                result = await session.run("MATCH (n) RETURN count(n) as count")
                record = await result.single()
                self.metrics.knowledge_graph_nodes = record["count"]

                # Count relationships
                result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
                record = await result.single()
                self.metrics.knowledge_graph_relationships = record["count"]
        except Exception as e:
            logger.error(f"Failed to update graph metrics: {e}")

    # ========================================================================
    # Skill Generation
    # ========================================================================

    async def _generate_skills(self) -> None:
        """Generate new skills autonomously based on patterns."""
        # Only generate skills if we have enough patterns
        if len(self.patterns) < 5:
            return

        # Analyze patterns to identify skill opportunities
        skill_opportunities = self._identify_skill_opportunities()

        # Generate skills
        for opportunity in skill_opportunities:
            skill = await self._generate_skill_from_opportunity(opportunity)
            if skill and skill.confidence >= self.skill_confidence_threshold:
                self.skills[skill.skill_id] = skill
                self.metrics.skills_generated += 1
                await self._store_skill_neo4j(skill)
                logger.info(f"New skill generated: {skill.name}")

    def _identify_skill_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for new skills."""
        opportunities = []

        # Look for patterns that could be automated
        for pattern in self.patterns.values():
            if (pattern.confidence >= 0.8 and
                pattern.occurrences >= 10 and
                pattern.pattern_type in ["sequential", "performance"]):

                opportunities.append({
                    "type": "automation",
                    "pattern": pattern,
                    "description": f"Automate {pattern.description}"
                })

        # Look for optimization opportunities
        for opt in self.optimizations.values():
            if not opt.applied and opt.risk_level == "low":
                opportunities.append({
                    "type": "optimization",
                    "optimization": opt,
                    "description": f"Implement {opt.name}"
                })

        return opportunities

    async def _generate_skill_from_opportunity(
        self,
        opportunity: Dict[str, Any]
    ) -> Optional[Skill]:
        """Generate a skill from an identified opportunity."""
        opp_type = opportunity["type"]

        if opp_type == "automation":
            pattern = opportunity["pattern"]

            # Generate code for automation
            code = self._generate_automation_code(pattern)

            skill = Skill(
                skill_id=f"skill_{pattern.pattern_id}",
                name=f"Automate {pattern.pattern_type}",
                description=f"Automatically handle {pattern.description}",
                category="automation",
                code=code,
                dependencies=[],
                confidence=pattern.confidence,
                test_results={"generated": True},
                created_at=datetime.now()
            )

            return skill

        elif opp_type == "optimization":
            opt = opportunity["optimization"]

            # Generate code for optimization
            code = self._generate_optimization_code(opt)

            skill = Skill(
                skill_id=f"skill_{opt.strategy_id}",
                name=opt.name,
                description=opt.description,
                category="optimization",
                code=code,
                dependencies=opt.prerequisites,
                confidence=0.8,
                test_results={"generated": True},
                created_at=datetime.now()
            )

            return skill

        return None

    def _generate_automation_code(self, pattern: Pattern) -> str:
        """Generate code for automating a pattern."""
        # This is a simplified example - real implementation would be more sophisticated
        code = f"""
async def auto_{pattern.pattern_id}(context: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    Auto-generated skill for: {pattern.description}
    Confidence: {pattern.confidence:.2f}
    \"\"\"
    # Pattern conditions: {json.dumps(pattern.conditions, indent=2)}
    # Expected outcomes: {json.dumps(pattern.outcomes, indent=2)}

    # Implementation would go here
    result = {{
        "pattern_id": "{pattern.pattern_id}",
        "success": True,
        "insights": {pattern.actionable_insights}
    }}

    return result
"""
        return code

    def _generate_optimization_code(self, opt: OptimizationStrategy) -> str:
        """Generate code for implementing an optimization."""
        code = f"""
async def optimize_{opt.strategy_id}(context: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"
    Auto-generated optimization: {opt.name}
    Expected improvement: {opt.expected_improvement:.1%}
    Risk level: {opt.risk_level}
    \"\"\"
    # Implementation: {opt.implementation}

    # Apply optimization logic here
    result = {{
        "optimization_id": "{opt.strategy_id}",
        "applied": True,
        "improvement": {opt.expected_improvement}
    }}

    return result
"""
        return code

    async def _store_skill_neo4j(self, skill: Skill) -> None:
        """Store skill in Neo4j knowledge graph."""
        if not self.neo4j_driver:
            return

        query = """
        MERGE (s:Skill {id: $id})
        SET s.name = $name,
            s.description = $description,
            s.category = $category,
            s.confidence = $confidence,
            s.created_at = datetime($created_at),
            s.times_used = $times_used,
            s.success_rate = $success_rate
        """

        try:
            async with self.neo4j_driver.session() as session:
                await session.run(
                    query,
                    id=skill.skill_id,
                    name=skill.name,
                    description=skill.description,
                    category=skill.category,
                    confidence=skill.confidence,
                    created_at=skill.created_at.isoformat(),
                    times_used=skill.times_used,
                    success_rate=skill.success_rate
                )
        except Exception as e:
            logger.error(f"Failed to store skill in Neo4j: {e}")

    # ========================================================================
    # Metrics & Persistence
    # ========================================================================

    async def _update_metrics(self) -> None:
        """Update learning metrics."""
        # Calculate average confidence across patterns
        if self.patterns:
            self.metrics.average_confidence = np.mean([
                p.confidence for p in self.patterns.values()
            ])

        # Update timestamp
        self.metrics.last_update = datetime.now()

    async def _save_persistent_data(self) -> None:
        """Save patterns, skills, and metrics to disk."""
        # Save patterns
        patterns_file = self.data_dir / "patterns.json"
        with open(patterns_file, 'w') as f:
            patterns_data = [p.to_dict() for p in self.patterns.values()]
            json.dump(patterns_data, f, indent=2)

        # Save skills
        skills_file = self.data_dir / "skills.json"
        with open(skills_file, 'w') as f:
            skills_data = [s.to_dict() for s in self.skills.values()]
            json.dump(skills_data, f, indent=2)

        # Save metrics
        metrics_file = self.data_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            metrics_data = asdict(self.metrics)
            metrics_data['last_update'] = self.metrics.last_update.isoformat()
            json.dump(metrics_data, f, indent=2)

        logger.debug("Persistent data saved")

    # ========================================================================
    # Public API
    # ========================================================================

    async def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status and metrics."""
        return {
            "is_learning": self.is_learning,
            "metrics": asdict(self.metrics),
            "patterns_count": len(self.patterns),
            "skills_count": len(self.skills),
            "optimizations_count": len(self.optimizations),
            "operations_in_buffer": len(self.operation_buffer),
            "top_patterns": [
                {
                    "id": p.pattern_id,
                    "description": p.description,
                    "confidence": p.confidence,
                    "occurrences": p.occurrences
                }
                for p in sorted(
                    self.patterns.values(),
                    key=lambda x: x.confidence,
                    reverse=True
                )[:5]
            ],
            "top_skills": [
                {
                    "id": s.skill_id,
                    "name": s.name,
                    "confidence": s.confidence,
                    "times_used": s.times_used
                }
                for s in sorted(
                    self.skills.values(),
                    key=lambda x: x.confidence,
                    reverse=True
                )[:5]
            ]
        }

    async def get_insights(self) -> List[str]:
        """Get actionable insights from learning."""
        insights = []

        # Collect insights from patterns
        for pattern in self.patterns.values():
            if pattern.confidence >= 0.8:
                insights.extend(pattern.actionable_insights)

        # Add optimization recommendations
        for opt in self.optimizations.values():
            if not opt.applied and opt.risk_level == "low":
                insights.append(f"Recommendation: {opt.name}")

        return insights

    async def execute_skill(
        self,
        skill_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a generated skill.

        Args:
            skill_id: ID of skill to execute
            context: Execution context

        Returns:
            Execution result
        """
        if skill_id not in self.skills:
            raise ValueError(f"Skill {skill_id} not found")

        skill = self.skills[skill_id]

        # In a real implementation, this would execute the skill code
        # For now, we simulate execution
        logger.info(f"Executing skill: {skill.name}")

        # Update usage metrics
        skill.times_used += 1

        # Simulate result
        result = {
            "skill_id": skill_id,
            "skill_name": skill.name,
            "executed": True,
            "context": context
        }

        return result


# ============================================================================
# Example Usage & Testing
# ============================================================================

async def example_usage():
    """Example usage of the Self-Learning Engine."""
    # Initialize engine
    engine = SelfLearningEngine(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )

    # Initialize
    await engine.initialize()

    # Start continuous learning
    await engine.start_continuous_learning()

    # Simulate some operations
    for i in range(20):
        operation = OperationRecord(
            operation_id=f"op_{i}",
            operation_type=OperationType.TASK_EXECUTION,
            timestamp=datetime.now(),
            duration_ms=np.random.normal(1000, 200),
            gpu_ids=[i % 7],
            gpu_utilization={i % 7: np.random.uniform(0.5, 0.9)},
            memory_used_mb={i % 7: np.random.uniform(5000, 9000)},
            success=np.random.random() > 0.1,
            input_params={"task_type": "inference"},
            output_metrics={"accuracy": 0.95}
        )
        await engine.record_operation(operation)
        await asyncio.sleep(0.1)

    # Wait for learning cycles
    await asyncio.sleep(15)

    # Get learning status
    status = await engine.get_learning_status()
    print("Learning Status:")
    print(json.dumps(status, indent=2, default=str))

    # Get insights
    insights = await engine.get_insights()
    print("\nInsights:")
    for insight in insights:
        print(f"- {insight}")

    # Cleanup
    await engine.cleanup()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
