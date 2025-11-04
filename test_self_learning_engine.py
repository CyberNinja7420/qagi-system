"""
Test Suite for QAGI Self-Learning Engine
=========================================

Comprehensive tests for the Self-Learning Engine including:
- Operation recording and analysis
- Pattern recognition
- Performance optimization
- Knowledge graph integration
- Skill generation
- Continuous learning loop

Author: QAGI System
Date: 2025-11-03
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import tempfile
import shutil

import numpy as np

from self_learning_engine import (
    SelfLearningEngine,
    OperationRecord,
    OperationType,
    Pattern,
    Skill,
    OptimizationStrategy,
    LearningMetrics
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def temp_data_dir():
    """Create temporary data directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def learning_engine(temp_data_dir):
    """Create a learning engine instance for testing."""
    engine = SelfLearningEngine(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        operation_buffer_size=1000,
        pattern_min_occurrences=3,
        pattern_confidence_threshold=0.7,
        skill_confidence_threshold=0.8,
        learning_cycle_interval_seconds=1.0,
        data_dir=temp_data_dir
    )

    # Initialize without Neo4j for unit tests
    # In integration tests, Neo4j would be required
    try:
        await engine.initialize()
    except Exception:
        # Skip Neo4j initialization for unit tests
        pass

    yield engine

    await engine.cleanup()


@pytest.fixture
def sample_operations() -> List[OperationRecord]:
    """Generate sample operations for testing."""
    operations = []

    # Create diverse operations
    operation_types = list(OperationType)

    for i in range(50):
        op = OperationRecord(
            operation_id=f"test_op_{i}",
            operation_type=operation_types[i % len(operation_types)],
            timestamp=datetime.now() - timedelta(minutes=50-i),
            duration_ms=np.random.normal(1000, 200),
            gpu_ids=[i % 7],
            gpu_utilization={i % 7: np.random.uniform(0.4, 0.95)},
            memory_used_mb={i % 7: np.random.uniform(4000, 10000)},
            success=np.random.random() > 0.15,  # 85% success rate
            input_params={
                "task_id": f"task_{i}",
                "complexity": np.random.choice(["low", "medium", "high"])
            },
            output_metrics={
                "accuracy": np.random.uniform(0.8, 0.99),
                "throughput": np.random.uniform(100, 1000)
            }
        )
        operations.append(op)

    return operations


# ============================================================================
# Unit Tests - Operation Recording
# ============================================================================

@pytest.mark.asyncio
async def test_record_operation(learning_engine, sample_operations):
    """Test recording operations."""
    operation = sample_operations[0]

    await learning_engine.record_operation(operation)

    assert len(learning_engine.operation_buffer) == 1
    assert learning_engine.metrics.total_operations_analyzed == 1


@pytest.mark.asyncio
async def test_record_multiple_operations(learning_engine, sample_operations):
    """Test recording multiple operations."""
    for op in sample_operations:
        await learning_engine.record_operation(op)

    assert len(learning_engine.operation_buffer) == len(sample_operations)
    assert learning_engine.metrics.total_operations_analyzed == len(sample_operations)


@pytest.mark.asyncio
async def test_operation_buffer_size_limit(learning_engine):
    """Test that operation buffer respects size limit."""
    buffer_size = learning_engine.operation_buffer_size

    # Record more operations than buffer size
    for i in range(buffer_size + 100):
        op = OperationRecord(
            operation_id=f"op_{i}",
            operation_type=OperationType.TASK_EXECUTION,
            timestamp=datetime.now(),
            duration_ms=1000,
            gpu_ids=[0],
            gpu_utilization={0: 0.8},
            memory_used_mb={0: 5000},
            success=True
        )
        await learning_engine.record_operation(op)

    # Buffer should not exceed max size
    assert len(learning_engine.operation_buffer) == buffer_size


# ============================================================================
# Unit Tests - Operation Analysis
# ============================================================================

@pytest.mark.asyncio
async def test_calculate_success_rates(learning_engine, sample_operations):
    """Test success rate calculation."""
    for op in sample_operations:
        await learning_engine.record_operation(op)

    success_rates = learning_engine._calculate_success_rates(
        list(learning_engine.operation_buffer)
    )

    # Should have success rates for each operation type
    assert isinstance(success_rates, dict)
    assert all(0 <= rate <= 1 for rate in success_rates.values())


@pytest.mark.asyncio
async def test_performance_trends(learning_engine, sample_operations):
    """Test performance trend calculation."""
    for op in sample_operations:
        await learning_engine.record_operation(op)

    trends = learning_engine._calculate_performance_trends(
        list(learning_engine.operation_buffer)
    )

    # Should have trends for each operation type
    assert isinstance(trends, dict)

    for op_type, stats in trends.items():
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'trend' in stats
        assert stats['trend'] in ['improving', 'degrading', 'stable', 'insufficient_data']


@pytest.mark.asyncio
async def test_identify_anomalies(learning_engine, sample_operations):
    """Test anomaly detection."""
    # Add normal operations
    for op in sample_operations[:40]:
        await learning_engine.record_operation(op)

    # Add anomalous operations (very slow)
    for i in range(5):
        anomalous_op = OperationRecord(
            operation_id=f"anomaly_{i}",
            operation_type=OperationType.TASK_EXECUTION,
            timestamp=datetime.now(),
            duration_ms=50000,  # Very slow
            gpu_ids=[0],
            gpu_utilization={0: 0.8},
            memory_used_mb={0: 5000},
            success=True
        )
        await learning_engine.record_operation(op)

    anomalies = learning_engine._identify_anomalies(
        list(learning_engine.operation_buffer)
    )

    # Should detect some anomalies
    assert isinstance(anomalies, list)


# ============================================================================
# Unit Tests - Pattern Recognition
# ============================================================================

@pytest.mark.asyncio
async def test_recognize_sequential_patterns(learning_engine):
    """Test sequential pattern recognition."""
    # Create operations with clear sequence pattern
    sequence = [
        OperationType.TASK_EXECUTION,
        OperationType.MODEL_INFERENCE,
        OperationType.GPU_COMPUTATION
    ]

    # Repeat sequence multiple times
    for repeat in range(10):
        for i, op_type in enumerate(sequence):
            op = OperationRecord(
                operation_id=f"seq_{repeat}_{i}",
                operation_type=op_type,
                timestamp=datetime.now(),
                duration_ms=1000,
                gpu_ids=[0],
                gpu_utilization={0: 0.8},
                memory_used_mb={0: 5000},
                success=True
            )
            await learning_engine.record_operation(op)

    # Recognize patterns
    patterns = learning_engine._recognize_sequential_patterns(
        list(learning_engine.operation_buffer)
    )

    # Should find the repeated sequence
    assert len(patterns) > 0


@pytest.mark.asyncio
async def test_recognize_correlation_patterns(learning_engine):
    """Test correlation pattern recognition."""
    # Create operations with correlation between GPU util and duration
    for i in range(50):
        gpu_util = 0.5 + i * 0.01  # Increasing utilization
        duration = 500 + i * 20     # Increasing duration (correlated)

        op = OperationRecord(
            operation_id=f"corr_{i}",
            operation_type=OperationType.GPU_COMPUTATION,
            timestamp=datetime.now(),
            duration_ms=duration,
            gpu_ids=[0],
            gpu_utilization={0: gpu_util},
            memory_used_mb={0: 5000},
            success=True
        )
        await learning_engine.record_operation(op)

    patterns = learning_engine._recognize_correlation_patterns(
        list(learning_engine.operation_buffer)
    )

    # Should find correlation
    assert isinstance(patterns, list)


@pytest.mark.asyncio
async def test_recognize_failure_patterns(learning_engine):
    """Test failure pattern recognition."""
    # Create failed operations with common characteristic (high memory)
    for i in range(20):
        op = OperationRecord(
            operation_id=f"fail_{i}",
            operation_type=OperationType.GPU_COMPUTATION,
            timestamp=datetime.now(),
            duration_ms=1000,
            gpu_ids=[0],
            gpu_utilization={0: 0.8},
            memory_used_mb={0: 9500},  # High memory
            success=False,
            error_message="OOM"
        )
        await learning_engine.record_operation(op)

    patterns = learning_engine._recognize_failure_patterns(
        list(learning_engine.operation_buffer)
    )

    # Should identify high memory as failure pattern
    assert len(patterns) > 0


# ============================================================================
# Unit Tests - Performance Optimization
# ============================================================================

@pytest.mark.asyncio
async def test_analyze_current_performance(learning_engine, sample_operations):
    """Test current performance analysis."""
    for op in sample_operations:
        await learning_engine.record_operation(op)

    analysis = learning_engine._analyze_current_performance()

    # Should have performance metrics
    assert isinstance(analysis, dict)
    assert len(analysis) > 0


@pytest.mark.asyncio
async def test_generate_optimizations(learning_engine):
    """Test optimization generation."""
    # Create scenario with underutilized GPU
    for i in range(50):
        op = OperationRecord(
            operation_id=f"underutil_{i}",
            operation_type=OperationType.TASK_EXECUTION,
            timestamp=datetime.now(),
            duration_ms=1000,
            gpu_ids=[3],
            gpu_utilization={3: 0.2},  # Low utilization
            memory_used_mb={3: 2000},
            success=True
        )
        await learning_engine.record_operation(op)

    analysis = learning_engine._analyze_current_performance()
    optimizations = learning_engine._generate_optimizations(analysis)

    # Should generate optimization for underutilized GPU
    assert isinstance(optimizations, list)


# ============================================================================
# Unit Tests - Skill Generation
# ============================================================================

@pytest.mark.asyncio
async def test_identify_skill_opportunities(learning_engine):
    """Test skill opportunity identification."""
    # Add high-confidence pattern
    pattern = Pattern(
        pattern_id="test_pattern",
        pattern_type="sequential",
        description="Test pattern",
        confidence=0.9,
        occurrences=15,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        conditions={"test": True},
        outcomes={"success": True},
        actionable_insights=["Test insight"]
    )
    learning_engine.patterns[pattern.pattern_id] = pattern

    opportunities = learning_engine._identify_skill_opportunities()

    # Should identify opportunity from pattern
    assert len(opportunities) > 0


@pytest.mark.asyncio
async def test_generate_skill_from_opportunity(learning_engine):
    """Test skill generation from opportunity."""
    pattern = Pattern(
        pattern_id="test_pattern",
        pattern_type="sequential",
        description="Test sequential pattern",
        confidence=0.9,
        occurrences=15,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        conditions={"sequence": ["A", "B", "C"]},
        outcomes={"success": True},
        actionable_insights=["Automate this sequence"]
    )

    opportunity = {
        "type": "automation",
        "pattern": pattern,
        "description": "Test opportunity"
    }

    skill = await learning_engine._generate_skill_from_opportunity(opportunity)

    # Should generate skill
    assert skill is not None
    assert isinstance(skill, Skill)
    assert skill.category == "automation"
    assert len(skill.code) > 0


@pytest.mark.asyncio
async def test_execute_skill(learning_engine):
    """Test skill execution."""
    # Create and add a skill
    skill = Skill(
        skill_id="test_skill",
        name="Test Skill",
        description="Test skill for testing",
        category="test",
        code="# Test code",
        dependencies=[],
        confidence=0.9,
        test_results={"passed": True},
        created_at=datetime.now()
    )
    learning_engine.skills[skill.skill_id] = skill

    # Execute skill
    result = await learning_engine.execute_skill(
        skill_id="test_skill",
        context={"test": True}
    )

    # Should execute successfully
    assert result["executed"] is True
    assert skill.times_used == 1


# ============================================================================
# Unit Tests - Persistence
# ============================================================================

@pytest.mark.asyncio
async def test_save_and_load_patterns(learning_engine):
    """Test pattern persistence."""
    # Add pattern
    pattern = Pattern(
        pattern_id="persist_test",
        pattern_type="test",
        description="Test persistence",
        confidence=0.85,
        occurrences=10,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        conditions={"test": True},
        outcomes={"result": "success"},
        actionable_insights=["Test insight"]
    )
    learning_engine.patterns[pattern.pattern_id] = pattern

    # Save
    await learning_engine._save_persistent_data()

    # Verify file exists
    patterns_file = learning_engine.data_dir / "patterns.json"
    assert patterns_file.exists()

    # Load in new engine
    new_engine = SelfLearningEngine(data_dir=learning_engine.data_dir)
    await new_engine._load_persistent_data()

    # Verify pattern loaded
    assert pattern.pattern_id in new_engine.patterns
    loaded_pattern = new_engine.patterns[pattern.pattern_id]
    assert loaded_pattern.description == pattern.description
    assert loaded_pattern.confidence == pattern.confidence


@pytest.mark.asyncio
async def test_save_and_load_skills(learning_engine):
    """Test skill persistence."""
    # Add skill
    skill = Skill(
        skill_id="persist_skill",
        name="Persistent Skill",
        description="Test skill persistence",
        category="test",
        code="# Persistent code",
        dependencies=[],
        confidence=0.9,
        test_results={"passed": True},
        created_at=datetime.now()
    )
    learning_engine.skills[skill.skill_id] = skill

    # Save
    await learning_engine._save_persistent_data()

    # Verify file exists
    skills_file = learning_engine.data_dir / "skills.json"
    assert skills_file.exists()

    # Load in new engine
    new_engine = SelfLearningEngine(data_dir=learning_engine.data_dir)
    await new_engine._load_persistent_data()

    # Verify skill loaded
    assert skill.skill_id in new_engine.skills
    loaded_skill = new_engine.skills[skill.skill_id]
    assert loaded_skill.name == skill.name
    assert loaded_skill.confidence == skill.confidence


# ============================================================================
# Integration Tests - Continuous Learning
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.integration
async def test_continuous_learning_loop(learning_engine, sample_operations):
    """Test continuous learning loop."""
    # Start learning
    await learning_engine.start_continuous_learning()

    # Add operations while learning
    for op in sample_operations:
        await learning_engine.record_operation(op)
        await asyncio.sleep(0.05)

    # Let it learn for a few cycles
    await asyncio.sleep(5)

    # Stop learning
    await learning_engine.stop_learning()

    # Verify learning occurred
    assert learning_engine.metrics.total_operations_analyzed > 0

    # Check if patterns were identified
    # (may or may not find patterns depending on data)
    assert isinstance(learning_engine.patterns, dict)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_learning_status(learning_engine, sample_operations):
    """Test learning status retrieval."""
    # Add some data
    for op in sample_operations[:10]:
        await learning_engine.record_operation(op)

    status = await learning_engine.get_learning_status()

    # Verify status structure
    assert "is_learning" in status
    assert "metrics" in status
    assert "patterns_count" in status
    assert "skills_count" in status
    assert "optimizations_count" in status
    assert "operations_in_buffer" in status
    assert "top_patterns" in status
    assert "top_skills" in status


@pytest.mark.asyncio
@pytest.mark.integration
async def test_get_insights(learning_engine):
    """Test insights retrieval."""
    # Add pattern with insights
    pattern = Pattern(
        pattern_id="insight_test",
        pattern_type="test",
        description="Test pattern",
        confidence=0.85,
        occurrences=10,
        first_seen=datetime.now(),
        last_seen=datetime.now(),
        conditions={},
        outcomes={},
        actionable_insights=["Insight 1", "Insight 2"]
    )
    learning_engine.patterns[pattern.pattern_id] = pattern

    insights = await learning_engine.get_insights()

    # Should return insights
    assert isinstance(insights, list)
    assert "Insight 1" in insights
    assert "Insight 2" in insights


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.performance
async def test_operation_recording_performance(learning_engine):
    """Test operation recording performance."""
    import time

    num_operations = 1000
    operations = []

    # Generate operations
    for i in range(num_operations):
        op = OperationRecord(
            operation_id=f"perf_{i}",
            operation_type=OperationType.TASK_EXECUTION,
            timestamp=datetime.now(),
            duration_ms=1000,
            gpu_ids=[i % 7],
            gpu_utilization={i % 7: 0.8},
            memory_used_mb={i % 7: 5000},
            success=True
        )
        operations.append(op)

    # Measure recording time
    start = time.time()
    for op in operations:
        await learning_engine.record_operation(op)
    elapsed = time.time() - start

    # Should be fast (< 1 second for 1000 operations)
    assert elapsed < 1.0

    ops_per_second = num_operations / elapsed
    print(f"\nOperation recording rate: {ops_per_second:.0f} ops/sec")


@pytest.mark.asyncio
@pytest.mark.performance
async def test_pattern_recognition_performance(learning_engine, sample_operations):
    """Test pattern recognition performance."""
    import time

    # Record operations
    for op in sample_operations * 10:  # 500 operations
        await learning_engine.record_operation(op)

    # Measure pattern recognition time
    start = time.time()
    await learning_engine._recognize_patterns()
    elapsed = time.time() - start

    # Should be reasonable (< 5 seconds)
    assert elapsed < 5.0

    print(f"\nPattern recognition time: {elapsed:.2f} seconds")
    print(f"Patterns found: {len(learning_engine.patterns)}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
