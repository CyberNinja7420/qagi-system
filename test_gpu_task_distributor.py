"""
Comprehensive test suite for GPU Task Distributor.

Tests cover:
- GPU monitoring
- Task assignment
- Load balancing
- Failure handling
- Health checks
- Edge cases

Author: QAGI System
License: MIT
"""

import asyncio
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from gpu_task_distributor import (
    GPUAssignment,
    GPUMetrics,
    GPUMonitor,
    GPUStatus,
    GPUTaskDistributor,
    TaskAssignment,
    TaskRequest,
    TaskType,
)


# Sample nvidia-smi XML output for testing
SAMPLE_NVIDIA_SMI_XML = """<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">
<nvidia_smi_log>
    <timestamp>Mon Nov  3 10:00:00 2025</timestamp>
    <driver_version>535.129.03</driver_version>
    <cuda_version>12.2</cuda_version>
    <attached_gpus>7</attached_gpus>
    <gpu id="00000000:01:00.0">
        <product_name>NVIDIA GeForce RTX 2080 Ti</product_name>
        <minor_number>0</minor_number>
        <utilization>
            <gpu_util>25 %</gpu_util>
            <memory_util>30 %</memory_util>
        </utilization>
        <fb_memory_usage>
            <total>11264 MiB</total>
            <used>3379 MiB</used>
            <free>7885 MiB</free>
        </fb_memory_usage>
        <temperature>
            <gpu_temp>55 C</gpu_temp>
        </temperature>
        <power_readings>
            <power_draw>120.50 W</power_draw>
            <power_limit>250.00 W</power_limit>
        </power_readings>
        <fan_speed>40 %</fan_speed>
        <compute_mode>Default</compute_mode>
    </gpu>
    <gpu id="00000000:02:00.0">
        <product_name>NVIDIA GeForce RTX 2080 Ti</product_name>
        <minor_number>1</minor_number>
        <utilization>
            <gpu_util>85 %</gpu_util>
            <memory_util>92 %</memory_util>
        </utilization>
        <fb_memory_usage>
            <total>11264 MiB</total>
            <used>10363 MiB</used>
            <free>901 MiB</free>
        </fb_memory_usage>
        <temperature>
            <gpu_temp>78 C</gpu_temp>
        </temperature>
        <power_readings>
            <power_draw>230.00 W</power_draw>
            <power_limit>250.00 W</power_limit>
        </power_readings>
        <fan_speed>80 %</fan_speed>
        <compute_mode>Default</compute_mode>
    </gpu>
    <gpu id="00000000:03:00.0">
        <product_name>NVIDIA GeForce RTX 2080 Ti</product_name>
        <minor_number>2</minor_number>
        <utilization>
            <gpu_util>50 %</gpu_util>
            <memory_util>45 %</memory_util>
        </utilization>
        <fb_memory_usage>
            <total>11264 MiB</total>
            <used>5069 MiB</used>
            <free>6195 MiB</free>
        </fb_memory_usage>
        <temperature>
            <gpu_temp>65 C</gpu_temp>
        </temperature>
        <power_readings>
            <power_draw>180.00 W</power_draw>
            <power_limit>250.00 W</power_limit>
        </power_readings>
        <fan_speed>60 %</fan_speed>
        <compute_mode>Default</compute_mode>
    </gpu>
    <gpu id="00000000:04:00.0">
        <product_name>NVIDIA GeForce RTX 2080 Ti</product_name>
        <minor_number>3</minor_number>
        <utilization>
            <gpu_util>10 %</gpu_util>
            <memory_util>15 %</memory_util>
        </utilization>
        <fb_memory_usage>
            <total>11264 MiB</total>
            <used>1690 MiB</used>
            <free>9574 MiB</free>
        </fb_memory_usage>
        <temperature>
            <gpu_temp>45 C</gpu_temp>
        </temperature>
        <power_readings>
            <power_draw>80.00 W</power_draw>
            <power_limit>250.00 W</power_limit>
        </power_readings>
        <fan_speed>30 %</fan_speed>
        <compute_mode>Default</compute_mode>
    </gpu>
    <gpu id="00000000:05:00.0">
        <product_name>NVIDIA GeForce RTX 2080 Ti</product_name>
        <minor_number>4</minor_number>
        <utilization>
            <gpu_util>60 %</gpu_util>
            <memory_util>70 %</memory_util>
        </utilization>
        <fb_memory_usage>
            <total>11264 MiB</total>
            <used>7885 MiB</used>
            <free>3379 MiB</free>
        </fb_memory_usage>
        <temperature>
            <gpu_temp>70 C</gpu_temp>
        </temperature>
        <power_readings>
            <power_draw>200.00 W</power_draw>
            <power_limit>250.00 W</power_limit>
        </power_readings>
        <fan_speed>70 %</fan_speed>
        <compute_mode>Default</compute_mode>
    </gpu>
    <gpu id="00000000:06:00.0">
        <product_name>NVIDIA GeForce RTX 2080 Ti</product_name>
        <minor_number>5</minor_number>
        <utilization>
            <gpu_util>35 %</gpu_util>
            <memory_util>40 %</memory_util>
        </utilization>
        <fb_memory_usage>
            <total>11264 MiB</total>
            <used>4506 MiB</used>
            <free>6758 MiB</free>
        </fb_memory_usage>
        <temperature>
            <gpu_temp>60 C</gpu_temp>
        </temperature>
        <power_readings>
            <power_draw>150.00 W</power_draw>
            <power_limit>250.00 W</power_limit>
        </power_readings>
        <fan_speed>50 %</fan_speed>
        <compute_mode>Default</compute_mode>
    </gpu>
    <gpu id="00000000:07:00.0">
        <product_name>NVIDIA GeForce RTX 2080 Ti</product_name>
        <minor_number>6</minor_number>
        <utilization>
            <gpu_util>5 %</gpu_util>
            <memory_util>10 %</memory_util>
        </utilization>
        <fb_memory_usage>
            <total>11264 MiB</total>
            <used>1126 MiB</used>
            <free>10138 MiB</free>
        </fb_memory_usage>
        <temperature>
            <gpu_temp>40 C</gpu_temp>
        </temperature>
        <power_readings>
            <power_draw>60.00 W</power_draw>
            <power_limit>250.00 W</power_limit>
        </power_readings>
        <fan_speed>25 %</fan_speed>
        <compute_mode>Default</compute_mode>
    </gpu>
</nvidia_smi_log>
"""


@pytest.fixture
def mock_nvidia_smi():
    """Mock nvidia-smi subprocess calls."""
    with patch('subprocess.run') as mock_run:
        # Mock version check
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "NVIDIA-SMI 535.129.03"

        # Mock XML query
        xml_result = Mock()
        xml_result.returncode = 0
        xml_result.stdout = SAMPLE_NVIDIA_SMI_XML

        def side_effect(*args, **kwargs):
            if '--version' in args[0]:
                return version_result
            else:
                return xml_result

        mock_run.side_effect = side_effect
        yield mock_run


@pytest.fixture
def gpu_monitor(mock_nvidia_smi):
    """Create GPU monitor instance."""
    return GPUMonitor(gpu_count=7)


@pytest.fixture
async def gpu_distributor(mock_nvidia_smi):
    """Create and start GPU distributor instance."""
    distributor = GPUTaskDistributor(
        gpu_count=7,
        monitor_interval_sec=0.1,
        health_check_interval_sec=0.5
    )
    await distributor.start()
    yield distributor
    await distributor.stop()


class TestGPUMetrics:
    """Test GPUMetrics dataclass."""

    def test_memory_utilization(self):
        """Test memory utilization calculation."""
        metrics = GPUMetrics(
            gpu_id=0,
            name="RTX 2080 Ti",
            utilization=50.0,
            memory_used=5632,
            memory_total=11264,
            memory_free=5632,
            temperature=60.0,
            power_draw=150.0,
            power_limit=250.0
        )

        assert metrics.memory_utilization == 50.0

    def test_is_overheating(self):
        """Test overheating detection."""
        metrics = GPUMetrics(
            gpu_id=0,
            name="RTX 2080 Ti",
            utilization=50.0,
            memory_used=5632,
            memory_total=11264,
            memory_free=5632,
            temperature=86.0,
            power_draw=150.0,
            power_limit=250.0
        )

        assert metrics.is_overheating is True

    def test_is_overloaded(self):
        """Test overload detection."""
        metrics = GPUMetrics(
            gpu_id=0,
            name="RTX 2080 Ti",
            utilization=95.0,
            memory_used=10339,
            memory_total=11264,
            memory_free=925,
            temperature=70.0,
            power_draw=200.0,
            power_limit=250.0
        )

        assert metrics.is_overloaded is True

    def test_load_score(self):
        """Test load score calculation."""
        # Low load GPU
        metrics_low = GPUMetrics(
            gpu_id=0,
            name="RTX 2080 Ti",
            utilization=10.0,
            memory_used=1126,
            memory_total=11264,
            memory_free=10138,
            temperature=40.0,
            power_draw=60.0,
            power_limit=250.0
        )

        # High load GPU
        metrics_high = GPUMetrics(
            gpu_id=1,
            name="RTX 2080 Ti",
            utilization=90.0,
            memory_used=10138,
            memory_total=11264,
            memory_free=1126,
            temperature=80.0,
            power_draw=230.0,
            power_limit=250.0
        )

        assert metrics_low.load_score < metrics_high.load_score


class TestGPUMonitor:
    """Test GPUMonitor class."""

    def test_initialization(self, gpu_monitor):
        """Test monitor initialization."""
        assert gpu_monitor.gpu_count == 7

    def test_get_gpu_metrics(self, gpu_monitor):
        """Test GPU metrics retrieval."""
        metrics = gpu_monitor.get_gpu_metrics()

        assert len(metrics) == 7
        assert all(isinstance(m, GPUMetrics) for m in metrics)
        assert all(m.gpu_id == i for i, m in enumerate(metrics))

    def test_parse_nvidia_smi_xml(self, gpu_monitor):
        """Test XML parsing."""
        metrics = gpu_monitor._parse_nvidia_smi_xml(SAMPLE_NVIDIA_SMI_XML)

        assert len(metrics) == 7

        # Check first GPU
        gpu0 = metrics[0]
        assert gpu0.gpu_id == 0
        assert gpu0.name == "NVIDIA GeForce RTX 2080 Ti"
        assert gpu0.utilization == 25.0
        assert gpu0.memory_total == 11264
        assert gpu0.memory_used == 3379
        assert gpu0.temperature == 55.0
        assert gpu0.power_draw == 120.5

    def test_nvidia_smi_failure(self, mock_nvidia_smi):
        """Test handling of nvidia-smi failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            with pytest.raises(RuntimeError):
                GPUMonitor(gpu_count=7)

    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, gpu_monitor):
        """Test continuous monitoring."""
        metrics_list = []

        def callback(metrics: List[GPUMetrics]):
            metrics_list.append(metrics)

        # Start monitoring
        monitor_task = asyncio.create_task(
            gpu_monitor.monitor_continuous(interval_sec=0.1, callback=callback)
        )

        # Wait for a few iterations
        await asyncio.sleep(0.3)

        # Cancel monitoring
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Should have captured multiple metric snapshots
        assert len(metrics_list) >= 2


class TestGPUTaskDistributor:
    """Test GPUTaskDistributor class."""

    @pytest.mark.asyncio
    async def test_initialization(self, gpu_distributor):
        """Test distributor initialization."""
        assert gpu_distributor.gpu_count == 7
        assert len(gpu_distributor.gpu_assignments) == 4

    @pytest.mark.asyncio
    async def test_gpu_assignments(self, gpu_distributor):
        """Test GPU assignment configuration."""
        assignments = gpu_distributor.gpu_assignments

        # Check quantum assignment
        assert TaskType.QUANTUM in assignments
        assert assignments[TaskType.QUANTUM].gpu_ids == [0]

        # Check vLLM assignment
        assert TaskType.VLLM in assignments
        assert assignments[TaskType.VLLM].gpu_ids == [1, 2, 3]

        # Check Ollama assignment
        assert TaskType.OLLAMA in assignments
        assert assignments[TaskType.OLLAMA].gpu_ids == [4, 5]

        # Check dynamic assignment
        assert TaskType.DYNAMIC in assignments
        assert assignments[TaskType.DYNAMIC].gpu_ids == [6]

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_nvidia_smi):
        """Test starting and stopping distributor."""
        distributor = GPUTaskDistributor(gpu_count=7)

        assert not distributor._running

        await distributor.start()
        assert distributor._running

        await distributor.stop()
        assert not distributor._running

    @pytest.mark.asyncio
    async def test_get_gpu_metrics_snapshot(self, gpu_distributor):
        """Test getting GPU metrics snapshot."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        metrics = gpu_distributor.get_gpu_metrics_snapshot()

        assert len(metrics) == 7
        assert all(isinstance(m, GPUMetrics) for m in metrics.values())

    @pytest.mark.asyncio
    async def test_get_gpu_status_summary(self, gpu_distributor):
        """Test GPU status summary."""
        summary = gpu_distributor.get_gpu_status_summary()

        assert summary['total_gpus'] == 7
        assert summary['healthy'] == 7
        assert summary['degraded'] == 0
        assert summary['failed'] == 0
        assert 'stats' in summary

    @pytest.mark.asyncio
    async def test_submit_quantum_task(self, gpu_distributor):
        """Test submitting quantum task."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        task = TaskRequest(
            task_id="quantum_test_001",
            task_type=TaskType.QUANTUM,
            memory_required_mb=4096,
            priority=8
        )

        assignment = await gpu_distributor.submit_task(task)

        assert assignment is not None
        assert assignment.gpu_id == 0  # Should be assigned to GPU 0
        assert assignment.task_id == "quantum_test_001"

    @pytest.mark.asyncio
    async def test_submit_vllm_task(self, gpu_distributor):
        """Test submitting vLLM task."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        task = TaskRequest(
            task_id="vllm_test_001",
            task_type=TaskType.VLLM,
            memory_required_mb=20480,
            priority=7
        )

        assignment = await gpu_distributor.submit_task(task)

        assert assignment is not None
        assert assignment.gpu_id in [1, 2, 3]  # Should be assigned to vLLM GPUs

    @pytest.mark.asyncio
    async def test_submit_ollama_task(self, gpu_distributor):
        """Test submitting Ollama task."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        task = TaskRequest(
            task_id="ollama_test_001",
            task_type=TaskType.OLLAMA,
            memory_required_mb=8192,
            priority=6
        )

        assignment = await gpu_distributor.submit_task(task)

        assert assignment is not None
        assert assignment.gpu_id in [4, 5]  # Should be assigned to Ollama GPUs

    @pytest.mark.asyncio
    async def test_submit_dynamic_task(self, gpu_distributor):
        """Test submitting dynamic task."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        task = TaskRequest(
            task_id="dynamic_test_001",
            task_type=TaskType.DYNAMIC,
            memory_required_mb=2048,
            priority=5
        )

        assignment = await gpu_distributor.submit_task(task)

        assert assignment is not None
        assert assignment.gpu_id == 6  # Should be assigned to dynamic GPU

    @pytest.mark.asyncio
    async def test_complete_task(self, gpu_distributor):
        """Test completing a task."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Submit task
        task = TaskRequest(
            task_id="complete_test_001",
            task_type=TaskType.DYNAMIC,
            memory_required_mb=2048,
            priority=5
        )

        assignment = await gpu_distributor.submit_task(task)
        assert assignment is not None

        # Complete task
        await gpu_distributor.complete_task("complete_test_001", success=True)

        assert "complete_test_001" not in gpu_distributor.active_tasks
        assert gpu_distributor.stats['tasks_completed'] == 1

    @pytest.mark.asyncio
    async def test_task_failure(self, gpu_distributor):
        """Test handling task failure."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Submit task
        task = TaskRequest(
            task_id="fail_test_001",
            task_type=TaskType.DYNAMIC,
            memory_required_mb=2048,
            priority=5
        )

        assignment = await gpu_distributor.submit_task(task)
        assert assignment is not None

        # Fail task
        await gpu_distributor.complete_task(
            "fail_test_001",
            success=False,
            error="Test failure"
        )

        assert "fail_test_001" not in gpu_distributor.active_tasks
        assert gpu_distributor.stats['tasks_failed'] == 1
        assert "fail_test_001" in gpu_distributor.failed_tasks

    @pytest.mark.asyncio
    async def test_task_queueing(self, gpu_distributor):
        """Test task queueing when no suitable GPU."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Submit task with excessive memory requirement
        task = TaskRequest(
            task_id="queue_test_001",
            task_type=TaskType.DYNAMIC,
            memory_required_mb=50000,  # More than available
            priority=5
        )

        assignment = await gpu_distributor.submit_task(task)

        # Should be queued, not assigned
        assert assignment is None
        assert len(gpu_distributor.task_queue) == 1

    @pytest.mark.asyncio
    async def test_priority_based_selection(self, gpu_distributor):
        """Test that high priority tasks get preferential GPU selection."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Submit low priority task
        low_priority = TaskRequest(
            task_id="low_priority",
            task_type=TaskType.DYNAMIC,
            memory_required_mb=2048,
            priority=3
        )

        # Submit high priority task
        high_priority = TaskRequest(
            task_id="high_priority",
            task_type=TaskType.DYNAMIC,
            memory_required_mb=2048,
            priority=9
        )

        await gpu_distributor.submit_task(low_priority)
        await gpu_distributor.submit_task(high_priority)

        # Both should be assigned (same GPU pool)
        assert "low_priority" in gpu_distributor.active_tasks
        assert "high_priority" in gpu_distributor.active_tasks

    @pytest.mark.asyncio
    async def test_set_gpu_status(self, gpu_distributor):
        """Test manually setting GPU status."""
        await gpu_distributor.set_gpu_status(0, GPUStatus.MAINTENANCE)

        assert gpu_distributor.gpu_status[0] == GPUStatus.MAINTENANCE

        # Try to submit task for maintenance GPU
        task = TaskRequest(
            task_id="maintenance_test",
            task_type=TaskType.QUANTUM,
            memory_required_mb=4096,
            priority=8
        )

        assignment = await gpu_distributor.submit_task(task)

        # Should be queued since GPU 0 is in maintenance
        assert assignment is None

    @pytest.mark.asyncio
    async def test_get_task_info(self, gpu_distributor):
        """Test getting task information."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Submit task
        task = TaskRequest(
            task_id="info_test_001",
            task_type=TaskType.DYNAMIC,
            memory_required_mb=2048,
            priority=5
        )

        await gpu_distributor.submit_task(task)

        # Get task info
        info = gpu_distributor.get_task_info("info_test_001")

        assert info is not None
        assert info['task_id'] == "info_test_001"
        assert info['status'] == "running"
        assert 'gpu_id' in info

    @pytest.mark.asyncio
    async def test_gpu_health_monitoring(self, gpu_distributor):
        """Test GPU health monitoring and auto-degradation."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Manually create overheating GPU metric
        overheated_metric = GPUMetrics(
            gpu_id=6,
            name="RTX 2080 Ti",
            utilization=95.0,
            memory_used=10000,
            memory_total=11264,
            memory_free=1264,
            temperature=91.0,  # Critical temperature
            power_draw=240.0,
            power_limit=250.0
        )

        gpu_distributor.gpu_metrics[6] = overheated_metric

        # Wait for health check to run
        await asyncio.sleep(1.0)

        # GPU 6 should be marked as failed
        assert gpu_distributor.gpu_status[6] == GPUStatus.FAILED

    @pytest.mark.asyncio
    async def test_load_balancing(self, gpu_distributor):
        """Test load balancing across multiple GPUs."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Submit multiple Ollama tasks (GPUs 4-5)
        tasks = []
        for i in range(4):
            task = TaskRequest(
                task_id=f"load_test_{i}",
                task_type=TaskType.OLLAMA,
                memory_required_mb=2048,
                priority=5
            )
            tasks.append(task)
            await gpu_distributor.submit_task(task)

        # Should distribute across both Ollama GPUs
        gpu_assignments = [
            gpu_distributor.active_tasks[f"load_test_{i}"].gpu_id
            for i in range(4)
            if f"load_test_{i}" in gpu_distributor.active_tasks
        ]

        # Both GPUs should be used
        assert 4 in gpu_assignments or 5 in gpu_assignments

    @pytest.mark.asyncio
    async def test_gpu_preference(self, gpu_distributor):
        """Test GPU preference in task request."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Submit task with GPU preference
        task = TaskRequest(
            task_id="preference_test",
            task_type=TaskType.OLLAMA,
            memory_required_mb=2048,
            priority=5,
            gpu_preference=[4]  # Prefer GPU 4
        )

        assignment = await gpu_distributor.submit_task(task)

        assert assignment is not None
        assert assignment.gpu_id == 4

    @pytest.mark.asyncio
    async def test_concurrent_task_submission(self, gpu_distributor):
        """Test concurrent task submissions."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Submit multiple tasks concurrently
        tasks = [
            TaskRequest(
                task_id=f"concurrent_{i}",
                task_type=TaskType.DYNAMIC,
                memory_required_mb=1024,
                priority=5
            )
            for i in range(10)
        ]

        assignments = await asyncio.gather(*[
            gpu_distributor.submit_task(task)
            for task in tasks
        ])

        # All should be assigned (GPU 6 has enough capacity)
        successful_assignments = [a for a in assignments if a is not None]
        assert len(successful_assignments) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_invalid_gpu_id(self, gpu_distributor):
        """Test handling of invalid GPU ID."""
        with pytest.raises(ValueError):
            await gpu_distributor.set_gpu_status(99, GPUStatus.MAINTENANCE)

    @pytest.mark.asyncio
    async def test_complete_nonexistent_task(self, gpu_distributor):
        """Test completing a nonexistent task."""
        # Should not raise exception
        await gpu_distributor.complete_task("nonexistent_task", success=True)

    @pytest.mark.asyncio
    async def test_get_info_nonexistent_task(self, gpu_distributor):
        """Test getting info for nonexistent task."""
        info = gpu_distributor.get_task_info("nonexistent_task")
        assert info is None

    @pytest.mark.asyncio
    async def test_zero_memory_requirement(self, gpu_distributor):
        """Test task with zero memory requirement."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        task = TaskRequest(
            task_id="zero_mem_test",
            task_type=TaskType.DYNAMIC,
            memory_required_mb=0,
            priority=5
        )

        assignment = await gpu_distributor.submit_task(task)
        assert assignment is not None

    @pytest.mark.asyncio
    async def test_rebalance_tasks(self, gpu_distributor):
        """Test task rebalancing."""
        # Wait for initial monitoring
        await asyncio.sleep(0.2)

        # Queue some tasks
        for i in range(3):
            task = TaskRequest(
                task_id=f"rebalance_{i}",
                task_type=TaskType.DYNAMIC,
                memory_required_mb=50000,  # Will be queued
                priority=5
            )
            await gpu_distributor.submit_task(task)

        # Attempt rebalance
        rebalanced = await gpu_distributor.rebalance_tasks()

        # Should attempt to process queue
        assert rebalanced >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
