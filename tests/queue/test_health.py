"""Tests for health check system."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from deepcrawl_chat.queue.health import (
    HealthChecker,
    HealthMonitor,
    HealthStatus,
    HealthCheckResult,
    SystemHealth
)
from deepcrawl_chat.queue.manager import QueueManager
from deepcrawl_chat.queue.distributor import TaskDistributor, WorkerInfo
from deepcrawl_chat.queue.status_tracker import TaskStatusTracker
from deepcrawl_chat.queue.metrics import MetricsCollector


@pytest.fixture
async def mock_redis():
    """Create mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True
    redis_mock.set.return_value = True
    redis_mock.get.return_value = b"test"
    redis_mock.delete.return_value = 1
    redis_mock.info.return_value = {
        "used_memory": 1024 * 1024,  # 1MB
        "maxmemory": 10 * 1024 * 1024,  # 10MB
        "connected_clients": 5
    }
    return redis_mock


@pytest.fixture
async def mock_queue_manager():
    """Create mock queue manager."""
    manager = AsyncMock(spec=QueueManager)
    manager.get_all_queue_sizes.return_value = {
        "crawl": MagicMock(pending=10, processing=2, failed=0),
        "process": MagicMock(pending=5, processing=1, failed=1)
    }
    return manager


@pytest.fixture
async def mock_task_distributor():
    """Create mock task distributor."""
    distributor = AsyncMock(spec=TaskDistributor)
    distributor.get_active_workers.return_value = {
        "worker1": WorkerInfo(
            worker_id="worker1",
            worker_type="crawler",
            capabilities={"crawl"},
            current_load=0.5,
            last_heartbeat=datetime.now(),
            max_concurrent_tasks=10
        ),
        "worker2": WorkerInfo(
            worker_id="worker2",
            worker_type="processor",
            capabilities={"process"},
            current_load=0.3,
            last_heartbeat=datetime.now(),
            max_concurrent_tasks=5
        )
    }
    return distributor


@pytest.fixture
async def mock_status_tracker():
    """Create mock status tracker."""
    return AsyncMock(spec=TaskStatusTracker)


@pytest.fixture
async def mock_metrics_collector():
    """Create mock metrics collector."""
    return AsyncMock(spec=MetricsCollector)


@pytest.fixture
async def health_checker(
    mock_redis,
    mock_queue_manager,
    mock_task_distributor,
    mock_status_tracker,
    mock_metrics_collector
):
    """Create health checker instance."""
    return HealthChecker(
        redis_client=mock_redis,
        queue_manager=mock_queue_manager,
        task_distributor=mock_task_distributor,
        status_tracker=mock_status_tracker,
        metrics_collector=mock_metrics_collector,
        check_interval=1.0  # Fast interval for testing
    )


class TestHealthChecker:
    """Test HealthChecker class."""

    async def test_redis_connectivity_healthy(self, health_checker, mock_redis):
        """Test healthy Redis connectivity check."""
        result = await health_checker._check_redis_connectivity()

        assert result.component == "redis_connectivity"
        assert result.status == HealthStatus.HEALTHY
        assert "Redis connectivity OK" in result.message
        assert result.duration_ms > 0

        # Verify Redis operations were called
        mock_redis.ping.assert_called_once()
        mock_redis.set.assert_called_once()
        mock_redis.get.assert_called_once()
        mock_redis.delete.assert_called_once()

    async def test_redis_connectivity_slow(self, health_checker, mock_redis):
        """Test slow Redis connectivity check."""
        # Mock slow operations
        async def slow_ping():
            await asyncio.sleep(1.1)  # Simulate slow operation
            return True

        mock_redis.ping.side_effect = slow_ping

        result = await health_checker._check_redis_connectivity()

        assert result.component == "redis_connectivity"
        assert result.status == HealthStatus.DEGRADED
        assert "Redis connectivity slow" in result.message
        assert result.duration_ms > 1000

    async def test_redis_connectivity_failed(self, health_checker, mock_redis):
        """Test failed Redis connectivity check."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        result = await health_checker._check_redis_connectivity()

        assert result.component == "redis_connectivity"
        assert result.status == HealthStatus.CRITICAL
        assert "Redis connectivity failed" in result.message

    async def test_redis_memory_healthy(self, health_checker):
        """Test healthy Redis memory check."""
        result = await health_checker._check_redis_memory()

        assert result.component == "redis_memory"
        assert result.status == HealthStatus.HEALTHY
        assert "Redis memory usage OK" in result.message
        assert result.details["memory_usage_percent"] == 10.0  # 1MB / 10MB * 100

    async def test_redis_memory_critical(self, health_checker, mock_redis):
        """Test critical Redis memory check."""
        mock_redis.info.return_value = {
            "used_memory": 9.5 * 1024 * 1024,  # 9.5MB
            "maxmemory": 10 * 1024 * 1024,  # 10MB
        }

        result = await health_checker._check_redis_memory()

        assert result.component == "redis_memory"
        assert result.status == HealthStatus.CRITICAL
        assert "Redis memory usage critical" in result.message

    async def test_redis_memory_no_limit(self, health_checker, mock_redis):
        """Test Redis memory check with no limit set."""
        mock_redis.info.return_value = {
            "used_memory": 1024 * 1024,  # 1MB
            "maxmemory": 0,  # No limit
        }

        result = await health_checker._check_redis_memory()

        assert result.component == "redis_memory"
        assert result.status == HealthStatus.HEALTHY
        assert "no limit set" in result.message

    async def test_queue_operations_healthy(self, health_checker):
        """Test healthy queue operations check."""
        result = await health_checker._check_queue_operations()

        assert result.component == "queue_operations"
        assert result.status == HealthStatus.HEALTHY
        assert "Queue operations OK" in result.message
        assert result.details["total_pending"] == 15
        assert result.details["total_processing"] == 3
        assert result.details["total_failed"] == 1

    async def test_queue_operations_high_failed(self, health_checker, mock_queue_manager):
        """Test queue operations with high failed count."""
        mock_queue_manager.get_all_queue_sizes.return_value = {
            "crawl": MagicMock(pending=10, processing=2, failed=150)
        }

        result = await health_checker._check_queue_operations()

        assert result.component == "queue_operations"
        assert result.status == HealthStatus.UNHEALTHY
        assert "High number of failed tasks" in result.message

    async def test_queue_operations_no_manager(self, mock_redis):
        """Test queue operations check without queue manager."""
        checker = HealthChecker(redis_client=mock_redis)

        result = await checker._check_queue_operations()

        assert result.component == "queue_operations"
        assert result.status == HealthStatus.DEGRADED
        assert "Queue manager not configured" in result.message

    async def test_worker_health_healthy(self, health_checker):
        """Test healthy worker health check."""
        result = await health_checker._check_worker_health()

        assert result.component == "worker_health"
        assert result.status == HealthStatus.HEALTHY
        assert "Workers healthy" in result.message
        assert result.details["active_workers"] == 2

    async def test_worker_health_no_workers(self, health_checker, mock_task_distributor):
        """Test worker health with no active workers."""
        mock_task_distributor.get_active_workers.return_value = {}

        result = await health_checker._check_worker_health()

        assert result.component == "worker_health"
        assert result.status == HealthStatus.CRITICAL
        assert "No active workers" in result.message

    async def test_worker_health_high_load(self, health_checker, mock_task_distributor):
        """Test worker health with high load."""
        mock_task_distributor.get_active_workers.return_value = {
            "worker1": WorkerInfo(
                worker_id="worker1",
                worker_type="crawler",
                capabilities={"crawl"},
                current_load=0.95,  # High load
                last_heartbeat=datetime.now(),
                max_concurrent_tasks=10
            ),
            "worker2": WorkerInfo(
                worker_id="worker2",
                worker_type="processor",
                capabilities={"process"},
                current_load=0.8,
                last_heartbeat=datetime.now(),
                max_concurrent_tasks=5
            )
        }

        result = await health_checker._check_worker_health()

        assert result.component == "worker_health"
        assert result.status == HealthStatus.DEGRADED
        assert "High worker load detected" in result.message

    async def test_task_processing_healthy(self, health_checker):
        """Test healthy task processing check."""
        result = await health_checker._check_task_processing()

        assert result.component == "task_processing"
        assert result.status == HealthStatus.HEALTHY
        assert "Task processing appears healthy" in result.message

    async def test_task_processing_no_tracker(self, mock_redis):
        """Test task processing check without status tracker."""
        checker = HealthChecker(redis_client=mock_redis)

        result = await checker._check_task_processing()

        assert result.component == "task_processing"
        assert result.status == HealthStatus.DEGRADED
        assert "Status tracker not configured" in result.message

    async def test_system_resources_healthy(self, health_checker):
        """Test healthy system resources check."""
        result = await health_checker._check_system_resources()

        assert result.component == "system_resources"
        assert result.status == HealthStatus.HEALTHY
        assert "System resources OK" in result.message
        assert result.details["connected_clients"] == 5

    async def test_system_resources_high_connections(self, health_checker, mock_redis):
        """Test system resources with high connection count."""
        mock_redis.info.return_value = {"connected_clients": 1500}

        result = await health_checker._check_system_resources()

        assert result.component == "system_resources"
        assert result.status == HealthStatus.DEGRADED
        assert "High Redis connection count" in result.message

    async def test_run_health_checks(self, health_checker):
        """Test running all health checks."""
        system_health = await health_checker.run_health_checks()

        assert isinstance(system_health, SystemHealth)
        assert system_health.status == HealthStatus.HEALTHY
        assert len(system_health.checks) == 6  # Number of health checks
        assert system_health.uptime_seconds > 0
        assert system_health.is_healthy

    async def test_health_check_failure_tracking(self, health_checker, mock_redis):
        """Test failure count tracking."""
        # Simulate Redis failures
        mock_redis.ping.side_effect = Exception("Connection failed")

        # Run multiple health checks
        for _ in range(3):
            await health_checker.run_health_checks()

        # Check failure count
        assert health_checker.failure_counts.get("redis_connectivity", 0) == 3

    async def test_health_history_tracking(self, health_checker):
        """Test health history tracking."""
        # Run multiple health checks
        for _ in range(5):
            await health_checker.run_health_checks()

        # Check history
        assert len(health_checker.health_history) == 5

        # Get health history
        history = await health_checker.get_health_history(3)
        assert len(history) == 3

    async def test_health_history_limit(self, health_checker):
        """Test health history size limit."""
        # Set small history limit
        health_checker.max_history_size = 3

        # Run more checks than the limit
        for _ in range(5):
            await health_checker.run_health_checks()

        # Check history is limited
        assert len(health_checker.health_history) == 3

    async def test_overall_status_determination(self, health_checker):
        """Test overall status determination logic."""
        # Test with healthy results
        healthy_results = [
            HealthCheckResult("test1", HealthStatus.HEALTHY, "OK", datetime.now(), 10),
            HealthCheckResult("test2", HealthStatus.HEALTHY, "OK", datetime.now(), 15)
        ]
        assert health_checker._determine_overall_status(healthy_results) == HealthStatus.HEALTHY

        # Test with critical result
        critical_results = [
            HealthCheckResult("test1", HealthStatus.HEALTHY, "OK", datetime.now(), 10),
            HealthCheckResult("test2", HealthStatus.CRITICAL, "Failed", datetime.now(), 15)
        ]
        assert health_checker._determine_overall_status(critical_results) == HealthStatus.CRITICAL

        # Test with multiple unhealthy results
        unhealthy_results = [
            HealthCheckResult("test1", HealthStatus.UNHEALTHY, "Bad", datetime.now(), 10),
            HealthCheckResult("test2", HealthStatus.UNHEALTHY, "Bad", datetime.now(), 15)
        ]
        assert health_checker._determine_overall_status(unhealthy_results) == HealthStatus.UNHEALTHY

    async def test_get_current_health(self, health_checker):
        """Test getting current health status."""
        # No health checks run yet
        current = await health_checker.get_current_health()
        assert current is None

        # Run health check
        await health_checker.run_health_checks()

        # Get current health
        current = await health_checker.get_current_health()
        assert current is not None
        assert isinstance(current, SystemHealth)


class TestHealthMonitor:
    """Test HealthMonitor class."""

    @pytest.fixture
    async def health_monitor(self, health_checker):
        """Create health monitor instance."""
        return HealthMonitor(health_checker)

    async def test_start_stop_monitoring(self, health_monitor):
        """Test starting and stopping health monitoring."""
        # Start monitoring
        await health_monitor.start()
        assert health_monitor._monitoring_task is not None

        # Stop monitoring
        await health_monitor.stop()
        assert health_monitor._monitoring_task is None

    async def test_double_start_warning(self, health_monitor):
        """Test warning on double start."""
        await health_monitor.start()

        with patch.object(health_monitor.logger, 'warning') as mock_warning:
            await health_monitor.start()
            mock_warning.assert_called_once()

        await health_monitor.stop()

    async def test_get_health_status(self, health_monitor):
        """Test getting health status through monitor."""
        # No health checks run yet
        status = await health_monitor.get_health_status()
        assert status is None

        # Run health check
        await health_monitor.run_health_check()

        # Get status
        status = await health_monitor.get_health_status()
        assert status is not None
        assert isinstance(status, SystemHealth)

    async def test_run_health_check_on_demand(self, health_monitor):
        """Test running health check on demand."""
        result = await health_monitor.run_health_check()

        assert isinstance(result, SystemHealth)
        assert result.status in [status for status in HealthStatus]


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_health_check_result_creation(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            component="test",
            status=HealthStatus.HEALTHY,
            message="All good",
            timestamp=datetime.now(),
            duration_ms=100.0,
            details={"key": "value"}
        )

        assert result.component == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.duration_ms == 100.0
        assert result.details == {"key": "value"}


class TestSystemHealth:
    """Test SystemHealth dataclass."""

    def test_system_health_creation(self):
        """Test creating system health."""
        checks = [
            HealthCheckResult("test", HealthStatus.HEALTHY, "OK", datetime.now(), 10)
        ]

        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            message="All systems operational",
            timestamp=datetime.now(),
            checks=checks,
            uptime_seconds=3600.0
        )

        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All systems operational"
        assert len(health.checks) == 1
        assert health.uptime_seconds == 3600.0
        assert health.is_healthy

    def test_system_health_unhealthy(self):
        """Test unhealthy system health."""
        checks = [
            HealthCheckResult("test", HealthStatus.CRITICAL, "Failed", datetime.now(), 10)
        ]

        health = SystemHealth(
            status=HealthStatus.CRITICAL,
            message="System critical",
            timestamp=datetime.now(),
            checks=checks,
            uptime_seconds=3600.0
        )

        assert health.status == HealthStatus.CRITICAL
        assert not health.is_healthy

    def test_system_health_is_healthy(self):
        """Test system health is_healthy property."""
        checks = [
            HealthCheckResult("test", HealthStatus.HEALTHY, "OK", datetime.now(), 10)
        ]

        health = SystemHealth(
            status=HealthStatus.HEALTHY,
            message="All systems operational",
            timestamp=datetime.now(),
            checks=checks,
            uptime_seconds=3600.0
        )

        assert health.is_healthy

        # Test degraded is still considered healthy
        health.status = HealthStatus.DEGRADED
        assert health.is_healthy

        # Test unhealthy is not healthy
        health.status = HealthStatus.UNHEALTHY
        assert not health.is_healthy


@pytest.mark.integration
class TestHealthIntegration:
    """Integration tests for health system."""

    async def test_full_health_check_cycle(self, mock_redis):
        """Test full health check cycle with real components."""
        # Create health checker with minimal components
        checker = HealthChecker(redis_client=mock_redis, check_interval=0.1)
        monitor = HealthMonitor(checker)

        try:
            # Start monitoring
            await monitor.start()

            # Wait for a check to complete
            await asyncio.sleep(0.2)

            # Verify health status
            status = await monitor.get_health_status()
            assert status is not None
            assert isinstance(status, SystemHealth)

            # Run manual check
            manual_status = await monitor.run_health_check()
            assert isinstance(manual_status, SystemHealth)

        finally:
            # Stop monitoring
            await monitor.stop()

    async def test_health_check_with_failures(self, mock_redis):
        """Test health checks with component failures."""
        # Configure Redis to fail
        mock_redis.ping.side_effect = Exception("Redis down")

        checker = HealthChecker(redis_client=mock_redis)

        # Run health check
        health = await checker.run_health_checks()

        # Verify degraded/critical status
        assert health.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]

        # Check that Redis connectivity failed
        redis_check = next(
            (check for check in health.checks if check.component == "redis_connectivity"),
            None
        )
        assert redis_check is not None
        assert redis_check.status == HealthStatus.CRITICAL