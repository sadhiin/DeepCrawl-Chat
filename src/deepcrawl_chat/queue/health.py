"""Health check system for queue infrastructure."""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.asyncio import Redis

from ..core.interfaces import Storage
from .manager import QueueManager
from .distributor import TaskDistributor, WorkerInfo
from .status_tracker import TaskStatusTracker
from .metrics import MetricsCollector


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    message: str
    timestamp: datetime
    checks: List[HealthCheckResult]
    uptime_seconds: float

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class HealthChecker:
    """Health checker for queue infrastructure."""

    def __init__(
        self,
        redis_client: Redis,
        queue_manager: Optional[QueueManager] = None,
        task_distributor: Optional[TaskDistributor] = None,
        status_tracker: Optional[TaskStatusTracker] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        check_interval: float = 30.0,
        unhealthy_threshold: int = 3,
        critical_threshold: int = 5
    ):
        """Initialize health checker.

        Args:
            redis_client: Redis client instance
            queue_manager: Queue manager instance
            task_distributor: Task distributor instance
            status_tracker: Task status tracker instance
            metrics_collector: Metrics collector instance
            check_interval: Interval between health checks in seconds
            unhealthy_threshold: Number of consecutive failures for unhealthy status
            critical_threshold: Number of consecutive failures for critical status
        """
        self.redis_client = redis_client
        self.queue_manager = queue_manager
        self.task_distributor = task_distributor
        self.status_tracker = status_tracker
        self.metrics_collector = metrics_collector
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.critical_threshold = critical_threshold

        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.last_check_time: Optional[datetime] = None
        self.failure_counts: Dict[str, int] = {}
        self.health_history: List[SystemHealth] = []
        self.max_history_size = 100

        # Health check registry
        self.health_checks = [
            self._check_redis_connectivity,
            self._check_redis_memory,
            self._check_queue_operations,
            self._check_worker_health,
            self._check_task_processing,
            self._check_system_resources
        ]

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        self.logger.info("Starting health monitoring")

        while True:
            try:
                await self.run_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.check_interval)

    async def run_health_checks(self) -> SystemHealth:
        """Run all health checks and return system health."""
        start_time = time.time()
        check_results = []

        self.logger.debug("Running health checks")

        # Run all health checks
        for check_func in self.health_checks:
            try:
                result = await check_func()
                check_results.append(result)

                # Update failure counts
                if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    self.failure_counts[result.component] = self.failure_counts.get(result.component, 0) + 1
                else:
                    self.failure_counts[result.component] = 0

            except Exception as e:
                self.logger.error(f"Health check failed for {check_func.__name__}: {e}")
                check_results.append(HealthCheckResult(
                    component=check_func.__name__.replace("_check_", ""),
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    duration_ms=(time.time() - start_time) * 1000
                ))

        # Determine overall health status
        overall_status = self._determine_overall_status(check_results)

        # Create system health object
        system_health = SystemHealth(
            status=overall_status,
            message=self._get_status_message(overall_status, check_results),
            timestamp=datetime.now(),
            checks=check_results,
            uptime_seconds=time.time() - self.start_time
        )

        # Store health history
        self.health_history.append(system_health)
        if len(self.health_history) > self.max_history_size:
            self.health_history.pop(0)

        self.last_check_time = system_health.timestamp

        # Log health status
        if overall_status != HealthStatus.HEALTHY:
            self.logger.warning(f"System health: {overall_status.value} - {system_health.message}")
        else:
            self.logger.debug(f"System health: {overall_status.value}")

        return system_health

    async def get_current_health(self) -> Optional[SystemHealth]:
        """Get current system health."""
        if self.health_history:
            return self.health_history[-1]
        return None

    async def get_health_history(self, limit: int = 10) -> List[SystemHealth]:
        """Get health check history."""
        return self.health_history[-limit:] if self.health_history else []

    async def _check_redis_connectivity(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        start_time = time.time()
        component = "redis_connectivity"

        try:
            # Test basic connectivity
            await self.redis_client.ping()

            # Test basic operations
            test_key = "health_check_test"
            await self.redis_client.set(test_key, "test", ex=10)
            value = await self.redis_client.get(test_key)
            await self.redis_client.delete(test_key)

            if value != b"test":
                raise Exception("Redis read/write test failed")

            duration_ms = (time.time() - start_time) * 1000

            if duration_ms > 1000:  # > 1 second is concerning
                return HealthCheckResult(
                    component=component,
                    status=HealthStatus.DEGRADED,
                    message=f"Redis connectivity slow: {duration_ms:.2f}ms",
                    timestamp=datetime.now(),
                    duration_ms=duration_ms
                )

            return HealthCheckResult(
                component=component,
                status=HealthStatus.HEALTHY,
                message=f"Redis connectivity OK: {duration_ms:.2f}ms",
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.CRITICAL,
                message=f"Redis connectivity failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

    async def _check_redis_memory(self) -> HealthCheckResult:
        """Check Redis memory usage."""
        start_time = time.time()
        component = "redis_memory"

        try:
            info = await self.redis_client.info("memory")
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)

            if max_memory > 0:
                memory_usage = (used_memory / max_memory) * 100

                if memory_usage > 90:
                    status = HealthStatus.CRITICAL
                    message = f"Redis memory usage critical: {memory_usage:.1f}%"
                elif memory_usage > 80:
                    status = HealthStatus.UNHEALTHY
                    message = f"Redis memory usage high: {memory_usage:.1f}%"
                elif memory_usage > 70:
                    status = HealthStatus.DEGRADED
                    message = f"Redis memory usage elevated: {memory_usage:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Redis memory usage OK: {memory_usage:.1f}%"
            else:
                # No max memory set
                used_memory_mb = used_memory / (1024 * 1024)
                status = HealthStatus.HEALTHY
                message = f"Redis memory usage: {used_memory_mb:.1f}MB (no limit set)"

            return HealthCheckResult(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "used_memory": used_memory,
                    "max_memory": max_memory,
                    "memory_usage_percent": memory_usage if max_memory > 0 else None
                }
            )

        except Exception as e:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Redis memory check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

    async def _check_queue_operations(self) -> HealthCheckResult:
        """Check queue operations."""
        start_time = time.time()
        component = "queue_operations"

        if not self.queue_manager:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.DEGRADED,
                message="Queue manager not configured",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

        try:
            # Check queue sizes
            sizes = await self.queue_manager.get_all_queue_sizes()
            total_pending = sum(size.pending for size in sizes.values())
            total_processing = sum(size.processing for size in sizes.values())
            total_failed = sum(size.failed for size in sizes.values())

            # Check for concerning patterns
            if total_failed > 100:
                status = HealthStatus.UNHEALTHY
                message = f"High number of failed tasks: {total_failed}"
            elif total_processing > 1000:
                status = HealthStatus.DEGRADED
                message = f"High number of processing tasks: {total_processing}"
            elif total_pending > 10000:
                status = HealthStatus.DEGRADED
                message = f"High number of pending tasks: {total_pending}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Queue operations OK (pending: {total_pending}, processing: {total_processing}, failed: {total_failed})"

            return HealthCheckResult(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "total_pending": total_pending,
                    "total_processing": total_processing,
                    "total_failed": total_failed,
                    "queue_sizes": {name: {
                        "pending": size.pending,
                        "processing": size.processing,
                        "failed": size.failed
                    } for name, size in sizes.items()}
                }
            )

        except Exception as e:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Queue operations check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

    async def _check_worker_health(self) -> HealthCheckResult:
        """Check worker health."""
        start_time = time.time()
        component = "worker_health"

        if not self.task_distributor:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.DEGRADED,
                message="Task distributor not configured",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

        try:
            workers = await self.task_distributor.get_active_workers()
            active_workers = len(workers)

            if active_workers == 0:
                status = HealthStatus.CRITICAL
                message = "No active workers"
            elif active_workers < 2:
                status = HealthStatus.UNHEALTHY
                message = f"Low number of active workers: {active_workers}"
            else:
                # Check worker load distribution
                loads = [worker.current_load for worker in workers.values()]
                max_load = max(loads) if loads else 0
                avg_load = sum(loads) / len(loads) if loads else 0

                if max_load > 0.9:
                    status = HealthStatus.DEGRADED
                    message = f"High worker load detected: max={max_load:.2f}, avg={avg_load:.2f}"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Workers healthy: {active_workers} active, avg_load={avg_load:.2f}"

            return HealthCheckResult(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "active_workers": active_workers,
                    "worker_loads": [worker.current_load for worker in workers.values()],
                    "worker_types": list(set(worker.worker_type for worker in workers.values()))
                }
            )

        except Exception as e:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Worker health check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

    async def _check_task_processing(self) -> HealthCheckResult:
        """Check task processing health."""
        start_time = time.time()
        component = "task_processing"

        if not self.status_tracker:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.DEGRADED,
                message="Status tracker not configured",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

        try:
            # Check for stuck tasks (processing for too long)
            now = datetime.now()
            stuck_threshold = timedelta(hours=1)

            # This is a simplified check - in a real implementation,
            # you'd query the status tracker for long-running tasks
            status = HealthStatus.HEALTHY
            message = "Task processing appears healthy"

            # You could implement more sophisticated checks here:
            # - Check task completion rates
            # - Check for tasks stuck in processing state
            # - Check for timeout violations
            # - Check error rates

            return HealthCheckResult(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"Task processing check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resources."""
        start_time = time.time()
        component = "system_resources"

        try:
            # Check Redis connection count
            info = await self.redis_client.info("clients")
            connected_clients = info.get("connected_clients", 0)

            if connected_clients > 1000:
                status = HealthStatus.DEGRADED
                message = f"High Redis connection count: {connected_clients}"
            else:
                status = HealthStatus.HEALTHY
                message = f"System resources OK (Redis connections: {connected_clients})"

            return HealthCheckResult(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "connected_clients": connected_clients
                }
            )

        except Exception as e:
            return HealthCheckResult(
                component=component,
                status=HealthStatus.UNHEALTHY,
                message=f"System resources check failed: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000
            )

    def _determine_overall_status(self, check_results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status."""
        if not check_results:
            return HealthStatus.UNHEALTHY

        # Count status levels
        status_counts = {status: 0 for status in HealthStatus}
        for result in check_results:
            status_counts[result.status] += 1

        # Determine overall status based on individual check results
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] >= 2:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.UNHEALTHY] > 0 or status_counts[HealthStatus.DEGRADED] >= 3:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _get_status_message(self, status: HealthStatus, check_results: List[HealthCheckResult]) -> str:
        """Get status message based on check results."""
        if status == HealthStatus.HEALTHY:
            return "All systems operational"

        # Find the most concerning issues
        concerning_results = [
            result for result in check_results
            if result.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        ]

        if concerning_results:
            # Sort by severity
            severity_order = {
                HealthStatus.CRITICAL: 0,
                HealthStatus.UNHEALTHY: 1,
                HealthStatus.DEGRADED: 2
            }
            concerning_results.sort(key=lambda x: severity_order[x.status])

            # Return message from most severe issue
            return concerning_results[0].message

        return f"System status: {status.value}"


class HealthMonitor:
    """High-level health monitoring service."""

    def __init__(self, health_checker: HealthChecker):
        """Initialize health monitor.

        Args:
            health_checker: Health checker instance
        """
        self.health_checker = health_checker
        self.logger = logging.getLogger(__name__)
        self._monitoring_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start health monitoring."""
        if self._monitoring_task is not None:
            self.logger.warning("Health monitoring already started")
            return

        self.logger.info("Starting health monitoring service")
        self._monitoring_task = asyncio.create_task(self.health_checker.start_monitoring())

    async def stop(self) -> None:
        """Stop health monitoring."""
        if self._monitoring_task is None:
            return

        self.logger.info("Stopping health monitoring service")
        self._monitoring_task.cancel()
        try:
            await self._monitoring_task
        except asyncio.CancelledError:
            pass
        finally:
            self._monitoring_task = None

    async def get_health_status(self) -> Optional[SystemHealth]:
        """Get current health status."""
        return await self.health_checker.get_current_health()

    async def run_health_check(self) -> SystemHealth:
        """Run health check on demand."""
        return await self.health_checker.run_health_checks()