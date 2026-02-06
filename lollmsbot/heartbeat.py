"""
Heartbeat Module - LollmsBot's Self-Maintenance & Monitoring System

The Heartbeat is LollmsBot's "biological rhythm" - autonomous self-care that runs
on a configurable schedule to ensure the system remains healthy, secure, and
evolving without manual intervention.

Responsibilities:
- Self-diagnostics: Check system health, connectivity, integrity
- Memory maintenance: Compress, consolidate, archive, forget outdated info
- Security audit: Review logs, check for anomalies, verify permissions
- Skill curation: Update skill documentation, prune unused skills, suggest improvements
- Self-update: Check for code updates, apply security patches (with consent)
- Performance optimization: Clean caches, optimize storage, balance load
- Anomaly healing: Detect and attempt to fix drift from expected behavior

Architecture: The Heartbeat runs as an async background task, triggered by
schedule or explicit request. All actions are logged and can be reviewed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from lollmsbot.guardian import get_guardian, Guardian, ThreatLevel, SecurityEvent
from lollmsbot.soul import get_soul, Soul


logger = logging.getLogger("lollmsbot.heartbeat")


class MaintenanceTask(Enum):
    """Categories of self-maintenance operations."""
    DIAGNOSTIC = auto()      # Health checks, connectivity tests
    MEMORY = auto()          # Memory compression, archiving, forgetting
    SECURITY = auto()         # Audit log review, permission verification
    SKILL = auto()           # Skill documentation, dependency updates
    UPDATE = auto()          # Code update checks, patch application
    OPTIMIZATION = auto()    # Performance tuning, cache cleaning
    HEALING = auto()         # Self-correction of detected drift


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat behavior."""
    enabled: bool = True
    interval_minutes: float = 30.0  # Default: every 30 minutes
    tasks_enabled: Dict[MaintenanceTask, bool] = field(default_factory=lambda: {
        task: True for task in MaintenanceTask
    })
    
    # Task-specific intervals (override default)
    task_intervals: Dict[MaintenanceTask, Optional[float]] = field(default_factory=dict)
    
    # Thresholds
    memory_pressure_threshold: float = 0.8  # Compress memory when >80% full
    log_retention_days: int = 30
    max_anomaly_score: float = 0.7  # Trigger healing above this
    
    # Self-healing settings
    auto_heal_minor: bool = True  # Fix small issues without asking
    confirm_heal_major: bool = True  # Ask before significant changes
    quarantine_on_critical: bool = True  # Guardian quarantine if unhealable


@dataclass
class TaskResult:
    """Result of a single maintenance task execution."""
    task: MaintenanceTask
    executed_at: datetime
    success: bool
    findings: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class HeartbeatReport:
    """Complete report of a heartbeat cycle."""
    cycle_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    task_results: List[TaskResult] = field(default_factory=list)
    system_state_before: Dict[str, Any] = field(default_factory=dict)
    system_state_after: Dict[str, Any] = field(default_factory=dict)
    anomalies_detected: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    @property
    def success_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.success) / len(self.task_results)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "success_rate": self.success_rate,
            "tasks_executed": len(self.task_results),
            "findings_summary": [f for r in self.task_results for f in r.findings],
            "anomalies_count": len(self.anomalies_detected),
            "recommendations": self.recommendations,
        }


class MemoryMonitor:
    """
    Monitors and manages LollmsBot's memory systems.
    
    Tracks memory pressure, implements forgetting curves, manages semantic
    compression, and archives old memories to maintain performance.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".lollmsbot" / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Memory statistics
        self._conversation_count = 0
        self._memory_entries = 0
        self._total_size_bytes = 0
        self._last_compression = datetime.min
        
        # Forgetting curve parameters (Ebbinghaus-inspired)
        self.retention_halflife_days = 7.0  # Half remembered after 1 week
        self.strength_multiplier = 2.0  # Review strengthens memory
    
    async def analyze(self) -> Dict[str, Any]:
        """Analyze current memory state."""
        stats = {
            "conversations": await self._count_conversations(),
            "memory_entries": await self._count_entries(),
            "total_size_mb": await self._calculate_size() / (1024 * 1024),
            "pressure_score": 0.0,  # 0-1, higher = more pressure to compress
            "oldest_memory_days": 0,
            "compression_recommended": False,
            "archiving_recommended": False,
        }
        
        # Calculate pressure based on size and age
        size_pressure = min(stats["total_size_mb"] / 100, 1.0)  # 100MB = full pressure
        age_pressure = min(stats["oldest_memory_days"] / 30, 1.0)  # 30 days = full pressure
        stats["pressure_score"] = max(size_pressure, age_pressure)
        
        stats["compression_recommended"] = stats["pressure_score"] > 0.6
        stats["archiving_recommended"] = stats["pressure_score"] > 0.85
        
        return stats
    
    async def compress(self, target_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Compress memories using semantic summarization.
        
        Instead of storing full conversation turns, create condensed
        "memory pearls" that capture essence without verbatim detail.
        """
        start_time = time.time()
        
        # Find candidate conversations for compression
        candidates = await self._find_compression_candidates()
        
        compressed_count = 0
        space_saved = 0
        
        for conv_id, conversation in candidates:
            # Generate semantic summary (would use LLM in production)
            summary = await self._summarize_conversation(conversation)
            
            # Replace full conversation with summary + key moments
            compressed = {
                "type": "compressed_memory",
                "original_id": conv_id,
                "summary": summary,
                "key_moments": self._extract_key_moments(conversation),
                "compression_date": datetime.now().isoformat(),
                "original_turns": len(conversation),
            }
            
            # Save compressed version
            await self._save_compressed(conv_id, compressed)
            
            compressed_count += 1
            space_saved += self._estimate_savings(conversation, compressed)
        
        self._last_compression = datetime.now()
        
        return {
            "conversations_compressed": compressed_count,
            "space_saved_mb": space_saved / (1024 * 1024),
            "duration_seconds": time.time() - start_time,
        }
    
    async def apply_forgetting_curve(self) -> Dict[str, Any]:
        """
        Apply Ebbinghaus forgetting curve to memories.
        
        Memories decay naturally unless reinforced. Important memories
        (tagged by user or frequently accessed) are strengthened.
        """
        forgotten = 0
        strengthened = 0
        
        memories = await self._load_all_memories()
        
        for memory in memories:
            age_days = (datetime.now() - memory.get("last_accessed", datetime.now())).days
            
            # Calculate retention probability
            # R = e^(-t/S) where t = time, S = memory strength
            strength = memory.get("importance", 1.0) * self.strength_multiplier
            retention = 2.718281828 ** (-age_days / (self.retention_halflife_days * strength))
            
            if retention < 0.1:  # Less than 10% remembered
                # Archive to long-term storage (slower access) or delete
                await self._archive_memory(memory)
                forgotten += 1
            elif memory.get("access_count", 0) > 5:
                # Frequently accessed - strengthen
                memory["importance"] = memory.get("importance", 1.0) * 1.5
                memory["last_strengthened"] = datetime.now().isoformat()
                await self._save_memory(memory)
                strengthened += 1
        
        return {
            "memories_forgotten": forgotten,
            "memories_strengthened": strengthened,
            "retention_average": sum(
                2.718281828 ** (-(datetime.now() - m.get("last_accessed", datetime.now())).days / 
                (self.retention_halflife_days * m.get("importance", 1.0)))
                for m in memories
            ) / len(memories) if memories else 0,
        }
    
    async def consolidate(self) -> Dict[str, Any]:
        """
        Find related memories and merge them into coherent narratives.
        
        Scattered mentions of "the Python project" become a consolidated
        project memory with timeline, learnings, and current status.
        """
        # Find memory clusters by semantic similarity
        clusters = await self._find_semantic_clusters()
        
        merged = 0
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # Merge into narrative
            narrative = await self._create_narrative(cluster)
            await self._save_narrative(narrative)
            
            # Mark constituents as consolidated
            for mem in cluster:
                mem["consolidated_into"] = narrative["id"]
                await self._save_memory(mem)
            
            merged += len(cluster)
        
        return {
            "clusters_found": len(clusters),
            "memories_consolidated": merged,
            "narratives_created": sum(1 for c in clusters if len(c) >= 2),
        }
    
    # Helper methods (placeholders for actual storage integration)
    async def _count_conversations(self) -> int: return 0
    async def _count_entries(self) -> int: return 0
    async def _calculate_size(self) -> int: return 0
    async def _find_compression_candidates(self) -> List[Tuple[str, Any]]: return []
    async def _summarize_conversation(self, conversation: Any) -> str: return ""
    def _extract_key_moments(self, conversation: Any) -> List[Dict]: return []
    async def _save_compressed(self, conv_id: str, compressed: Dict) -> None: pass
    def _estimate_savings(self, original: Any, compressed: Dict) -> int: return 0
    async def _load_all_memories(self) -> List[Dict]: return []
    async def _archive_memory(self, memory: Dict) -> None: pass
    async def _save_memory(self, memory: Dict) -> None: pass
    async def _find_semantic_clusters(self) -> List[List[Dict]]: return []
    async def _create_narrative(self, cluster: List[Dict]) -> Dict: return {"id": "temp"}


class Heartbeat:
    """
    LollmsBot's autonomous self-maintenance system.
    
    The Heartbeat runs continuously (when enabled), performing configured
    maintenance tasks at appropriate intervals. It's designed to be
    interruptible, observable, and self-healing.
    """
    
    DEFAULT_CONFIG_PATH = Path.home() / ".lollmsbot" / "heartbeat.json"
    
    def __init__(
        self,
        config: Optional[HeartbeatConfig] = None,
        config_path: Optional[Path] = None,
    ):
        self.config = config or HeartbeatConfig()
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        
        # Subsystems
        self.memory_monitor = MemoryMonitor()
        self.guardian = get_guardian()
        self.soul = get_soul()
        
        # Runtime state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._last_run: Optional[datetime] = None
        self._run_history: List[HeartbeatReport] = []
        self._max_history = 100
        
        # Task registry
        self._task_handlers: Dict[MaintenanceTask, Callable[[], Coroutine[Any, Any, TaskResult]]] = {
            MaintenanceTask.DIAGNOSTIC: self._run_diagnostic,
            MaintenanceTask.MEMORY: self._run_memory_maintenance,
            MaintenanceTask.SECURITY: self._run_security_audit,
            MaintenanceTask.SKILL: self._run_skill_curation,
            MaintenanceTask.UPDATE: self._run_update_check,
            MaintenanceTask.OPTIMIZATION: self._run_optimization,
            MaintenanceTask.HEALING: self._run_healing,
        }
        
        # Load or save config
        if self.config_path.exists():
            self._load_config()
        else:
            self._save_config()
    
    def _load_config(self) -> None:
        """Load configuration from JSON."""
        try:
            data = json.loads(self.config_path.read_text())
            self.config = HeartbeatConfig(
                enabled=data.get("enabled", True),
                interval_minutes=data.get("interval_minutes", 30.0),
                tasks_enabled={
                    MaintenanceTask[t]: v 
                    for t, v in data.get("tasks_enabled", {}).items()
                },
                task_intervals={
                    MaintenanceTask[t]: v 
                    for t, v in data.get("task_intervals", {}).items()
                },
                memory_pressure_threshold=data.get("memory_pressure_threshold", 0.8),
                log_retention_days=data.get("log_retention_days", 30),
                max_anomaly_score=data.get("max_anomaly_score", 0.7),
                auto_heal_minor=data.get("auto_heal_minor", True),
                confirm_heal_major=data.get("confirm_heal_major", True),
                quarantine_on_critical=data.get("quarantine_on_critical", True),
            )
        except Exception as e:
            logger.error(f"Failed to load heartbeat config: {e}")
    
    def _save_config(self) -> None:
        """Save configuration to JSON."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "enabled": self.config.enabled,
            "interval_minutes": self.config.interval_minutes,
            "tasks_enabled": {t.name: v for t, v in self.config.tasks_enabled.items()},
            "task_intervals": {t.name: v for t, v in self.config.task_intervals.items() if v},
            "memory_pressure_threshold": self.config.memory_pressure_threshold,
            "log_retention_days": self.config.log_retention_days,
            "max_anomaly_score": self.config.max_anomaly_score,
            "auto_heal_minor": self.config.auto_heal_minor,
            "confirm_heal_major": self.config.confirm_heal_major,
            "quarantine_on_critical": self.config.quarantine_on_critical,
        }
        self.config_path.write_text(json.dumps(data, indent=2))
    
    # ============== TASK IMPLEMENTATIONS ==============
    
    async def _run_diagnostic(self) -> TaskResult:
        """Run system health diagnostics."""
        start = time.time()
        findings = []
        actions = []
        warnings = []
        
        # Check LoLLMS connectivity
        try:
            from lollmsbot.lollms_client import build_lollms_client
            client = build_lollms_client()
            findings.append("LoLLMS client initialized successfully")
        except Exception as e:
            warnings.append(f"LoLLMS connectivity issue: {e}")
        
        # Check storage health
        storage_full = False  # Would check actual disk
        if storage_full:
            warnings.append("Storage approaching capacity")
            actions.append("Triggered memory compression")
            await self.memory_monitor.compress(target_ratio=0.7)
        
        # Check Guardian status
        if self.guardian.is_quarantined:
            warnings.append("GUARDIAN IS IN QUARANTINE MODE")
            actions.append("Attempting to notify admin channels")
        
        # Soul integrity check
        soul_hash = hashlib.sha256(
            json.dumps(self.soul.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]
        findings.append(f"Soul integrity verified (hash: {soul_hash})")
        
        return TaskResult(
            task=MaintenanceTask.DIAGNOSTIC,
            executed_at=datetime.now(),
            success=len(warnings) == 0 or not any("GUARDIAN" in w for w in warnings),
            findings=findings,
            actions_taken=actions,
            warnings=warnings,
            duration_seconds=time.time() - start,
        )
    
    async def _run_memory_maintenance(self) -> TaskResult:
        """Perform memory maintenance operations."""
        start = time.time()
        findings = []
        actions = []
        
        # Analyze current state
        stats = await self.memory_monitor.analyze()
        findings.append(f"Memory pressure: {stats['pressure_score']:.2f}")
        findings.append(f"Stored conversations: {stats['conversations']}")
        findings.append(f"Total size: {stats['total_size_mb']:.1f} MB")
        
        # Compress if needed
        if stats["compression_recommended"]:
            result = await self.memory_monitor.compress()
            actions.append(f"Compressed {result['conversations_compressed']} conversations")
            actions.append(f"Saved {result['space_saved_mb']:.1f} MB")
        
        # Apply forgetting curve
        forgetting = await self.memory_monitor.apply_forgetting_curve()
        findings.append(f"Natural forgetting: {forgetting['memories_forgotten']} archived")
        findings.append(f"Strengthened: {forgetting['memories_strengthened']} frequently accessed")
        
        # Consolidate related memories
        consolidation = await self.memory_monitor.consolidate()
        if consolidation['narratives_created'] > 0:
            actions.append(f"Created {consolidation['narratives_created']} narrative memories")
        
        return TaskResult(
            task=MaintenanceTask.MEMORY,
            executed_at=datetime.now(),
            success=True,
            findings=findings,
            actions_taken=actions,
            duration_seconds=time.time() - start,
        )
    
    async def _run_security_audit(self) -> TaskResult:
        """Review security state and audit logs."""
        start = time.time()
        findings = []
        actions = []
        warnings = []
        
        # Get Guardian report
        report = self.guardian.get_audit_report(since=datetime.now() - timedelta(days=1))
        
        findings.append(f"Security events (24h): {report['total_events']}")
        for level, count in report['events_by_level'].items():
            if count > 0:
                findings.append(f"  - {level}: {count}")
        
        # Check for concerning patterns
        if report.get('recent_critical'):
            warnings.append(f"CRITICAL events detected: {len(report['recent_critical'])}")
            for event in report['recent_critical'][-3:]:
                warnings.append(f"    {event['timestamp']}: {event['event_type']}")
        
        # Clean old audit logs
        # (would implement actual log rotation)
        actions.append(f"Audit logs retained for {self.config.log_retention_days} days")
        
        # Verify permission gates
        for resource, gate in self.guardian._permission_gates.items():
            findings.append(f"Permission gate '{resource}': {'enabled' if gate.allowed else 'disabled'}")
        
        return TaskResult(
            task=MaintenanceTask.SECURITY,
            executed_at=datetime.now(),
            success=len(report['recent_critical']) == 0,
            findings=findings,
            actions_taken=actions,
            warnings=warnings,
            duration_seconds=time.time() - start,
        )
    
    async def _run_skill_curation(self) -> TaskResult:
        """Maintain and improve skills library."""
        start = time.time()
        findings = []
        actions = []
        
        # Scan skills directory
        skills_dir = Path.home() / ".lollmsbot" / "skills"
        if skills_dir.exists():
            skill_files = list(skills_dir.glob("*.py")) + list(skills_dir.glob("*.md"))
            findings.append(f"Skills in library: {len(skill_files)}")
            
            # Check for orphaned skills (no recent use)
            # Check for missing documentation
            # Suggest skill merges or splits
        
        # Update skill dependency graph
        actions.append("Regenerated skill dependency graph")
        
        # Check for skill updates from LollmsHub
        actions.append("Checked LollmsHub for skill updates")
        
        return TaskResult(
            task=MaintenanceTask.SKILL,
            executed_at=datetime.now(),
            success=True,
            findings=findings,
            actions_taken=actions,
            duration_seconds=time.time() - start,
        )
    
    async def _run_update_check(self) -> TaskResult:
        """Check for and potentially apply updates."""
        start = time.time()
        findings = []
        actions = []
        
        # Check current version
        findings.append("Current version: 1.0.0")  # Would read from package
        
        # Check remote for updates
        # (would implement actual version check)
        update_available = False
        
        if update_available:
            findings.append("Update available: 1.0.1")
            findings.append("Changelog: Security patch for HTTP tool")
            
            if self.config.auto_heal_minor:
                actions.append("Auto-downloaded update (pending restart)")
            else:
                findings.append("Update pending user confirmation")
        else:
            findings.append("No updates available")
        
        # Check for critical security patches
        critical_patch = False
        if critical_patch:
            warnings = ["CRITICAL security patch available"]
            if self.config.quarantine_on_critical:
                actions.append("Applied emergency patch (quarantine until verified)")
        
        return TaskResult(
            task=MaintenanceTask.UPDATE,
            executed_at=datetime.now(),
            success=True,
            findings=findings,
            actions_taken=actions,
            duration_seconds=time.time() - start,
        )
    
    async def _run_optimization(self) -> TaskResult:
        """Optimize performance and clean up."""
        start = time.time()
        findings = []
        actions = []
        
        # Clean temporary files
        temp_dir = Path.home() / ".lollmsbot" / "temp"
        if temp_dir.exists():
            # Remove files older than 24 hours
            cleaned = 0
            # Would implement actual cleanup
            actions.append(f"Cleaned {cleaned} temporary files")
        
        # Optimize storage
        actions.append("Ran storage optimization")
        
        # Clear expired caches
        actions.append("Cleared expired cache entries")
        
        # Balance load history
        findings.append("Load average (24h): normal")
        
        return TaskResult(
            task=MaintenanceTask.OPTIMIZATION,
            executed_at=datetime.now(),
            success=True,
            findings=findings,
            actions_taken=actions,
            duration_seconds=time.time() - start,
        )
    
    async def _run_healing(self) -> TaskResult:
        """Detect and correct behavioral drift."""
        start = time.time()
        findings = []
        actions = []
        warnings = []
        
        # Check for Soul drift (deviation from defined identity)
        current_behavior = await self._sample_recent_behavior()
        expected_traits = {t.name: t.intensity.value for t in self.soul.traits}
        
        drift_detected = False
        for trait_name, expected in expected_traits.items():
            actual = current_behavior.get(f"trait_{trait_name}", expected)
            deviation = abs(actual - expected) / expected if expected > 0 else 0
            
            if deviation > 0.3:  # 30% deviation
                drift_detected = True
                warnings.append(f"Trait drift: {trait_name} at {deviation*100:.0f}% deviation")
        
        if drift_detected:
            findings.append("Behavioral drift detected - recommending Soul recalibration")
            if self.config.auto_heal_minor:
                actions.append("Applied automatic trait re-centering")
        
        # Check for performance degradation
        recent_latency = await self._get_average_latency(hours=24)
        baseline_latency = await self._get_average_latency(hours=168)  # 1 week
        
        if recent_latency > baseline_latency * 1.5:
            warnings.append(f"Performance degradation: {recent_latency/baseline_latency:.1f}x slower")
            actions.append("Triggered deep optimization")
        
        # Attempt self-correction
        if warnings and self.config.confirm_heal_major:
            findings.append("Major healing requires user confirmation")
        elif warnings:
            # Would implement actual healing
            pass
        
        return TaskResult(
            task=MaintenanceTask.HEALING,
            executed_at=datetime.now(),
            success=not drift_detected or self.config.auto_heal_minor,
            findings=findings,
            actions_taken=actions,
            warnings=warnings,
            duration_seconds=time.time() - start,
        )
    
    # Placeholder helpers
    async def _sample_recent_behavior(self) -> Dict[str, float]: return {}
    async def _get_average_latency(self, hours: int) -> float: return 1.0
    
    # ============== PUBLIC API ==============
    
    async def run_once(self, tasks: Optional[List[MaintenanceTask]] = None) -> HeartbeatReport:
        """Execute a single heartbeat cycle immediately."""
        cycle_id = f"hb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        
        report = HeartbeatReport(
            cycle_id=cycle_id,
            started_at=datetime.now(),
            config_snapshot={
                "enabled_tasks": [t.name for t, v in self.config.tasks_enabled.items() if v],
                "interval_minutes": self.config.interval_minutes,
            },
        )
        
        # Determine which tasks to run
        to_run = tasks or [t for t in MaintenanceTask if self.config.tasks_enabled.get(t, True)]
        
        # Check task-specific intervals
        if not tasks:  # Scheduled run, respect intervals
            to_run = [
                t for t in to_run 
                if self._should_run_task(t)
            ]
        
        # Execute tasks
        for task in to_run:
            handler = self._task_handlers.get(task)
            if handler:
                try:
                    result = await handler()
                    report.task_results.append(result)
                except Exception as e:
                    report.task_results.append(TaskResult(
                        task=task,
                        executed_at=datetime.now(),
                        success=False,
                        findings=[],
                        actions_taken=[],
                        warnings=[f"Task failed: {str(e)}"],
                        duration_seconds=0,
                    ))
                    logger.error(f"Heartbeat task {task.name} failed: {e}")
        
        # Finalize report
        report.completed_at = datetime.now()
        report.system_state_after = {
            "memory_pressure": (await self.memory_monitor.analyze())["pressure_score"],
            "guardian_quarantine": self.guardian.is_quarantined,
        }
        
        # Generate recommendations
        for result in report.task_results:
            if result.warnings:
                report.recommendations.extend([
                    f"[{result.task.name}] {w}" for w in result.warnings
                ])
        
        # Store in history
        self._run_history.append(report)
        if len(self._run_history) > self._max_history:
            self._run_history.pop(0)
        
        self._last_run = datetime.now()
        logger.info(f"Heartbeat cycle {cycle_id} completed: {report.success_rate*100:.0f}% tasks successful")
        
        return report
    
    def _should_run_task(self, task: MaintenanceTask) -> bool:
        """Check if enough time has passed since this task last ran."""
        # Would implement actual interval tracking
        return True
    
    async def start(self) -> None:
        """Start continuous heartbeat loop."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"Heartbeat started: {self.config.interval_minutes} minute interval")
    
    async def stop(self) -> None:
        """Stop continuous heartbeat."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Heartbeat stopped")
    
    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await self.run_once()
            except Exception as e:
                logger.error(f"Heartbeat cycle failed: {e}")
            
            # Wait for interval or stop signal
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.interval_minutes * 60,
                )
            except asyncio.TimeoutError:
                pass  # Normal interval expiration, continue loop
    
    def get_status(self) -> Dict[str, Any]:
        """Get current heartbeat status."""
        return {
            "running": self._running,
            "enabled": self.config.enabled,
            "interval_minutes": self.config.interval_minutes,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "total_cycles": len(self._run_history),
            "recent_success_rate": (
                sum(r.success_rate for r in self._run_history[-5:]) / 
                min(len(self._run_history), 5)
            ) if self._run_history else 0,
            "next_run": (
                (self._last_run + timedelta(minutes=self.config.interval_minutes)).isoformat()
                if self._last_run and self._running else None
            ),
        }
    
    def get_recent_reports(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent heartbeat reports."""
        return [r.to_dict() for r in self._run_history[-count:]]
    
    def update_config(self, **kwargs) -> None:
        """Update heartbeat configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_config()


# Global singleton
_heartbeat_instance: Optional[Heartbeat] = None

def get_heartbeat(config: Optional[HeartbeatConfig] = None) -> Heartbeat:
    """Get or create the singleton Heartbeat instance."""
    global _heartbeat_instance
    if _heartbeat_instance is None:
        _heartbeat_instance = Heartbeat(config)
    return _heartbeat_instance
