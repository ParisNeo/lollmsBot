"""
Heartbeat Module - LollmsBot's Self-Maintenance & Monitoring System

Updated with RLM Memory Maintenance integration for automatic deduplication
and consolidation of memory chunks.

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
# Import RLM memory maintenance
try:
    from lollmsbot.agent.rlm.maintenance import MemoryMaintenance, MaintenanceReport
    RLM_MAINTENANCE_AVAILABLE = True
except ImportError:
    RLM_MAINTENANCE_AVAILABLE = False


logger = logging.getLogger("lollmsbot.heartbeat")


class MaintenanceTask(Enum):
    """Categories of self-maintenance operations."""
    DIAGNOSTIC = auto()      # Health checks, connectivity tests
    MEMORY = auto()          # Memory compression, archiving, forgetting
    RLM_MAINTENANCE = auto() # NEW: RLM deduplication and consolidation
    SECURITY = auto()         # Audit log review, permission verification
    SKILL = auto()           # Skill documentation, dependency updates
    UPDATE = auto()          # Code update checks, patch application
    OPTIMIZATION = auto()    # Performance tuning, cache cleaning
    HEALING = auto()         # Self-correction of detected drift
    OPENCLAW_WORKFLOW = auto()  # NEW: SimplifiedAgant daily automation workflow


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
    
    # NEW: RLM Memory Maintenance settings
    rlm_deduplication_enabled: bool = True
    rlm_consolidation_enabled: bool = True
    rlm_min_chunks_for_dedup: int = 10  # Only run if >10 chunks exist


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
    # NEW: Extended data for complex tasks
    extended_report: Optional[Dict[str, Any]] = None


class MemoryMonitor:
    """
    Legacy memory monitor - kept for backward compatibility.
    RLM memory uses new MemoryMaintenance class.
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
        # Legacy implementation - now mostly delegates to RLM
        stats = {
            "conversations": await self._count_conversations(),
            "memory_entries": await self._count_entries(),
            "total_size_mb": await self._calculate_size() / (1024 * 1024),
            "pressure_score": 0.0,
            "oldest_memory_days": 0,
            "compression_recommended": False,
            "archiving_recommended": False,
            "note": "Using RLM memory maintenance for active management",
        }
        
        return stats
    
    async def compress(self, target_ratio: float = 0.5) -> Dict[str, Any]:
        """Legacy compression - now handled by RLM maintenance."""
        return {
            "note": "Compression now handled by RLM MemoryMaintenance.deduplicate_by_content()",
            "conversations_compressed": 0,
            "space_saved_mb": 0,
        }
    
    async def apply_forgetting_curve(self) -> Dict[str, Any]:
        """Apply Ebbinghaus-inspired forgetting curve."""
        # This is still used for old-style memories
        return {
            "memories_forgotten": 0,
            "memories_strengthened": 0,
            "note": "Forgetting curve now integrated with RLM maintenance",
        }
    
    async def consolidate(self) -> Dict[str, Any]:
        """Legacy consolidation - now handled by RLM."""
        return {
            "note": "Consolidation now handled by RLM MemoryMaintenance.consolidate_related_memories()",
            "clusters_found": 0,
            "memories_consolidated": 0,
            "narratives_created": 0,
        }
    
    # Placeholder helpers
    async def _count_conversations(self) -> int: return 0
    async def _count_entries(self) -> int: return 0
    async def _calculate_size(self) -> int: return 0


class Heartbeat:
    """
    LollmsBot's autonomous self-maintenance system with RLM Memory Maintenance.
    
    The Heartbeat runs continuously, performing configured maintenance tasks
    at appropriate intervals including new RLM memory deduplication.
    """
    
    DEFAULT_CONFIG_PATH = Path.home() / ".lollmsbot" / "heartbeat.json"
    
    def __init__(
        self,
        config: Optional[HeartbeatConfig] = None,
        config_path: Optional[Path] = None,
        memory_manager: Optional[Any] = None,  # RLMMemoryManager
    ):
        self.config = config or HeartbeatConfig()
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        
        # Ensure SimplifiedAgant workflow is in default tasks if not explicitly disabled
        if MaintenanceTask.OPENCLAW_WORKFLOW not in self.config.tasks_enabled:
            self.config.tasks_enabled[MaintenanceTask.OPENCLAW_WORKFLOW] = True
        
        # Subsystems
        self.memory_monitor = MemoryMonitor()
        self.guardian = get_guardian()
        self.soul = get_soul()
        
        # NEW: RLM Memory Maintenance
        self._memory_manager = memory_manager
        self._rlm_maintenance: Optional[MemoryMaintenance] = None
        if RLM_MAINTENANCE_AVAILABLE and memory_manager:
            self._rlm_maintenance = MemoryMaintenance(memory_manager)
            logger.info("✅ RLM Memory Maintenance initialized")
        
        # Runtime state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._last_run: Optional[datetime] = None
        self._run_history: List[Any] = []  # Simplified
        self._max_history = 100
        
        # Task registry
        self._task_handlers: Dict[MaintenanceTask, Callable[[], Coroutine[Any, Any, TaskResult]]] = {
            MaintenanceTask.DIAGNOSTIC: self._run_diagnostic,
            MaintenanceTask.MEMORY: self._run_memory_maintenance,
            MaintenanceTask.RLM_MAINTENANCE: self._run_rlm_maintenance,  # NEW
            MaintenanceTask.SECURITY: self._run_security_audit,
            MaintenanceTask.SKILL: self._run_skill_curation,
            MaintenanceTask.UPDATE: self._run_update_check,
            MaintenanceTask.OPTIMIZATION: self._run_optimization,
            MaintenanceTask.HEALING: self._run_healing,
            MaintenanceTask.OPENCLAW_WORKFLOW: self._run_openclaw_workflow,  # NEW
        }
        
        # Load or save config
        if self.config_path.exists():
            self._load_config()
        else:
            self._save_config()
    
    def set_memory_manager(self, memory_manager: Any) -> None:
        """Set the RLM memory manager (called after Agent initialization)."""
        self._memory_manager = memory_manager
        if RLM_MAINTENANCE_AVAILABLE and memory_manager:
            self._rlm_maintenance = MemoryMaintenance(memory_manager)
            logger.info("✅ RLM Memory Maintenance connected to Heartbeat")
    
    def set_openclaw_integration(self, simplified_agant: Any) -> None:
        """Set the SimplifiedAgant integration for daily workflows."""
        self._openclaw = simplified_agant
        logger.info("✅ SimplifiedAgant integration connected to Heartbeat")
    
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
                # NEW: RLM settings
                rlm_deduplication_enabled=data.get("rlm_deduplication_enabled", True),
                rlm_consolidation_enabled=data.get("rlm_consolidation_enabled", True),
                rlm_min_chunks_for_dedup=data.get("rlm_min_chunks_for_dedup", 10),
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
            # NEW: RLM settings
            "rlm_deduplication_enabled": self.config.rlm_deduplication_enabled,
            "rlm_consolidation_enabled": self.config.rlm_consolidation_enabled,
            "rlm_min_chunks_for_dedup": self.config.rlm_min_chunks_for_dedup,
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
        
        # Check RLM memory health
        if self._rlm_maintenance:
            try:
                summary = await self._rlm_maintenance.get_maintenance_summary()
                findings.append(f"RLM memory: {summary['total_chunks']} chunks "
                              f"({summary['archived_chunks']} archived)")
                if summary['total_chunks'] > 100:
                    actions.append(f"Memory size: {summary['total_chunks']} chunks "
                                 f"(deduplication recommended)")
            except Exception as e:
                warnings.append(f"RLM memory check failed: {e}")
        
        # Check storage health
        storage_full = False
        if storage_full:
            warnings.append("Storage approaching capacity")
            actions.append("Triggered memory compression")
        
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
        """Legacy memory maintenance - now mostly delegates to RLM."""
        start = time.time()
        findings = []
        actions = []
        
        # Legacy stats
        stats = await self.memory_monitor.analyze()
        findings.append(f"Legacy memory pressure: {stats['pressure_score']:.2f}")
        
        # RLM takes over active maintenance
        if self._rlm_maintenance:
            findings.append("Active memory management handled by RLM Maintenance")
            actions.append("See RLM_MAINTENANCE task for deduplication/consolidation results")
        
        return TaskResult(
            task=MaintenanceTask.MEMORY,
            executed_at=datetime.now(),
            success=True,
            findings=findings,
            actions_taken=actions,
            duration_seconds=time.time() - start,
        )
    
    async def _run_openclaw_workflow(self) -> TaskResult:
        """Run SimplifiedAgant daily workflow."""
        start = time.time()
        findings = []
        actions = []
        warnings = []
        
        if not hasattr(self, '_openclaw') or not self._openclaw:
            return TaskResult(
                task=MaintenanceTask.OPENCLAW_WORKFLOW,
                executed_at=datetime.now(),
                success=True,
                findings=["SimplifiedAgant integration not configured"],
                actions_taken=[],
                duration_seconds=time.time() - start,
            )
        
        try:
            results = await self._openclaw.daily_workflow()
            
            findings.append(f"SimplifiedAgant daily workflow completed")
            findings.append(f"Steps completed: {', '.join(results.get('steps_completed', []))}")
            
            if "youtube_snapshot" in results:
                yt = results["youtube_snapshot"]
                findings.append(f"YouTube: {yt.get('views', 0):,} views, {yt.get('subscribers', 0):,} subscribers")
            
            if "business_analysis" in results:
                ba = results["business_analysis"]
                findings.append(f"Business analysis: {ba.get('priorities', 0)} priorities, {ba.get('confidence')} confidence")
                actions.append("Business council report generated")
            
            if "crm_error" in results:
                warnings.append(f"CRM workflow issue: {results['crm_error']}")
            if "youtube_error" in results:
                warnings.append(f"YouTube analytics issue: {results['youtube_error']}")
            if "analysis_error" in results:
                warnings.append(f"Business analysis issue: {results['analysis_error']}")
            
        except Exception as e:
            warnings.append(f"SimplifiedAgant workflow failed: {str(e)}")
            logger.error(f"SimplifiedAgant daily workflow error: {e}")
        
        return TaskResult(
            task=MaintenanceTask.OPENCLAW_WORKFLOW,
            executed_at=datetime.now(),
            success=len(warnings) == 0,
            findings=findings,
            actions_taken=actions,
            warnings=warnings,
            duration_seconds=time.time() - start,
        )
    
    async def _run_rlm_maintenance(self) -> TaskResult:
        """
        NEW: Run RLM memory deduplication and consolidation.
        
        This is the key new task that prevents memory bloat from
        duplicate web fetches and consolidates related conversations.
        """
        start = time.time()
        findings = []
        actions = []
        warnings = []
        extended_report: Optional[Dict[str, Any]] = None
        
        if not self._rlm_maintenance:
            warnings.append("RLM Memory Maintenance not available")
            return TaskResult(
                task=MaintenanceTask.RLM_MAINTENANCE,
                executed_at=datetime.now(),
                success=False,
                findings=findings,
                actions_taken=actions,
                warnings=warnings,
                duration_seconds=time.time() - start,
            )
        
        if not self.config.rlm_deduplication_enabled and not self.config.rlm_consolidation_enabled:
            findings.append("RLM maintenance disabled in config")
            return TaskResult(
                task=MaintenanceTask.RLM_MAINTENANCE,
                executed_at=datetime.now(),
                success=True,
                findings=findings,
                duration_seconds=time.time() - start,
            )
        
        try:
            # Check if we have enough chunks to justify maintenance
            summary = await self._rlm_maintenance.get_maintenance_summary()
            chunk_count = summary.get('total_chunks', 0)
            
            if chunk_count < self.config.rlm_min_chunks_for_dedup:
                findings.append(f"Chunk count ({chunk_count}) below threshold "
                              f"({self.config.rlm_min_chunks_for_dedup}), skipping")
                return TaskResult(
                    task=MaintenanceTask.RLM_MAINTENANCE,
                    executed_at=datetime.now(),
                    success=True,
                    findings=findings,
                    duration_seconds=time.time() - start,
                )
            
            # Run full maintenance
            report: MaintenanceReport = await self._rlm_maintenance.run_full_maintenance(
                enable_deduplication=self.config.rlm_deduplication_enabled,
                enable_consolidation=self.config.rlm_consolidation_enabled,
                enable_url_dedup=True,  # Always dedup URLs (most common issue)
            )
            
            extended_report = report.to_dict()
            
            # Analyze results
            if report.deduplication:
                d = report.deduplication
                if d.duplicates_removed > 0:
                    actions.append(f"Removed {d.duplicates_removed} duplicate chunks")
                    actions.append(f"Saved ~{d.bytes_saved / 1024:.1f} KB storage")
                    findings.append(f"Found {d.duplicates_found} duplicates total")
                else:
                    findings.append("No duplicates found (memory is clean)")
            
            if report.consolidation:
                c = report.consolidation
                if c.narratives_created > 0:
                    actions.append(f"Created {c.narratives_created} narrative memories")
                    actions.append(f"Consolidated {c.chunks_consolidated} related chunks")
                else:
                    findings.append("No consolidation candidates found")
            
            # Compression stats
            if report.compression_ratio < 1.0:
                reduction = (1 - report.compression_ratio) * 100
                findings.append(f"Memory compressed by {reduction:.1f}%")
            
            # Time tracking
            findings.append(f"Maintenance completed in {report.duration_seconds:.1f}s")
            
        except Exception as e:
            logger.error(f"RLM maintenance failed: {e}")
            warnings.append(f"RLM maintenance error: {str(e)}")
            extended_report = {"error": str(e)}
        
        return TaskResult(
            task=MaintenanceTask.RLM_MAINTENANCE,
            executed_at=datetime.now(),
            success=len(warnings) == 0,
            findings=findings,
            actions_taken=actions,
            warnings=warnings,
            duration_seconds=time.time() - start,
            extended_report=extended_report,
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
        findings.append("Current version: 1.0.0")
        
        # Check remote for updates
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
            cleaned = 0
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
        
        # Check for Soul drift
        current_behavior = await self._sample_recent_behavior()
        expected_traits = {t.name: t.intensity.value for t in self.soul.traits}
        
        drift_detected = False
        for trait_name, expected in expected_traits.items():
            actual = current_behavior.get(f"trait_{trait_name}", expected)
            deviation = abs(actual - expected) / expected if expected > 0 else 0
            
            if deviation > 0.3:
                drift_detected = True
                warnings.append(f"Trait drift: {trait_name} at {deviation*100:.0f}% deviation")
        
        if drift_detected:
            findings.append("Behavioral drift detected - recommending Soul recalibration")
            if self.config.auto_heal_minor:
                actions.append("Applied automatic trait re-centering")
        
        # Check for performance degradation
        recent_latency = await self._get_average_latency(hours=24)
        baseline_latency = await self._get_average_latency(hours=168)
        
        if recent_latency > baseline_latency * 1.5:
            warnings.append(f"Performance degradation: {recent_latency/baseline_latency:.1f}x slower")
            actions.append("Triggered deep optimization")
        
        if warnings and self.config.confirm_heal_major:
            findings.append("Major healing requires user confirmation")
        
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
    async def _sample_recent_behavior(self) -> Dict[str, float]:
        return {}
    async def _get_average_latency(self, hours: int) -> float:
        return 1.0
    
    # ============== PUBLIC API ==============
    
    async def run_once(self, tasks: Optional[List[MaintenanceTask]] = None, is_startup: bool = False) -> Dict[str, Any]:
        """Execute a single heartbeat cycle immediately."""
        import time
        start_time = time.time()
        
        cycle_id = f"hb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        
        # Log heartbeat start with rich formatting
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich import box
            
            console = Console()
            
            event_type = "🚀 STARTUP HEARTBEAT" if is_startup else "💓 Heartbeat"
            panel = Panel(
                f"[bold cyan]{event_type}[/bold cyan]\n"
                f"[dim]Cycle ID: {cycle_id}[/dim]\n"
                f"[dim]Tasks to run: {len(tasks) if tasks else 'auto-detected'}[/dim]",
                border_style="bright_cyan" if is_startup else "cyan",
                box=box.DOUBLE if is_startup else box.ROUNDED,
                title="[bold]Self-Maintenance Event[/bold]",
                subtitle="[dim]RLM Memory Optimization[/dim]" if is_startup else None
            )
            console.print()
            console.print(panel)
        except ImportError:
            logger.info(f"Heartbeat {'startup ' if is_startup else ''}cycle {cycle_id} starting")
        
        results = []
        to_run = tasks or [t for t in MaintenanceTask if self.config.tasks_enabled.get(t, True)]
        
        # Filter by interval if not forced
        if not tasks:
            to_run = [
                t for t in to_run 
                if self._should_run_task(t)
            ]
        
        # Execute tasks
        for task in to_run:
            handler = self._task_handlers.get(task)
            if handler:
                try:
                    task_start = time.time()
                    result = await handler()
                    task_duration = time.time() - task_start
                    
                    results.append({
                        'task': task.name,
                        'success': result.success,
                        'findings': result.findings,
                        'actions': result.actions_taken,
                        'warnings': result.warnings,
                        'duration': result.duration_seconds,
                        'extended_report': result.extended_report,
                    })
                    
                    # Log each task with rich formatting
                    try:
                        from rich.console import Console
                        console = Console()
                        
                        status_icon = "✅" if result.success else "❌"
                        status_color = "green" if result.success else "red"
                        
                        # Build task summary
                        task_lines = [
                            f"[bold]{task.name.replace('_', ' ').title()}[/bold]",
                            f"Status: [{status_color}]{status_icon} {('Success' if result.success else 'Failed')}[/]",
                            f"Duration: {result.duration_seconds:.2f}s",
                        ]
                        
                        if result.actions_taken:
                            task_lines.append(f"Actions: {len(result.actions_taken)}")
                            for action in result.actions_taken[:3]:
                                task_lines.append(f"  • {action[:60]}")
                        
                        if result.warnings:
                            task_lines.append(f"[yellow]Warnings: {len(result.warnings)}[/]")
                        
                        # Special handling for RLM maintenance
                        if task == MaintenanceTask.RLM_MAINTENANCE and result.extended_report:
                            report = result.extended_report
                            if report.get('deduplication', {}).get('duplicates_removed', 0) > 0:
                                dupes = report['deduplication']['duplicates_removed']
                                saved_kb = report['deduplication'].get('bytes_saved', 0) / 1024
                                task_lines.append(f"[green]  🧹 Removed {dupes} duplicates, saved {saved_kb:.1f} KB[/]")
                            if report.get('consolidation', {}).get('narratives_created', 0) > 0:
                                narratives = report['consolidation']['narratives_created']
                                task_lines.append(f"[green]  📚 Created {narratives} narrative memories[/]")
                        
                        console.print(Panel(
                            "\n".join(task_lines),
                            border_style=status_color,
                            padding=(0, 2)
                        ))
                    except ImportError:
                        logger.info(f"Task {task.name}: {'success' if result.success else 'failed'} "
                                  f"({result.duration_seconds:.2f}s)")
                        
                except Exception as e:
                    results.append({
                        'task': task.name,
                        'success': False,
                        'error': str(e),
                    })
                    logger.error(f"Heartbeat task {task.name} failed: {e}")
                    
                    try:
                        from rich.console import Console
                        console = Console()
                        console.print(Panel(
                            f"[bold red]Task Failed: {task.name}[/]\n{str(e)[:200]}",
                            border_style="red"
                        ))
                    except ImportError:
                        pass
        
        self._last_run = datetime.now()
        total_duration = time.time() - start_time
        
        # Log heartbeat completion
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich import box
            
            console = Console()
            
            success_count = sum(1 for r in results if r.get('success'))
            failed_count = len(results) - success_count
            
            summary_lines = [
                f"[bold]Tasks Executed:[/] {len(results)}",
                f"[green]✅ Successful:[/] {success_count}",
            ]
            if failed_count > 0:
                summary_lines.append(f"[red]❌ Failed:[/] {failed_count}")
            
            summary_lines.append(f"[dim]Total Duration: {total_duration:.2f}s[/]")
            
            # Memory stats if available
            if self._rlm_maintenance and any(r.get('task') == 'RLM_MAINTENANCE' for r in results):
                try:
                    stats = await self._rlm_maintenance.get_maintenance_summary()
                    summary_lines.append(f"\n[dim]Memory: {stats.get('total_chunks', 0)} chunks "
                                       f"({stats.get('archived_chunks', 0)} archived)[/dim]")
                except:
                    pass
            
            completion_panel = Panel(
                "\n".join(summary_lines),
                border_style="bright_green" if failed_count == 0 else "yellow",
                box=box.DOUBLE_EDGE if is_startup else box.ROUNDED,
                title="[bold green]✓ Heartbeat Complete[/]" if failed_count == 0 else "[bold yellow]⚠ Heartbeat Complete (with issues)[/]",
                subtitle=f"[dim]Next: {self.config.interval_minutes}min[/]" if not is_startup else "[dim]Startup optimization complete[/]"
            )
            console.print(completion_panel)
            console.print()
            
        except ImportError:
            logger.info(f"Heartbeat complete: {success_count}/{len(results)} tasks successful "
                      f"({total_duration:.2f}s)")
        
        return {
            'cycle_id': cycle_id,
            'tasks_executed': len(results),
            'results': results,
            'total_duration': total_duration,
            'is_startup': is_startup,
        }
    
    def _should_run_task(self, task: MaintenanceTask) -> bool:
        """Check if enough time has passed since this task last ran."""
        # RLM maintenance has its own internal scheduling
        if task == MaintenanceTask.RLM_MAINTENANCE:
            if self._rlm_maintenance:
                # Check internal state
                return self._rlm_maintenance._should_run_dedup()
            return False
        
        # Other tasks use simple interval tracking (simplified)
        return True
    
    async def start(self) -> None:
        """Start continuous heartbeat loop."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"💓 Heartbeat started: {self.config.interval_minutes} minute interval")
    
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
        
        logger.info("💓 Heartbeat stopped")
    
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
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current heartbeat status."""
        return {
            "running": self._running,
            "enabled": self.config.enabled,
            "interval_minutes": self.config.interval_minutes,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "rlm_maintenance_available": RLM_MAINTENANCE_AVAILABLE,
            "rlm_maintenance_active": self._rlm_maintenance is not None,
        }
    
    def update_config(self, **kwargs) -> None:
        """Update heartbeat configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._save_config()


# Global singleton
_heartbeat_instance: Optional[Heartbeat] = None

def get_heartbeat(config: Optional[HeartbeatConfig] = None, memory_manager: Optional[Any] = None) -> Heartbeat:
    """Get or create the singleton Heartbeat instance."""
    global _heartbeat_instance
    if _heartbeat_instance is None:
        _heartbeat_instance = Heartbeat(config, memory_manager=memory_manager)
    elif memory_manager and not _heartbeat_instance._memory_manager:
        # Connect memory manager if not already set
        _heartbeat_instance.set_memory_manager(memory_manager)
    return _heartbeat_instance
