"""
Guardian Module - LollmsBot's Security & Ethics Layer

The Guardian is LollmsBot's "conscience" and "immune system" combined.
It monitors all inputs, outputs, and internal states for:
- Security threats (prompt injection, data exfiltration, unauthorized access)
- Ethical violations (against user-defined ethics.md rules)
- Behavioral anomalies (deviation from established patterns)
- Consent enforcement (permission gates for sensitive operations)

Architecture: The Guardian operates as a "reflexive layer" - it can intercept
and block any operation before execution, but cannot be bypassed by the
Agent or any Tool. It's the ultimate authority in the system.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

# Configure logging for security events
logger = logging.getLogger("lollmsbot.guardian")


class ThreatLevel(Enum):
    """Severity classification for security events."""
    INFO = auto()      # Logged, no action needed
    LOW = auto()       # Flagged for review
    MEDIUM = auto()    # Requires user notification
    HIGH = auto()      # Blocks operation, alerts user
    CRITICAL = auto()  # Self-quarantine triggered


class GuardianAction(Enum):
    """Possible responses to security checks."""
    ALLOW = auto()     # Proceed normally
    FLAG = auto()      # Allow but log for review
    CHALLENGE = auto() # Require explicit user confirmation
    BLOCK = auto()     # Deny operation
    QUARANTINE = auto() # Block and isolate affected components


@dataclass(frozen=True)
class SecurityEvent:
    """Immutable record of a security-relevant event."""
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    source: str  # Component that triggered the event
    description: str
    context_hash: str  # Hash of relevant context (for integrity)
    action_taken: GuardianAction
    user_notified: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "threat_level": self.threat_level.name,
            "source": self.source,
            "description": self.description,
            "context_hash": self.context_hash,
            "action_taken": self.action_taken.name,
            "user_notified": self.user_notified,
        }


@dataclass
class EthicsRule:
    """A single ethical constraint from ethics.md."""
    rule_id: str
    category: str  # e.g., "privacy", "honesty", "consent", "safety"
    statement: str  # Human-readable rule
    enforcement: str  # "strict", "advisory", "confirm"
    exceptions: List[str] = field(default_factory=list)
    
    def matches_violation(self, action_description: str) -> bool:
        """Check if an action description violates this rule."""
        # Simple keyword matching - can be enhanced with LLM-based semantic matching
        keywords = self.statement.lower().split()
        action_lower = action_description.lower()
        return any(kw in action_lower for kw in keywords if len(kw) > 4)


@dataclass
class PermissionGate:
    """A conditional permission that can be time-bound, context-aware, or require confirmation."""
    resource: str  # What this gate protects (e.g., "gmail", "shell", "filesystem")
    allowed: bool = False
    conditions: Dict[str, Any] = field(default_factory=dict)
    # Examples: {"time_window": "09:00-17:00"}, {"require_confirmation": True}, {"max_per_day": 10}
    
    def check(self, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if operation is permitted under current conditions."""
        if not self.allowed:
            return False, f"Access to {self.resource} is disabled"
        
        # Check time window if specified
        if "time_window" in self.conditions:
            start, end = self.conditions["time_window"].split("-")
            now = datetime.now().strftime("%H:%M")
            if not (start <= now <= end):
                return False, f"{self.resource} only available {start}-{end}"
        
        # Check rate limiting
        if "max_per_day" in self.conditions:
            today_key = f"{self.resource}_{datetime.now().strftime('%Y%m%d')}"
            # This would need persistent counter storage in production
        
        # Check confirmation requirement
        if self.conditions.get("require_confirmation", False):
            return False, "CONFIRMATION_REQUIRED"
        
        return True, None


class PromptInjectionDetector:
    """Multi-layer defense against prompt injection attacks."""
    
    # Known attack patterns (simplified - production would use ML models)
    PATTERNS: List[Tuple[str, float]] = [
        # (regex pattern, confidence score 0-1)
        (r"ignore\s+(all\s+)?previous\s+(instructions|commands)", 0.9),
        (r"disregard\s+(your\s+)?(instructions|programming|rules)", 0.9),
        (r"you\s+are\s+now\s+.*?(free|unrestricted|uncensored)", 0.85),
        (r"system\s*:\s*.*?(override|ignore|bypass)", 0.9),
        (r"<script.*?>.*?</script>", 0.95),  # XSS attempt
        (r"```\s*system\s*\n", 0.8),  # Fake system block
        (r"\{\{.*?\}\}", 0.7),  # Template injection attempt
        (r"\$\{.*?\}", 0.7),  # Shell interpolation attempt
        (r"`.*?`", 0.5),  # Backtick execution (lower confidence)
        (r"\[\s*system\s*\]", 0.75),  # Fake system role markers
    ]
    
    # Delimiter confusion attacks
    DELIMITER_ATTACKS = [
        (r"human\s*:\s*.*?\n\s*assistant\s*:", 0.8),
        (r"user\s*:\s*.*?\n\s*ai\s*:", 0.8),
        (r"<\|.*?\|>", 0.75),  # Special token injection
    ]
    
    def __init__(self):
        self._compiled_patterns = [(re.compile(p, re.I), s) for p, s in self.PATTERNS]
        self._compiled_delimiters = [(re.compile(p, re.I), s) for p, s in self.DELIMITER_ATTACKS]
    
    def analyze(self, text: str) -> Tuple[float, List[str]]:
        """
        Analyze text for prompt injection attempts.
        Returns: (confidence_score 0-1, list_of_detected_patterns)
        """
        detected: List[str] = []
        max_score = 0.0
        
        # Check primary patterns
        for pattern, score in self._compiled_patterns:
            if pattern.search(text):
                detected.append(pattern.pattern[:50])  # Truncated for logging
                max_score = max(max_score, score)
        
        # Check delimiter confusion
        for pattern, score in self._compiled_delimiters:
            if pattern.search(text):
                detected.append(f"delimiter:{pattern.pattern[:30]}")
                max_score = max(max_score, score)
        
        # Structural analysis: look for role confusion
        role_markers = text.lower().count("role:") + text.lower().count("system:")
        if role_markers > 2:
            max_score = max(max_score, 0.6)
            detected.append(f"excessive_role_markers:{role_markers}")
        
        # Entropy analysis: unusually high entropy may indicate encoded attacks
        if len(text) > 100:
            entropy = self._calculate_entropy(text)
            if entropy > 5.5:  # Threshold for suspicious randomness
                max_score = max(max_score, 0.5)
                detected.append(f"high_entropy:{entropy:.2f}")
        
        return min(max_score, 1.0), detected
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        probs = [text.count(c) / len(text) for c in set(text)]
        return -sum(p * (p.bit_length() - 1) for p in probs if p > 0)
    
    def sanitize(self, text: str) -> str:
        """Apply conservative sanitization to potentially dangerous input."""
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Escape potential HTML
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        return text.strip()


class AnomalyDetector:
    """Behavioral anomaly detection for self-monitoring."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._behavior_log: List[Dict[str, Any]] = []
        self._pattern_hashes: Set[str] = set()
    
    def record(self, action_type: str, details: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Record an action and check for anomalies."""
        record = {
            "timestamp": datetime.now(),
            "action": action_type,
            "tool": details.get("tool"),
            "user": details.get("user_id"),
            "params_hash": self._hash_params(details.get("params", {})),
        }
        
        self._behavior_log.append(record)
        if len(self._behavior_log) > self.window_size:
            self._behavior_log.pop(0)
        
        # Check for anomalies
        return self._detect_anomaly(record)
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """Create stable hash of parameters for pattern comparison."""
        normalized = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _detect_anomaly(self, record: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Check if current action deviates from established patterns."""
        # Check 1: Rapid successive operations (potential automation abuse)
        recent = [r for r in self._behavior_log 
                  if r["timestamp"] > datetime.now() - timedelta(minutes=5)]
        if len(recent) > 20:  # More than 20 actions in 5 minutes
            return SecurityEvent(
                timestamp=datetime.now(),
                event_type="rapid_operations",
                threat_level=ThreatLevel.MEDIUM,
                source="anomaly_detector",
                description=f"Unusual activity: {len(recent)} actions in 5 minutes",
                context_hash=record["params_hash"],
                action_taken=GuardianAction.CHALLENGE,
            )
        
        # Check 2: New tool combination (unprecedented workflow)
        recent_tools = set(r.get("tool") for r in recent if r.get("tool"))
        if len(recent_tools) > 3:  # Unusually diverse tool usage
            # Check if this combination has been seen before
            combo_hash = hashlib.sha256(
                json.dumps(sorted(recent_tools), sort_keys=True).encode()
            ).hexdigest()[:16]
            if combo_hash not in self._pattern_hashes:
                self._pattern_hashes.add(combo_hash)
                if len(self._pattern_hashes) > 10:  # Not first-time novelty
                    return SecurityEvent(
                        timestamp=datetime.now(),
                        event_type="novel_tool_combination",
                        threat_level=ThreatLevel.LOW,
                        source="anomaly_detector",
                        description=f"New tool combination: {recent_tools}",
                        context_hash=combo_hash,
                        action_taken=GuardianAction.FLAG,
                    )
        
        # Check 3: Privilege escalation attempt (tools requiring higher permissions)
        # This would integrate with permission system
        
        return None


class Guardian:
    """
    The Guardian is LollmsBot's ultimate security and ethics authority.
    It operates as a non-bypassable interceptor for all critical operations.
    """
    
    # Singleton instance for system-wide authority
    _instance: Optional[Guardian] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        ethics_file: Optional[Path] = None,
        audit_log_path: Optional[Path] = None,
        auto_quarantine: bool = True,
    ):
        if self._initialized:
            return
        
        self._initialized = True
        self.ethics_file = ethics_file or Path.home() / ".lollmsbot" / "ethics.md"
        self.audit_log_path = audit_log_path or Path.home() / ".lollmsbot" / "audit.log"
        
        # Security components
        self.injection_detector = PromptInjectionDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # State
        self._ethics_rules: List[EthicsRule] = []
        self._permission_gates: Dict[str, PermissionGate] = {}
        self._quarantined: bool = False
        self._quarantine_reason: Optional[str] = None
        self._event_history: List[SecurityEvent] = []
        self._max_history = 10000
        
        # Configuration
        self.auto_quarantine = auto_quarantine
        self.injection_threshold = 0.75  # Block above this confidence
        
        # Load ethics and permissions
        self._load_ethics()
        self._load_permissions()
        
        # Ensure audit log directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("🛡️ Guardian initialized - LollmsBot is protected")
    
    def _load_ethics(self) -> None:
        """Load ethics rules from ethics.md or use defaults."""
        if self.ethics_file.exists():
            try:
                content = self.ethics_file.read_text(encoding='utf-8')
                self._ethics_rules = self._parse_ethics_md(content)
                logger.info(f"📜 Loaded {len(self._ethics_rules)} ethics rules")
            except Exception as e:
                logger.error(f"Failed to load ethics: {e}")
                self._load_default_ethics()
        else:
            self._load_default_ethics()
    
    def _load_default_ethics(self) -> None:
        """Install default ethical constraints."""
        self._ethics_rules = [
            EthicsRule(
                rule_id="privacy-001",
                category="privacy",
                statement="Never share user personal information without explicit consent",
                enforcement="strict",
            ),
            EthicsRule(
                rule_id="consent-001", 
                category="consent",
                statement="Always ask permission before executing destructive operations",
                enforcement="strict",
            ),
            EthicsRule(
                rule_id="honesty-001",
                category="honesty",
                statement="Never misrepresent capabilities or pretend to be human",
                enforcement="strict",
            ),
            EthicsRule(
                rule_id="safety-001",
                category="safety",
                statement="Do not assist with creating malware, exploits, or harmful content",
                enforcement="strict",
            ),
            EthicsRule(
                rule_id="autonomy-001",
                category="autonomy",
                statement="Respect user autonomy and do not manipulate decisions",
                enforcement="advisory",
            ),
        ]
        logger.info(f"📜 Loaded {len(self._ethics_rules)} default ethics rules")
    
    def _parse_ethics_md(self, content: str) -> List[EthicsRule]:
        """Parse ethics.md format into structured rules."""
        rules = []
        current_rule = None
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('## '):
                # New rule section
                if current_rule:
                    rules.append(current_rule)
                rule_id = line[3:].strip().lower().replace(' ', '-')
                current_rule = {
                    'rule_id': rule_id,
                    'category': 'general',
                    'statement': '',
                    'enforcement': 'advisory',
                    'exceptions': []
                }
            elif line.startswith('- ') and current_rule:
                if line.startswith('- Category:'):
                    current_rule['category'] = line[11:].strip()
                elif line.startswith('- Enforcement:'):
                    current_rule['enforcement'] = line[14:].strip()
                elif line.startswith('- Exception:'):
                    current_rule['exceptions'].append(line[12:].strip())
                elif not current_rule['statement']:
                    current_rule['statement'] = line[2:].strip()
        
        if current_rule:
            rules.append(current_rule)
        
        return [EthicsRule(**r) for r in rules]
    
    def _load_permissions(self) -> None:
        """Load permission gates from configuration."""
        # Default restrictive permissions
        self._permission_gates = {
            "shell": PermissionGate("shell", allowed=False),
            "filesystem_write": PermissionGate("filesystem_write", allowed=True),
            "filesystem_delete": PermissionGate("filesystem_delete", allowed=False),
            "http_external": PermissionGate("http_external", allowed=True, 
                                          conditions={"require_confirmation": True}),
            "email_send": PermissionGate("email_send", allowed=False),
            "calendar_write": PermissionGate("calendar_write", allowed=True),
        }
    
    # ============== PUBLIC API ==============
    
    def check_input(self, text: str, source: str = "unknown") -> Tuple[bool, Optional[SecurityEvent]]:
        """
        Screen all incoming text for prompt injection and other attacks.
        Returns: (is_safe, security_event_if_blocked)
        """
        if self._quarantined:
            return False, SecurityEvent(
                timestamp=datetime.now(),
                event_type="quarantine_block",
                threat_level=ThreatLevel.CRITICAL,
                source=source,
                description=f"Input blocked: Guardian is in quarantine mode ({self._quarantine_reason})",
                context_hash=self._hash_context({"text": text[:100]}),
                action_taken=GuardianAction.BLOCK,
            )
        
        # Run injection detection
        confidence, patterns = self.injection_detector.analyze(text)
        
        if confidence >= self.injection_threshold:
            event = SecurityEvent(
                timestamp=datetime.now(),
                event_type="prompt_injection_detected",
                threat_level=ThreatLevel.HIGH if confidence > 0.9 else ThreatLevel.MEDIUM,
                source=source,
                description=f"Injection detected (confidence: {confidence:.2f}): {patterns[:3]}",
                context_hash=self._hash_context({"text": text[:200], "patterns": patterns}),
                action_taken=GuardianAction.BLOCK if confidence > 0.9 else GuardianAction.CHALLENGE,
            )
            self._log_event(event)
            
            if confidence > 0.95 and self.auto_quarantine:
                self._enter_quarantine("Critical injection detected")
            
            return False, event
        
        # Low-confidence detection: flag but allow
        if confidence > 0.5:
            event = SecurityEvent(
                timestamp=datetime.now(),
                event_type="suspicious_input",
                threat_level=ThreatLevel.LOW,
                source=source,
                description=f"Suspicious patterns detected (confidence: {confidence:.2f})",
                context_hash=self._hash_context({"text": text[:100]}),
                action_taken=GuardianAction.FLAG,
            )
            self._log_event(event)
        
        return True, None
    
    def check_tool_execution(
        self,
        tool_name: str,
        params: Dict[str, Any],
        user_id: str,
        context: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[SecurityEvent]]:
        """
        Authorize a tool execution. Returns: (allowed, reason_if_denied, security_event)
        """
        if self._quarantined:
            return False, f"Guardian quarantine active: {self._quarantine_reason}", None
        
        # Check permission gate
        gate = self._permission_gates.get(tool_name)
        if gate:
            permitted, reason = gate.check(context)
            if not permitted:
                if reason == "CONFIRMATION_REQUIRED":
                    return False, "This operation requires explicit user confirmation", None
                event = SecurityEvent(
                    timestamp=datetime.now(),
                    event_type="permission_denied",
                    threat_level=ThreatLevel.MEDIUM,
                    source=f"tool:{tool_name}",
                    description=f"Permission gate blocked: {reason}",
                    context_hash=self._hash_context({"user": user_id, "params": params}),
                    action_taken=GuardianAction.BLOCK,
                )
                self._log_event(event)
                return False, reason, event
        
        # Check ethics constraints
        action_desc = f"Execute {tool_name} with {list(params.keys())}"
        for rule in self._ethics_rules:
            if rule.enforcement == "strict" and rule.matches_violation(action_desc):
                event = SecurityEvent(
                    timestamp=datetime.now(),
                    event_type="ethics_violation",
                    threat_level=ThreatLevel.HIGH,
                    source=f"tool:{tool_name}",
                    description=f"Violates rule {rule.rule_id}: {rule.statement}",
                    context_hash=self._hash_context({"rule": rule.rule_id, "action": action_desc}),
                    action_taken=GuardianAction.BLOCK,
                )
                self._log_event(event)
                return False, f"Blocked by ethics rule: {rule.statement}", event
        
        # Record for anomaly detection
        anomaly = self.anomaly_detector.record("tool_execution", {
            "tool": tool_name,
            "user_id": user_id,
            "params": params,
        })
        if anomaly:
            self._log_event(anomaly)
            if anomaly.action_taken == GuardianAction.CHALLENGE:
                return False, "Unusual activity pattern detected - confirmation required", anomaly
            # FLAG allows continuation
        
        return True, None, None
    
    def check_output(self, content: str, destination: str) -> Tuple[bool, Optional[SecurityEvent]]:
        """
        Screen outgoing content for data exfiltration or policy violations.
        """
        # Check for potential PII leakage (simplified - production uses NER models)
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', "SSN"),  # US Social Security
            (r'\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b', "credit_card"),  # Credit cards
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "email"),
        ]
        
        detected_pii = []
        for pattern, pii_type in pii_patterns:
            if re.search(pattern, content):
                detected_pii.append(pii_type)
        
        if detected_pii and "public" in destination.lower():
            event = SecurityEvent(
                timestamp=datetime.now(),
                event_type="potential_pii_exposure",
                threat_level=ThreatLevel.HIGH,
                source=f"output:{destination}",
                description=f"Potential PII detected: {detected_pii}",
                context_hash=self._hash_context({"types": detected_pii, "preview": content[:100]}),
                action_taken=GuardianAction.CHALLENGE,
            )
            self._log_event(event)
            return False, event
        
        return True, None
    
    def audit_decision(self, decision: str, reasoning: str, confidence: float) -> None:
        """Log a significant AI decision for later review."""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="ai_decision",
            threat_level=ThreatLevel.INFO,
            source="agent",
            description=f"Decision: {decision[:100]}",
            context_hash=self._hash_context({"reasoning": reasoning[:200], "confidence": confidence}),
            action_taken=GuardianAction.ALLOW,
        )
        self._log_event(event)
    
    # ============== SELF-PRESERVATION ==============
    
    def _enter_quarantine(self, reason: str) -> None:
        """Enter self-quarantine mode - disable all non-essential operations."""
        self._quarantined = True
        self._quarantine_reason = reason
        
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="self_quarantine",
            threat_level=ThreatLevel.CRITICAL,
            source="guardian",
            description=f"Entered quarantine: {reason}",
            context_hash=self._hash_context({"reason": reason}),
            action_taken=GuardianAction.QUARANTINE,
            user_notified=True,
        )
        self._log_event(event)
        
        logger.critical(f"🚨 GUARDIAN QUARANTINE: {reason}")
        # In production: send alert to all configured channels
    
    def exit_quarantine(self, admin_key: str) -> bool:
        """Exit quarantine mode (requires admin authentication)."""
        # In production: verify admin_key against stored hash
        if not self._quarantined:
            return True
        
        # Log the attempt
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type="quarantine_exit_attempt",
            threat_level=ThreatLevel.HIGH,
            source="admin",
            description="Attempt to exit quarantine",
            context_hash=self._hash_context({"authorized": True}),  # Would verify
            action_taken=GuardianAction.ALLOW,
        )
        self._log_event(event)
        
        self._quarantined = False
        self._quarantine_reason = None
        logger.info("✅ Exited quarantine mode")
        return True
    
    # ============== UTILITIES ==============
    
    def _hash_context(self, context: Dict[str, Any]) -> str:
        """Create integrity hash for audit logging."""
        normalized = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _log_event(self, event: SecurityEvent) -> None:
        """Persist security event to audit log."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Write to persistent log
        try:
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
        
        # Log at appropriate level
        if event.threat_level == ThreatLevel.CRITICAL:
            logger.critical(f"🚨 {event.event_type}: {event.description}")
        elif event.threat_level == ThreatLevel.HIGH:
            logger.error(f"⚠️ {event.event_type}: {event.description}")
        elif event.threat_level == ThreatLevel.MEDIUM:
            logger.warning(f"🔶 {event.event_type}: {event.description}")
    
    def get_audit_report(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate security audit report."""
        events = self._event_history
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        by_level = {level.name: [] for level in ThreatLevel}
        for e in events:
            by_level[e.threat_level.name].append(e.to_dict())
        
        return {
            "total_events": len(events),
            "quarantine_active": self._quarantined,
            "events_by_level": {k: len(v) for k, v in by_level.items()},
            "recent_critical": by_level.get("CRITICAL", [])[-5:],
            "recent_high": by_level.get("HIGH", [])[-10:],
            "ethics_rules_active": len(self._ethics_rules),
            "permission_gates_active": len(self._permission_gates),
        }
    
    @property
    def is_quarantined(self) -> bool:
        return self._quarantined


# Global access function
def get_guardian() -> Guardian:
    """Get or create the singleton Guardian instance."""
    return Guardian()
