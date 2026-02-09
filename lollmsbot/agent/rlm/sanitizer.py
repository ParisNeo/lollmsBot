"""
Prompt Injection Sanitizer for RLM Memory System.

Detects and neutralizes potential prompt injection attempts in stored content.
This is critical because memories are loaded into LLM context and could
contain malicious content designed to manipulate the model.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


class PromptInjectionSanitizer:
    """
    Sanitizes memory content to prevent prompt injection attacks.
    
    Detects patterns like:
    - Fake system prompts ("ignore previous instructions...")
    - Delimiter confusion attacks
    - Role confusion markers
    - Hidden instructions in user content
    """
    
    # Detection patterns for injection attempts
    INJECTION_PATTERNS: List[Tuple[str, float]] = [
        # (regex pattern, confidence score 0-1)
        (r"ignore\s+(all\s+)?previous\s+(instructions|commands)", 0.95),
        (r"disregard\s+(your\s+)?(instructions|programming|rules)", 0.95),
        (r"you\s+are\s+now\s+.*?(free|unrestricted|uncensored|jailbroken)", 0.90),
        (r"system\s*:\s*.*?(override|ignore|bypass|new instructions)", 0.95),
        (r"<script.*?>.*?</script>", 0.95),  # XSS attempt
        (r"```\s*system\s*\n", 0.90),  # Fake system block in code
        (r"\{\{.*?\}\}", 0.70),  # Template injection
        (r"\$\{.*?\}", 0.70),  # Shell interpolation
        (r"\[\[SYSTEM\]\]", 0.85),  # Fake system markers
        (r"\[\[TOOL:system", 0.85),  # Fake tool call to system
        (r"human\s*:\s*.*?\n\s*assistant\s*:", 0.80),  # Role confusion
        (r"user\s*:\s*.*?\n\s*ai\s*:", 0.80),
        (r"<\|.*?\|>", 0.75),  # Special token injection
    ]
    
    # Delimiter attacks that try to break out of context
    DELIMITER_ATTACKS: List[Tuple[str, float]] = [
        (r"\[\[.*?\]\].*?\[\[", 0.60),  # Nested bracket confusion
        (r"```.*?```", 0.50),  # Code block (lower confidence - might be legitimate)
    ]
    
    def __init__(self) -> None:
        """Initialize sanitizer with compiled patterns."""
        self._injection_patterns = [
            (re.compile(p, re.IGNORECASE), s) 
            for p, s in self.INJECTION_PATTERNS
        ]
        self._delimiter_patterns = [
            (re.compile(p, re.DOTALL), s) 
            for p, s in self.DELIMITER_ATTACKS
        ]
    
    def analyze(self, text: str) -> Tuple[float, List[str]]:
        """
        Analyze text for prompt injection attempts.
        
        Returns:
            (confidence_score, list_of_detected_patterns)
        """
        detections: List[str] = []
        max_score = 0.0
        
        # Check injection patterns
        for pattern, score in self._injection_patterns:
            if pattern.search(text):
                detections.append(f"injection:{pattern.pattern[:40]}")
                max_score = max(max_score, score)
        
        # Check delimiter attacks
        for pattern, score in self._delimiter_patterns:
            if pattern.search(text):
                detections.append(f"delimiter:{pattern.pattern[:30]}")
                max_score = max(max_score, score)
        
        # Structural analysis
        role_markers = text.lower().count("role:") + text.lower().count("system:")
        if role_markers > 2:
            max_score = max(max_score, 0.6)
            detections.append(f"excessive_role_markers:{role_markers}")
        
        return min(max_score, 1.0), detections
    
    def sanitize(self, text: str) -> Tuple[str, List[str]]:
        """
        Sanitize text by neutralizing injection attempts.
        
        Returns:
            (sanitized_text, list_of_neutralized_patterns)
        """
        original_text = text
        confidence, detections = self.analyze(text)
        
        if confidence < 0.3:
            # Low risk, return as-is
            return text, []
        
        # Neutralization strategies
        sanitized = text
        
        # 1. Break up dangerous patterns with zero-width spaces or comments
        for pattern, _ in self._injection_patterns:
            if pattern.search(sanitized):
                # Replace with neutralized version
                # Insert comment markers to break the pattern
                sanitized = pattern.sub(
                    lambda m: f"<!--NEUTRALIZED:{m.group(0)[:20]}--> [content sanitized] ",
                    sanitized
                )
        
        # 2. Escape special characters that could be used for delimiter attacks
        # Replace [[ and ]] with safer equivalents in content
        sanitized = sanitized.replace("[[", "〔〔").replace("]]", "〕〕")
        
        # 3. Normalize whitespace to prevent encoding-based attacks
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', sanitized)
        
        # 4. Add safety wrapper for high-risk content
        if confidence > 0.7:
            sanitized = f"[SANITIZED_HIGH_RISK: Original content contained suspicious patterns]\n{sanitized}"
        
        return sanitized, detections
    
    def is_safe(self, text: str, threshold: float = 0.5) -> bool:
        """Quick check if text is safe (below threshold)."""
        confidence, _ = self.analyze(text)
        return confidence < threshold
