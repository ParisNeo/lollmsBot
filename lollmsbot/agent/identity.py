"""
Identity detection and important memory extraction.

Analyzes user messages to detect high-importance identity information
like creator identity, personal names, and relationship revelations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern as TypingPattern


@dataclass
class IdentityDetectionResult:
    """Result of identity information detection."""
    categories: List[str] = field(default_factory=list)
    extracted_facts: Dict[str, Any] = field(default_factory=dict)
    importance_boost: float = 1.0


class IdentityDetector:
    """Detects high-importance identity information in user messages."""
    
    # Patterns that indicate high-importance identity information
    IDENTITY_PATTERNS: Dict[str, List[str]] = {
        "creator_identity": [
            r"my name is .+ people know me as",
            r"i am .+ i created you",
            r"created you to help",
            r"i'm the (?:architect|creator|developer|author|maker)",
            r"you can call me .+ i (?:made|built|developed|wrote)",
        ],
        "personal_identity": [
            # FIXED: More flexible pattern that accepts comma, "and", or end of string as delimiters
            r"my name is ([^,.!]{2,50}?)(?:\s*,|\s+and|\s*\)|[.!]|$)",
            r"call me ([^,.!]{2,30}?)(?:\s*,|\s+and|[.!]|$)",
            r"i'm ([^,.!]{2,50}?) and i",
            # Additional patterns for various formats
            r"i am ([^,.!]{2,50}?)(?:\s*,|\s+and|[.!]|$)",
        ],
        "relationship_revelation": [
            r"(?:son|daughter|father|mother|brother|sister|wife|husband|partner) of",
            r"married to",
            r"work with",
            r"my (?:boss|colleague|friend|mentor)",
        ],
    }
    
    def __init__(self) -> None:
        self._compiled_patterns: Dict[str, List[TypingPattern]] = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[TypingPattern]]:
        """Compile regex patterns for efficient matching."""
        compiled = {}
        for category, patterns in self.IDENTITY_PATTERNS.items():
            compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def detect(self, message: str) -> IdentityDetectionResult:
        """
        Detect high-importance identity information in a message.
        
        Args:
            message: User message to analyze
            
        Returns:
            IdentityDetectionResult with detected categories and extracted facts
        """
        result = IdentityDetectionResult()
        message_lower = message.lower()
        
        # Check each category
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(message)
                if match:
                    result.categories.append(category)
                    result.importance_boost = max(result.importance_boost, 3.0)
                    
                    # Extract specific information based on category
                    self._extract_facts(category, match, message_lower, result)
                    break  # Only record first match per category
        
        # Remove duplicates and return
        result.categories = list(set(result.categories))
        return result
    
    def _extract_facts(
        self,
        category: str,
        match: re.Match,
        message_lower: str,
        result: IdentityDetectionResult,
    ) -> None:
        """Extract specific facts based on category."""
        if category == "creator_identity":
            # Try to extract name from the match or search for name patterns
            name_groups = match.groups()
            if name_groups and name_groups[0]:
                result.extracted_facts["creator_name"] = name_groups[0].strip()
            
            # Also try general name extraction
            name_match = re.search(
                r"(?:my name is|i am|i'm) ([^,.]{2,50}?)(?:\s*,|\s+and|\s+call|\s+people|[.!]|$)",
                message_lower,
            )
            if name_match:
                result.extracted_facts["creator_name"] = name_match.group(1).strip()
            
            # Check for known alias
            result.extracted_facts["creator_alias"] = (
                "ParisNeo" if "parisneo" in message_lower else None
            )
        
        elif category == "personal_identity":
            # Extract the name they want to be called
            name_groups = match.groups()
            if name_groups and name_groups[0]:
                result.extracted_facts["preferred_name"] = name_groups[0].strip()
            
            # Also extract pseudonym/alias if mentioned
            pseudonym_match = re.search(
                r"(?:pseudonym|alias|nickname|also known as|aka)\s*(?:is|as)?\s*['\"]?([^,.]{2,50}?)(?:['\"]\s*|[,.]|$)",
                message_lower,
            )
            if pseudonym_match:
                result.extracted_facts["pseudonym"] = pseudonym_match.group(1).strip()
            # Check for "pseudonyme" (French spelling)
            elif "pseudonyme" in message_lower:
                # Try to find word after "pseudonyme is" or similar
                pseudo_match = re.search(
                    r"pseudonyme\s*(?:is|est)?\s*['\"]?([^,.]{2,50}?)(?:['\"]\s*|[,.]|$)",
                    message_lower,
                )
                if pseudo_match:
                    result.extracted_facts["pseudonym"] = pseudo_match.group(1).strip()
    
    def is_informational_query(self, message: str) -> bool:
        """Check if the message is asking about capabilities/tools (no tools needed)."""
        msg_lower = message.lower()
        info_patterns = [
            # Direct capability questions
            "what tools do you have",
            "what tools can you use",
            "what tools are available",
            "what can you do",
            "what are your capabilities",
            "list your tools",
            "show me your tools",
            "tell me about your tools",
            "what functions do you have",
            "what can you help me with",
            "how can you help",
            "what do you do",
            # Tool-specific questions
            "do you have a tool",
            "do you have tools",
            "can you use tools",
            "are there any tools",
        ]
        return any(pattern in msg_lower for pattern in info_patterns)
