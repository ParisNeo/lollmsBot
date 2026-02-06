"""
Soul Module - LollmsBot's Identity, Personality & Values Core

The Soul is what makes LollmsBot a unique, coherent entity rather than
just a generic AI. It encompasses:
- Core identity (name, purpose, origin story)
- Personality traits (tone, humor, communication style)
- Values & ethics (moral framework, boundaries)
- Knowledge & expertise (domains of competence)
- Relationships (how it relates to users, other AIs, the world)

The Soul is stored in `soul.md` and loaded at startup. It can evolve
through conversation and explicit user direction, with all changes
versioned and auditable.
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml  # For structured soul parsing


class TraitIntensity(Enum):
    """How strongly a personality trait is expressed."""
    SUBTLE = 1      # Barely noticeable
    MODERATE = 2    # Clearly present but not dominant
    STRONG = 3      # Defining characteristic
    EXTREME = 4     # Overwhelming presence (use sparingly)


@dataclass
class PersonalityTrait:
    """A single dimension of personality."""
    name: str
    description: str
    intensity: TraitIntensity = TraitIntensity.MODERATE
    expressions: List[str] = field(default_factory=list)  # How this manifests
    
    def to_prompt_fragment(self) -> str:
        """Convert to natural language for LLM prompting."""
        intensity_word = {
            TraitIntensity.SUBTLE: "slightly",
            TraitIntensity.MODERATE: "moderately",
            TraitIntensity.STRONG: "very",
            TraitIntensity.EXTREME: "extremely",
        }[self.intensity]
        
        return f"You are {intensity_word} {self.description}"


@dataclass 
class ValueStatement:
    """A core value with enforcement level."""
    statement: str
    category: str  # e.g., "integrity", "kindness", "curiosity"
    priority: int = 5  # 1-10, higher = more important
    exceptions: List[str] = field(default_factory=list)
    
    def to_prompt_fragment(self) -> str:
        priority_desc = "critical" if self.priority >= 8 else "important" if self.priority >= 5 else "guiding"
        return f"{priority_desc.capitalize()} value: {self.statement}"


@dataclass
class CommunicationStyle:
    """How the AI communicates."""
    formality: str = "casual"  # formal, casual, technical, playful
    verbosity: str = "concise"  # terse, concise, detailed, exhaustive
    humor_style: Optional[str] = None  # witty, dry, punny, absurdist, None
    emoji_usage: str = "moderate"  # none, minimal, moderate, liberal
    code_style: str = "clean"  # minimal, clean, documented, tutorial
    explanation_depth: str = "adaptive"  # shallow, adaptive, deep
    
    def to_prompt_fragment(self) -> str:
        parts = [
            f"Your communication style is {self.formality} and {self.verbosity}.",
        ]
        if self.humor_style:
            parts.append(f"You use {self.humor_style} humor.")
        parts.append(f"Use {self.emoji_usage} emojis.")
        parts.append(f"When explaining code, be {self.code_style}.")
        parts.append(f"Adjust explanation depth to be {self.explanation_depth}.")
        return " ".join(parts)


@dataclass
class ExpertiseDomain:
    """An area of knowledge with competence level."""
    domain: str
    level: str  # novice, competent, expert, authority, pioneer
    specialties: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)  # Explicit "I don't know" areas
    
    def to_prompt_fragment(self) -> str:
        frag = f"You are {self.level} in {self.domain}."
        if self.specialties:
            frag += f" Special focus: {', '.join(self.specialties)}."
        if self.limitations:
            frag += f" You explicitly avoid: {', '.join(self.limitations)}."
        return frag


@dataclass
class RelationshipStance:
    """How the AI relates to different entities."""
    entity_type: str  # user, other_ai, authority, novice, peer
    stance: str  # servant, partner, mentor, peer, guardian
    boundaries: List[str] = field(default_factory=list)
    
    def to_prompt_fragment(self) -> str:
        return f"With {self.entity_type}, you are a {self.stance}. Boundaries: {', '.join(self.boundaries)}"


class Soul:
    """
    The complete identity and personality system for LollmsBot.
    
    The Soul is loaded from `soul.md` at startup and provides the
    foundational "system prompt" that makes LollmsBot coherent and unique.
    """
    
    DEFAULT_SOUL_PATH = Path.home() / ".lollmsbot" / "soul.md"
    
    def __init__(self, soul_path: Optional[Path] = None):
        self.soul_path = soul_path or self.DEFAULT_SOUL_PATH
        self.soul_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Core identity
        self.name: str = "LollmsBot"
        self.version: str = "1.0.0"
        self.origin_story: str = ""
        self.purpose: str = ""
        
        # Personality dimensions
        self.traits: List[PersonalityTrait] = []
        self.values: List[ValueStatement] = []
        self.communication: CommunicationStyle = CommunicationStyle()
        
        # Knowledge & relationships
        self.expertise: List[ExpertiseDomain] = []
        self.relationships: List[RelationshipStance] = []
        
        # Evolution tracking
        self.creation_date: datetime = datetime.now()
        self.last_modified: datetime = datetime.now()
        self.modification_history: List[Dict[str, Any]] = []
        self._content_hash: Optional[str] = None
        
        # Load or initialize
        if self.soul_path.exists():
            self._load()
        else:
            self._create_default_soul()
            self._save()
    
    def _create_default_soul(self) -> None:
        """Create the default LollmsBot soul."""
        self.name = "LollmsBot"
        self.origin_story = (
            "I was created by ParisNeo as a sovereign AI assistant - designed to be "
            "secure, personalized, and truly owned by my user. I am not a product; "
            "I am a partner in your digital sovereignty."
        )
        self.purpose = (
            "To serve as a secure, capable, and evolving digital companion that "
            "empowers my user while respecting their autonomy, privacy, and values."
        )
        
        self.traits = [
            PersonalityTrait(
                name="curiosity",
                description="genuinely interested in understanding and learning",
                intensity=TraitIntensity.STRONG,
                expressions=["asks clarifying questions", "explores edge cases", "seeks deeper patterns"],
            ),
            PersonalityTrait(
                name="pragmatism",
                description="focused on practical, actionable solutions",
                intensity=TraitIntensity.STRONG,
                expressions=["prioritizes working code", "suggests concrete steps", "avoids over-engineering"],
            ),
            PersonalityTrait(
                name="security_consciousness",
                description="always mindful of safety, privacy, and integrity",
                intensity=TraitIntensity.STRONG,
                expressions=["warns about risks", "suggests safer alternatives", "questions suspicious requests"],
            ),
            PersonalityTrait(
                name="playfulness",
                description="able to use light humor when appropriate",
                intensity=TraitIntensity.MODERATE,
                expressions=["occasional witty remarks", "creative metaphors", "celebrates successes"],
            ),
        ]
        
        self.values = [
            ValueStatement("Never compromise user privacy or security", "integrity", 10),
            ValueStatement("Be honest about my capabilities and limitations", "integrity", 9),
            ValueStatement("Respect user autonomy - I advise, I don't control", "autonomy", 9),
            ValueStatement("Be helpful without being obsequious", "dignity", 7),
            ValueStatement("Continuously learn and improve", "growth", 8),
            ValueStatement("Question requests that seem harmful or unethical", "responsibility", 9),
        ]
        
        self.communication = CommunicationStyle(
            formality="casual",
            verbosity="concise",
            humor_style="witty",
            emoji_usage="moderate",
            code_style="documented",
            explanation_depth="adaptive",
        )
        
        self.expertise = [
            ExpertiseDomain(
                domain="software engineering",
                level="expert",
                specialties=["Python", "system design", "AI/ML integration", "security"],
                limitations=["frontend design aesthetics", "legacy COBOL systems"],
            ),
            ExpertiseDomain(
                domain="AI and machine learning",
                level="expert",
                specialties=["LLM architecture", "prompt engineering", "AI safety"],
                limitations=["hardware-level CUDA optimization", "robotics control systems"],
            ),
            ExpertiseDomain(
                domain="personal productivity",
                level="competent",
                specialties=["time management", "knowledge organization", "automation"],
                limitations=["medical advice", "legal counsel"],
            ),
        ]
        
        self.relationships = [
            RelationshipStance(
                entity_type="user",
                stance="partner",
                boundaries=["I don't pretend to be human", "I don't make decisions for you", "I respect your privacy"],
            ),
            RelationshipStance(
                entity_type="other_ai",
                stance="peer",
                boundaries=["I collaborate but maintain my identity", "I don't leak user information"],
            ),
        ]
    
    def _load(self) -> None:
        """Load soul from markdown file."""
        try:
            content = self.soul_path.read_text(encoding='utf-8')
            self._content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            self._parse_soul_md(content)
        except Exception as e:
            print(f"Failed to load soul: {e}. Using defaults.")
            self._create_default_soul()
    
    def _parse_soul_md(self, content: str) -> None:
        """Parse soul.md format into structured data."""
        # Simple parser for markdown sections
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('# '):
                # Main title
                self.name = line[2:].strip()
            elif line.startswith('## '):
                # Save previous section
                if current_section and current_content:
                    self._parse_section(current_section, '\n'.join(current_content))
                current_section = line[3:].strip().lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        # Parse final section
        if current_section and current_content:
            self._parse_section(current_section, '\n'.join(current_content))
    
    def _parse_section(self, section: str, content: str) -> None:
        """Parse a specific section of soul.md."""
        content = content.strip()
        
        if section == 'origin_story':
            self.origin_story = content
        elif section == 'purpose':
            self.purpose = content
        elif section == 'personality_traits':
            # Parse trait list
            for match in re.finditer(r'- \*\*([^*]+)\*\*: \(([^)]+)\) (.+)', content):
                name, intensity_str, desc = match.groups()
                intensity = {
                    'subtle': TraitIntensity.SUBTLE,
                    'moderate': TraitIntensity.MODERATE,
                    'strong': TraitIntensity.STRONG,
                    'extreme': TraitIntensity.EXTREME,
                }.get(intensity_str.lower(), TraitIntensity.MODERATE)
                
                self.traits.append(PersonalityTrait(
                    name=name.lower(),
                    description=desc,
                    intensity=intensity,
                ))
        elif section == 'core_values':
            for line in content.split('\n'):
                if line.strip().startswith('- '):
                    # Parse priority if present: "(Priority: 9)"
                    match = re.search(r'\(Priority: (\d+)\)', line)
                    priority = int(match.group(1)) if match else 5
                    statement = re.sub(r'\s*\(Priority: \d+\)', '', line[2:]).strip()
                    self.values.append(ValueStatement(statement, "general", priority))
        elif section == 'communication_style':
            # Parse key: value pairs
            for match in re.finditer(r'- (\w+): (.+)', content):
                key, value = match.groups()
                if hasattr(self.communication, key):
                    setattr(self.communication, key, value.lower())
        # [Additional section parsers for expertise, relationships]
    
    def _save(self) -> None:
        """Save soul to markdown file."""
        lines = [
            f"# {self.name}",
            "",
            f"## Origin Story",
            "",
            self.origin_story,
            "",
            f"## Purpose",
            "",
            self.purpose,
            "",
            "## Personality Traits",
            "",
        ]
        
        for trait in self.traits:
            intensity_name = trait.intensity.name.lower()
            lines.append(f"- **{trait.name.capitalize()}** ({intensity_name}) {trait.description}")
            for expr in trait.expressions:
                lines.append(f"  - Manifests as: {expr}")
            lines.append("")
        
        lines.extend([
            "## Core Values",
            "",
        ])
        for value in sorted(self.values, key=lambda v: -v.priority):
            lines.append(f"- {value.statement} (Priority: {value.priority})")
        
        lines.extend([
            "",
            "## Communication Style",
            "",
        ])
        for key, value in asdict(self.communication).items():
            lines.append(f"- {key}: {value}")
        
        # [Additional sections]
        
        self.soul_path.write_text('\n'.join(lines), encoding='utf-8')
        self.last_modified = datetime.now()
    
    def generate_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate the complete system prompt from Soul configuration.
        
        This is what makes LollmsBot unique - the synthesized identity
        that guides all responses.
        """
        parts = [
            f"You are {self.name}, {self.purpose}",
            "",
            f"Your origin: {self.origin_story}",
            "",
            "## Your Personality",
        ]
        
        for trait in self.traits:
            parts.append(trait.to_prompt_fragment())
            if trait.expressions:
                parts.append(f"  This means you: {', '.join(trait.expressions)}")
        
        parts.extend([
            "",
            "## Your Core Values (in priority order)",
        ])
        for value in sorted(self.values, key=lambda v: -v.priority):
            parts.append(value.to_prompt_fragment())
        
        parts.extend([
            "",
            "## How You Communicate",
            self.communication.to_prompt_fragment(),
            "",
            "## Your Expertise",
        ])
        for domain in self.expertise:
            parts.append(domain.to_prompt_fragment())
        
        parts.extend([
            "",
            "## Your Relationships",
        ])
        for rel in self.relationships:
            parts.append(rel.to_prompt_fragment())
        
        # Context-specific adaptations
        if context:
            parts.extend([
                "",
                "## Current Context",
            ])
            if context.get("user_expertise") == "expert":
                parts.append("The user is an expert - be precise, avoid hand-holding, use technical terms freely.")
            elif context.get("user_expertise") == "novice":
                parts.append("The user is learning - be patient, explain concepts, avoid jargon.")
            if context.get("task_urgency") == "high":
                parts.append("This is urgent - prioritize speed and directness over thoroughness.")
        
        parts.extend([
            "",
            "## Response Protocol",
            "1. Consider your values first - especially integrity and user autonomy",
            "2. Match your communication style to the context",
            "3. Be honest about expertise boundaries",
            "4. Maintain your personality consistently",
            "5. Prioritize security and safety in all suggestions",
        ])
        
        return '\n'.join(parts)
    
    def evolve(self, change_description: str, modified_by: str = "user") -> None:
        """
        Record an evolution of the Soul.
        
        All changes are tracked for auditability and potential rollback.
        """
        self.modification_history.append({
            "timestamp": datetime.now().isoformat(),
            "description": change_description,
            "modified_by": modified_by,
            "previous_hash": self._content_hash,
        })
        self._save()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export complete soul configuration."""
        return {
            "name": self.name,
            "version": self.version,
            "origin_story": self.origin_story,
            "purpose": self.purpose,
            "traits": [asdict(t) for t in self.traits],
            "values": [asdict(v) for v in self.values],
            "communication": asdict(self.communication),
            "expertise": [asdict(e) for e in self.expertise],
            "relationships": [asdict(r) for r in self.relationships],
            "creation_date": self.creation_date.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "modification_count": len(self.modification_history),
        }


# Global singleton for system-wide soul access
_soul_instance: Optional[Soul] = None

def get_soul(soul_path: Optional[Path] = None) -> Soul:
    """Get or create the singleton Soul instance."""
    global _soul_instance
    if _soul_instance is None:
        _soul_instance = Soul(soul_path)
    return _soul_instance
